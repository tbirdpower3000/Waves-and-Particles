import assert from 'node:assert/strict';
import { readdirSync, readFileSync, statSync } from 'node:fs';
import { join } from 'node:path';
import { test } from 'node:test';
import { fileURLToPath } from 'node:url';
import { makeRng } from '../src/rng.js';
import * as signal from '../src/signal.js';
import { decimatePeaks } from '../src/ui/draw.js';
import {
  formatBer,
  mainTap,
  regionPower,
  runChannel,
  runLinkSafely,
  symbolPoints,
} from '../src/ui/link.js';
import {
  MODULES,
  loadContract,
  missingExports,
  panelDetail,
} from '../src/ui/load-contract.js';
import {
  LIMITS,
  assertGridBudget,
  gpuBufferGate,
  readBounded,
  readLabParams,
} from '../src/ui/params.js';
import {
  addProbe,
  cellFromOffset,
  createScene,
  materialFromRect,
  rebuildGrid,
  rectFromCells,
  sourceTiming,
} from '../src/ui/scene.js';
import { NOT_READY_TEXT, renderParamNotes, reportLinkError } from '../src/ui/status.js';
import { fdtd, gpu, rng, signal as mockSignal } from '../src/ui/dev-mocks.js';

const root = fileURLToPath(new URL('..', import.meta.url));

function read(rel) {
  return readFileSync(join(root, rel), 'utf8');
}

function isTestFile(name) {
  return name.endsWith('.test.js')
    || name.endsWith('-test.js')
    || name.endsWith('_test.js')
    || name.startsWith('test-')
    || name.startsWith('test_')
    || name === 'test.js';
}

function walkTests(dir, out = []) {
  for (const entry of readdirSync(dir, { withFileTypes: true })) {
    if (entry.name === 'node_modules' || entry.name === '.git') continue;
    const path = join(dir, entry.name);
    if (entry.isDirectory()) walkTests(path, out);
    else if (isTestFile(entry.name)) out.push(path);
  }
  return out;
}

test('package.json is ESM, npm test is node --test, and there are no runtime dependencies', () => {
  const pkg = JSON.parse(read('package.json'));
  assert.equal(pkg.name, 'waves-and-particles');
  assert.equal(pkg.type, 'module');
  assert.equal(pkg.scripts.test, 'node --test');
  assert.ok(!pkg.dependencies || Object.keys(pkg.dependencies).length === 0);
  assert.deepEqual(Object.keys(pkg.devDependencies), ['playwright']);
});

test('index.html is a no-build page with Field, Channel, and Signal panels', () => {
  const html = read('index.html');
  assert.match(html, /type="module"/);
  assert.match(html, /src="src\/ui\/main\.js"/);
  assert.match(html, /href="src\/ui\/lab\.css"/);
  assert.match(html, /id="panel-field"/);
  assert.match(html, /id="panel-channel"/);
  assert.match(html, /id="panel-signal"/);
  assert.match(html, /id="field-canvas"/);
  assert.match(html, /id="constellation"/);
  assert.match(html, /id="ber-plot"/);
  assert.match(html, /id="spectrum"/);
  assert.match(html, /id="scheme"/);
  assert.match(html, /id="ebn0"/);
  assert.match(html, /rel="icon"/);
  assert.match(html, /data:image\/svg\+xml/);
  assert.match(html, /Content-Security-Policy/);
  assert.match(html, /script-src 'self'/);
  assert.doesNotMatch(html, /unsafe-inline|unsafe-eval/);
  assert.doesNotMatch(html, /cdn|unpkg|jsdelivr|googleapis|cloudflare/i);
  assert.doesNotMatch(html, /\s(?:src|href)=["']\//);
  assert.doesNotMatch(html, /fdtd\.js|signal\.js|rng\.js/);
  assert.equal(NOT_READY_TEXT, 'module not ready');
  assert.match(read('src/ui/status.js'), /module not ready/);
});

test('production UI does not statically import the parallel modules or dev mocks', () => {
  const uiDir = join(root, 'src/ui');
  for (const name of readdirSync(uiDir)) {
    if (!name.endsWith('.js') || name === 'dev-mocks.js') continue;
    const text = readFileSync(join(uiDir, name), 'utf8');
    assert.doesNotMatch(text, /dev-mocks/, name);
    assert.doesNotMatch(text, /innerHTML|insertAdjacentHTML|outerHTML|document\.write/, name);
    assert.doesNotMatch(text, /from\s+['"][^'"]*(?:fdtd\.js|signal\.js|rng\.js|gpu\/)/, name);
    assert.doesNotMatch(text, /import\s*\(\s*['"][^'"]*(?:fdtd\.js|signal\.js|rng\.js|gpu\/)/, name);
    if (name === 'load-contract.js') {
      assert.match(text, /fdtd\.js/);
      assert.match(text, /signal\.js/);
      assert.match(text, /rng\.js/);
      assert.doesNotMatch(text, /\bfrom\s+['"]/);
    }
  }
  const link = read('src/ui/link.js');
  assert.match(link, /demodulate\(\s*noisy,\s*\{[^}]*offset[^}]*nSymbols/);
  assert.match(link, /pRef/);
  const panel = read('src/ui/signal-panel.js');
  assert.match(panel, /runLinkSafely/);
  assert.doesNotMatch(panel, /demodulate\s*\(/);
  assert.doesNotMatch(panel, /\.awgn\s*\(/);
  const field = read('src/ui/field-panel.js');
  assert.match(field, /const cells = gpuBufferGate\(scene\.nx, scene\.ny\);[\s\S]{0,500}createGpuStepper/);
});

test('ui modules import under Node without a DOM', async () => {
  await import('../src/ui/main.js');
});

test('missing modules are not imported and incomplete exports are not ready', async () => {
  let imports = 0;
  const base = 'http://lab.local/src/ui/load-contract.js';
  const missing = await loadContract({
    base,
    fetchImpl: async () => { throw new Error('offline'); },
    importImpl: async () => {
      imports += 1;
      throw new Error('should not import');
    },
  });
  assert.equal(imports, 0);
  assert.equal(missing.fdtd.ready, false);
  assert.equal(missing.signal.ready, false);
  assert.equal(missing.rng.ready, false);
  assert.equal(missing.gpu.ready, false);
  assert.match(panelDetail(missing, ['signal', 'rng']), /src\/signal\.js/);

  const htmlFetch = await loadContract({
    base,
    fetchImpl: async () => ({
      ok: true,
      headers: { get: () => 'text/html' },
      text: async () => '<!DOCTYPE html>',
    }),
    importImpl: async () => {
      imports += 1;
      return {};
    },
  });
  assert.equal(imports, 0);
  assert.equal(htmlFetch.fdtd.ready, false);

  const partial = { C0: 1, courantDt() {}, createGrid() {}, setMaterial() {}, addSource() {}, addProbe() {}, step() {} };
  const fetched = [];
  const fdtdSpec = { ...MODULES.find((spec) => spec.id === 'fdtd'), request: true };
  const incomplete = await loadContract({
    modules: [fdtdSpec],
    base,
    fetchImpl: async (url) => {
      fetched.push(String(url));
      if (!String(url).endsWith('/src/fdtd.js')) {
        return { ok: false, status: 404, headers: { get: () => '' }, text: async () => '' };
      }
      return { ok: true, headers: { get: () => 'text/javascript' }, text: async () => 'export const C0 = 1;' };
    },
    importImpl: async (url) => {
      imports += 1;
      assert.match(String(url), /\/src\/fdtd\.js$/);
      return partial;
    },
  });
  assert.equal(imports, 1);
  assert.ok(fetched.some((url) => url.endsWith('/src/fdtd.js')));
  assert.equal(incomplete.fdtd.ready, false);
  assert.equal(incomplete.fdtd.state, 'incomplete');
  assert.ok(incomplete.fdtd.missing.includes('snapshot'));

  const full = {
    C0: 1,
    courantDt() {},
    createGrid() {},
    setMaterial() {},
    addSource() {},
    addProbe() {},
    step() {},
    snapshot() {},
  };
  const ready = await loadContract({
    modules: [{ ...fdtdSpec, request: true }],
    base,
    fetchImpl: async (url) => {
      if (String(url).endsWith('/src/fdtd.js')) {
        return { ok: true, headers: { get: () => 'text/javascript' }, text: async () => 'export {}' };
      }
      return { ok: false, status: 404, headers: { get: () => '' }, text: async () => '' };
    },
    importImpl: async () => full,
  });
  assert.equal(ready.fdtd.ready, true);
  assert.equal(ready.fdtd.ns, full);
  assert.deepEqual(missingExports(fdtd, MODULES.find((spec) => spec.id === 'fdtd').exports), []);
  assert.deepEqual(missingExports(gpu, MODULES.find((spec) => spec.id === 'gpu').exports), []);
  assert.deepEqual(missingExports(mockSignal, MODULES.find((spec) => spec.id === 'signal').exports), []);
  assert.deepEqual(missingExports(rng, MODULES.find((spec) => spec.id === 'rng').exports), []);
});

test('scene rebuilds only through the contract exports', () => {
  const scene = createScene();
  assert.equal(addProbe(scene, 1, 2).ok, true);
  assert.equal(addProbe(scene, 1, 2).ok, false);
  assert.equal(addProbe(scene, 3, 4).ok, true);
  assert.equal(addProbe(scene, 5, 6).ok, true);
  assert.equal(addProbe(scene, 7, 8).ok, true);
  assert.match(addProbe(scene, 9, 10).reason, /4 receiver probes maximum/);
  const rect = rectFromCells({ i: 4, j: 1 }, { i: 2, j: 6 });
  assert.deepEqual(rect, { x0: 2, y0: 1, x1: 4, y1: 6 });
  assert.equal(materialFromRect(rect, 'metal', 4).pec, true);
  assert.equal(materialFromRect(rect, 'dielectric', 4).epsR, 4);
  assert.deepEqual(cellFromOffset(0, 0, 100, 50, 10, 5), { i: 0, j: 0 });
  assert.deepEqual(cellFromOffset(99.9, 49.9, 100, 50, 10, 5), { i: 9, j: 4 });
  assert.equal(cellFromOffset(-1, 0, 100, 50, 10, 5), null);

  const timing = sourceTiming({ dx: 1, dy: 1, dt: 0.5 }, () => { throw new Error('no'); }, 3e8, 'sine');
  assert.equal(timing.kind, 'sine');
  assert.equal(timing.f0, 3e8 / 24);
  const gaussian = sourceTiming({ dx: 1, dy: 1 }, () => 2, 24, 'gaussian');
  assert.equal(gaussian.spread, 24);
  assert.equal(gaussian.t0, 96);

  const calls = [];
  const api = {
    C0: 3e8,
    courantDt() { return 1; },
    createGrid(spec) {
      calls.push(['createGrid', spec.mode]);
      return { ...spec, dt: 1, probes: [] };
    },
    setMaterial(_grid, material) { calls.push(['setMaterial', material.pec, material.epsR]); },
    addSource(_grid, source) { calls.push(['addSource', source.kind, source.i]); },
    addProbe(grid, probe) {
      calls.push(['addProbe', probe.i]);
      return grid.probes.push(probe) - 1;
    },
  };
  const built = createScene();
  built.materials.push(materialFromRect(rect, 'metal', 1));
  built.source = { i: 8, j: 9, kind: 'sine' };
  addProbe(built, 3, 3);
  rebuildGrid(built, api, 'gaussian');
  assert.deepEqual(calls.map((call) => call[0]), ['createGrid', 'setMaterial', 'addSource', 'addProbe']);
  assert.equal(calls[2][1], 'gaussian');
  assert.equal(calls[3][1], 3);

  const spike = [0, 0, 0, 4];
  const series = decimatePeaks(spike, 4);
  assert.equal(series[3], 4);
  assert.deepEqual(symbolPoints(new Float64Array([2, 2]), new Float64Array([0, 0]), 2, 0, 1), [{ i: 2, q: 0 }]);
  assert.equal(formatBer(0), '0');
});

test('channel path passes the main tap, symbol count, and signal-region power', () => {
  const h = new Float64Array([0, 0, 0, 1]);
  assert.equal(mainTap(h), 3);
  assert.equal(mainTap(new Float64Array([1])), 0);

  const calls = [];
  const api = {
    modulate: (...args) => signal.modulate(...args),
    convolveChannel: (...args) => signal.convolveChannel(...args),
    awgn(sig, opts) {
      calls.push(['awgn', opts]);
      return signal.awgn(sig, opts);
    },
    demodulate(sig, opts) {
      calls.push(['demodulate', opts]);
      return signal.demodulate(sig, opts);
    },
    ber: (...args) => signal.ber(...args),
    chirp: (...args) => signal.chirp(...args),
  };
  const rng = makeRng(7);
  const bits = signal.bitsRandom(32, rng);
  const out = runChannel(api, { bits, scheme: 'bpsk', sps: 4, h, ebn0dB: 8, rng: makeRng(8) });
  assert.equal(out.offset, 3);
  assert.equal(out.nSymbols, 32);
  assert.ok(Math.abs(out.pRef - 1) < 1e-9, `pRef ${out.pRef}`);
  const awgnOpts = calls.find((call) => call[0] === 'awgn')[1];
  const demOpts = calls.find((call) => call[0] === 'demodulate')[1];
  assert.equal(awgnOpts.pRef, out.pRef);
  assert.equal(demOpts.offset, 3);
  assert.equal(demOpts.nSymbols, 32);
  assert.equal(demOpts.scheme, 'bpsk');
  assert.equal(demOpts.sps, 4);
  assert.equal(out.rxBits.length, out.txBits.length);

  const qpskBits = signal.bitsRandom(16, makeRng(3));
  const qpsk = runChannel(api, {
    bits: qpskBits, scheme: 'qpsk', sps: 2, h: new Float64Array([1]), ebn0dB: 6, rng: makeRng(4),
  });
  assert.equal(qpsk.nSymbols, 8);
  assert.equal(qpsk.txBits.length, 16);

  const delayed = runChannel(signal, {
    bits: signal.bitsRandom(2000, makeRng(20261121)),
    scheme: 'bpsk',
    sps: 8,
    h,
    ebn0dB: 6,
    rng: makeRng(20261121),
  });
  assert.equal(delayed.offset, 3);
  assert.ok(delayed.errorRate < 0.05, `delayed BER ${delayed.errorRate}`);

  const thrown = runLinkSafely({
    ...api,
    ber() { throw new Error('bit length mismatch: 3 vs 2'); },
  }, { bits, scheme: 'bpsk', sps: 4, h, ebn0dB: 8, rng: makeRng(1) });
  assert.equal(thrown.ok, false);
  assert.match(thrown.message, /bit length mismatch: 3 vs 2/);
  const el = { textContent: '' };
  reportLinkError(el, thrown.message);
  assert.equal(el.textContent, 'bit length mismatch: 3 vs 2');

  let demodulated = 0;
  let chirpRef = null;
  runLinkSafely({
    chirp: (...args) => signal.chirp(...args),
    convolveChannel: (...args) => signal.convolveChannel(...args),
    awgn(sig, opts) {
      chirpRef = opts.pRef;
      return signal.awgn(sig, opts);
    },
    demodulate() { demodulated += 1; },
  }, {
    scheme: 'chirp', h: new Float64Array([0, 0, 1]), ebn0dB: 6, rng: makeRng(1),
    f0: 100, f1: 400, T: 0.01, fs: 2000,
  });
  assert.equal(demodulated, 0);
  assert.ok(chirpRef > 0);

  const clean = signal.convolveChannel(
    signal.modulate(signal.bitsRandom(4, makeRng(1)), { scheme: 'bpsk', sps: 2 }),
    new Float64Array(8).fill(0).map((_v, i) => (i === 0 ? 1 : 0)),
  );
  const full = regionPower({ i: clean.i, q: clean.q }, 0, clean.i.length);
  const head = regionPower(clean, 0, 8);
  assert.ok(head > full);
});

test('URL numbers are finite and inside bounds before they can size a buffer', () => {
  assert.equal(readBounded('1e9', LIMITS.nx).ok, false);
  assert.equal(readBounded('NaN', LIMITS.nx).ok, false);
  assert.equal(readBounded('512.5', LIMITS.nx).ok, false);
  assert.equal(readBounded('32', LIMITS.nx).value, 32);
  assert.equal(readBounded('', LIMITS.nx).value, 512);
  const params = readLabParams('?nx=1000000000&ny=NaN&pml=64&seed=3&ebn0=6&eps=4');
  assert.equal(params.nx, 512);
  assert.equal(params.ny, 512);
  assert.ok(params.notes.some((note) => note.startsWith('ignored nx')));
  assert.equal(assertGridBudget(params.nx, params.ny), 512 * 512);
  assert.throws(() => gpuBufferGate(1e9, 512), /outside/);
  assert.throws(() => assertGridBudget(Number.POSITIVE_INFINITY, 32), /finite/);
  const el = { textContent: '' };
  renderParamNotes(el, params.notes);
  assert.match(el.textContent, /ignored nx/);
  assert.equal(el.innerHTML, undefined);
});

test('README lists every test file and CI runs node --test on Node 20 and 22', () => {
  const readme = read('README.md');
  const ci = read('.github/workflows/ci.yml');
  for (const path of walkTests(root)) {
    const rel = path.slice(root.length + 1).split('\\').join('/');
    assert.ok(readme.includes(rel), `README lists ${rel}`);
  }
  assert.ok(statSync(join(root, 'tests/rng.test.js')).isFile());
  assert.ok(statSync(join(root, 'tests/signal.test.js')).isFile());
  assert.match(readme, /python3 -m http\.server 8000/);
  assert.match(readme, /http:\/\/localhost:8000/);
  assert.match(readme, /no license yet/);
  assert.match(readme, /https:\/\/tbirdpower3000\.github\.io\/Waves-and-Particles\//);
  assert.match(readme, /2406 errors \(BER 2\.406e-3\)/);
  assert.match(readme, /2\.3883e-3/);
  assert.match(readme, /pending CI/);
  assert.match(readme, /gprMax/);
  assert.match(readme, /node --test/);
  assert.match(readme, /Node 20/);
  assert.match(readme, /Node 22/);
  assert.match(ci, /pull_request:/);
  assert.doesNotMatch(ci, /pull_request_target/);
  assert.match(ci, /branches:\s*\[main\]/);
  assert.match(ci, /permissions:\s*\n\s*contents:\s*read/);
  assert.match(ci, /node-version:\s*\['20', '22'\]/);
  assert.match(ci, /1100x800/);
  assert.match(ci, /WAP_BROWSER_SMOKE/);
  assert.match(ci, /playwright install --with-deps chromium/);
  assert.doesNotMatch(ci, /secrets\./);
  assert.doesNotMatch(ci, /node --test tests/);
  assert.doesNotMatch(ci, /node tests\//);
  const uses = [...ci.matchAll(/uses:\s*(\S+)/g)].map((match) => match[1]);
  assert.ok(uses.length >= 2);
  for (const action of uses) {
    assert.match(action, /@[0-9a-f]{40}$/);
  }
  const runs = [...ci.matchAll(/^\s*run:\s*(.+)$/gm)].map((match) => match[1].trim());
  assert.ok(runs.includes('node --test'));
  for (const command of runs) {
    assert.ok(
      command === 'node --test' || command === 'npm install' || command.startsWith('npx playwright install'),
      command,
    );
  }
});
