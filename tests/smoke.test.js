import assert from 'node:assert/strict';
import { readdirSync, readFileSync, statSync } from 'node:fs';
import { join } from 'node:path';
import { test } from 'node:test';
import { fileURLToPath } from 'node:url';
import { makeRng } from '../src/rng.js';
import * as signal from '../src/signal.js';
import { AXIS_LABELS, decimatePeaks, drawBer, drawConstellation, drawSeries, drawSpectrum } from '../src/ui/draw.js';
import { formatDt, mountField, softwareBackendLabel } from '../src/ui/field-panel.js';
import { courantDt } from '../src/fdtd.js';
import * as fdtdLive from '../src/fdtd.js';
import { createBus } from '../src/ui/bus.js';
import { mountSignal } from '../src/ui/signal-panel.js';
import {
  formatBer,
  mainCursor,
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
import { fdtd, gpu, rng, signal as mockSignal } from './dev-mocks.js';

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
    if (!name.endsWith('.js')) continue;
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
  assert.match(link, /demodulate\(\s*aligned,\s*\{[^}]*offset[^}]*nSymbols/);
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
  assert.equal(formatBer(0, 8192), '< 1/8192');
});

test('channel path passes the main tap, symbol count, and signal-region power', () => {
  const h = new Float64Array([0, 0, 0, 1]);
  assert.equal(mainTap(h), 3);
  assert.equal(mainTap(new Float64Array([1])), 0);
  assert.equal(mainTap(new Float64Array([-3, 1])), 0);
  const negativeCursor = mainCursor(new Float64Array([0, -2]));
  assert.equal(negativeCursor.index, 1);
  assert.ok(negativeCursor.re < 0 && negativeCursor.im === 0);

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

  for (const scheme of ['bpsk', 'qpsk']) {
    const nBits = 20000;
    const negative = runChannel(signal, {
      bits: signal.bitsRandom(nBits, makeRng(20261121)),
      scheme,
      sps: 8,
      h: new Float64Array([0, 0, -1]),
      ebn0dB: 6,
      rng: makeRng(99),
    });
    const theory = signal.berTheory(scheme, 6);
    const errors = negative.errorRate * negative.txBits.length;
    const expected = theory * negative.txBits.length;
    const z = Math.abs(errors - expected) / Math.sqrt(Math.max(expected, 1));
    assert.ok(z < 4, `${scheme} negative tap z=${z.toFixed(2)} BER ${negative.errorRate}`);
    assert.ok(negative.errorRate < 0.05, `${scheme} negative tap BER ${negative.errorRate}`);
    assert.ok(negative.cursor.re < 0);
  }

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
  const base = root.endsWith('/') ? root.slice(0, -1) : root;
  for (const path of walkTests(root)) {
    const rel = path.slice(base.length + 1).split('\\').join('/');
    assert.ok(readme.includes(rel), `README names ${rel}`);
  }
  assert.ok(statSync(join(root, 'tests/rng.test.js')).isFile());
  assert.ok(statSync(join(root, 'tests/signal.test.js')).isFile());
  assert.match(readme, /python3 -m http\.server 8000/);
  assert.match(readme, /http:\/\/localhost:8000/);
  assert.match(readme, /no license yet/);
  assert.match(readme, /https:\/\/tbirdpower3000\.github\.io\/Waves-and-Particles\//);
  assert.match(readme, /2406 errors \(BER 2\.406e-3\)/);
  assert.match(readme, /2\.3883e-3/);
  assert.match(readme, /tests\/fdtd\.test\.js/);
  assert.match(readme, /tests\/fdtd-gpu\.test\.js/);
  assert.match(readme, /S=0\.700036/);
  assert.match(readme, /0\.335932/);
  assert.match(readme, /SwiftShader/);
  assert.match(readme, /UNKNOWN on SwiftShader/);
  assert.doesNotMatch(readme, /A7[\s\S]{0,180}PASS/);
  assert.match(readme, /docs\/fdtd-perf\.md/);
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
  assert.match(ci, /persist-credentials:\s*false/);
  assert.match(ci, /npm ci/);
  assert.doesNotMatch(ci, /secrets\./);
  assert.doesNotMatch(ci, /node --test tests\/(?:\s|$)/);
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
      command === 'node --test'
        || command === 'npm ci'
        || command === 'node --test tests/smoke-browser.test.js'
        || command === 'bash rust/fdtd-wasm/ci.sh'
        || command.startsWith('npx playwright install'),
      command,
    );
  }
});

function words(text) {
  return text.trim().split(/\s+/).filter(Boolean);
}

function recordingCanvas(width = 320, height = 180) {
  const text = [];
  const placed = [];
  const rects = [];
  const arcs = [];
  const ctx = {
    setTransform() {},
    clearRect() {},
    fillRect() {},
    beginPath() {},
    moveTo() {},
    lineTo() {},
    closePath() {},
    stroke() {},
    fill() {},
    save() {},
    restore() {},
    strokeRect(x, y, w, h) { rects.push({ x, y, w, h }); },
    arc(x, y, radius) { arcs.push({ x, y, r: radius }); },
    fillText(value, x, y) {
      text.push(String(value));
      placed.push({ value: String(value), x, y });
    },
  };
  return {
    text,
    placed,
    rects,
    arcs,
    canvas: {
      width,
      height,
      clientWidth: width,
      clientHeight: height,
      getBoundingClientRect() { return { width, height, left: 0, top: 0 }; },
      getContext() { return ctx; },
    },
  };
}

test('plain labels, tooltips, and the flagged readout fixes', () => {
  const html = read('index.html');
  assert.match(html, />2D wave \+ radio link lab</);
  assert.match(html, /A top-down view of a radio wave spreading across a 2D grid/);
  assert.match(html, /What each receiver probe "heard" over time/);
  assert.match(html, /Digital data is sent through that channel with added noise/);
  assert.match(html, />Record pulse response</);
  assert.match(html, /aria-label="Probe 1 pulse response"/);
  assert.match(html, /aria-label="Probe 4 pulse response"/);
  assert.doesNotMatch(html, /Record impulse|impulse response/);
  assert.match(html, />Signal-to-noise per bit</);
  assert.match(html, />Bit error rate —</);
  assert.match(html, />Wave field</);
  assert.match(html, />Radio channel</);
  assert.match(html, />Digital link test</);
  assert.match(html, /Received signal vs time/);
  assert.match(html, /Each cell is a small square; the view is scaled to fit/);
  assert.match(html, />Simulator unavailable</);
  assert.match(html, /A software renderer such as SwiftShader, llvmpipe, or softpipe is unverified/);

  const titles = [...html.matchAll(/\btitle="([^"]*)"/g)].map((match) => match[1]
    .replace(/&quot;/g, '"')
    .replace(/&lt;/g, '<')
    .replace(/&amp;/g, '&'));
  assert.ok(titles.length >= 40, `title count ${titles.length}`);
  for (const title of titles) {
    assert.ok(words(title).length <= 25, title);
  }
  const buttons = [...html.matchAll(/<button\b[^>]*>([^<]+)<\/button>/g)].map((match) => match[1].trim());
  for (const label of buttons) assert.ok(words(label).length <= 4, label);

  const field = read('src/ui/field-panel.js');
  assert.match(field, /cells per wavelength must be 8–64/);
  assert.match(field, /recording pulse response/);
  assert.doesNotMatch(field, /frequency stays between 8 and 64/);
  assert.equal(formatDt(2.335e-9), '2.335 ns');
  assert.equal(formatDt(courantDt(1, 1)), '2.335 ns');
  assert.equal(formatDt(Number.NaN), '—');

  const signal = read('src/ui/signal-panel.js');
  assert.match(signal, /SNR \(per sample\), dB/);
  assert.match(signal, /Theory: AWGN, no multipath/);
  assert.match(signal, /Probe channel: the line ignores echoes\./);
  assert.match(signal, /No chirp error curve/);
  assert.match(signal, /\|s\| = √2/);

  const drawn = new Set();
  const probe = recordingCanvas();
  drawSeries(probe.canvas, [0.2, -0.4, 0.1]);
  const spectrum = recordingCanvas();
  drawSpectrum(spectrum.canvas, [0, -12, -3]);
  const ber = recordingCanvas();
  drawBer(ber.canvas, {
    theory: [{ ebn0dB: 6, ber: 1e-3 }],
    legend: AXIS_LABELS.berLegend,
    note: 'Probe channel: the line ignores echoes.',
  });
  const chirpBer = recordingCanvas();
  drawBer(chirpBer.canvas, { note: 'No chirp error curve' });
  const bpsk = recordingCanvas();
  drawConstellation(bpsk.canvas, [{ i: 1, q: 0 }], { circleLabel: AXIS_LABELS.unitCircle, idealRadius: 1 });
  const qpsk = recordingCanvas();
  drawConstellation(qpsk.canvas, [{ i: 1, q: 1 }], { circleLabel: AXIS_LABELS.qpskCircle, idealRadius: Math.SQRT2 });
  for (const sample of [probe, spectrum, ber, chirpBer, bpsk, qpsk]) {
    for (const line of sample.text) drawn.add(line);
  }
  for (const line of [
    'Time (steps, ns)',
    AXIS_LABELS.probeY,
    AXIS_LABELS.probeScale,
    AXIS_LABELS.spectrumCycles,
    AXIS_LABELS.spectrumX,
    AXIS_LABELS.spectrumY,
    AXIS_LABELS.inPhase,
    AXIS_LABELS.quadrature,
    AXIS_LABELS.unitCircle,
    AXIS_LABELS.qpskCircle,
    AXIS_LABELS.berX,
    AXIS_LABELS.berY,
    AXIS_LABELS.berLegend,
    ...AXIS_LABELS.berDecades,
    ...AXIS_LABELS.berTicks,
    'No chirp error curve',
    'Probe channel: the line ignores echoes.',
  ]) {
    assert.ok(drawn.has(line), line);
  }
  const bpskRadius = circleDataRadius(bpsk.arcs, { i: 1, q: 0 });
  const qpskRadius = circleDataRadius(qpsk.arcs, { i: 1, q: 1 });
  assert.ok(Math.abs(bpskRadius - 1) < 1e-6, `BPSK circle data radius ${bpskRadius}`);
  assert.ok(Math.abs(qpskRadius - Math.SQRT2) < 1e-6, `QPSK circle data radius ${qpskRadius}`);
  const legend = ber.placed.find((item) => item.value === AXIS_LABELS.berLegend);
  const frame = ber.rects[0];
  assert.ok(legend && frame && legend.y < frame.y - 1, 'BER legend overlaps the plot top');
  assertProbeChannelNote();
});

function circleDataRadius(arcs, point) {
  const circle = arcs.find((arc) => arc.r > 2);
  const dot = arcs.find((arc) => arc.r > 0 && arc.r <= 2);
  assert.ok(circle && dot, 'constellation circle or symbol dot was not drawn');
  const scale = Math.hypot(dot.x - circle.x, dot.y - circle.y) / Math.hypot(point.i, point.q);
  return circle.r / scale;
}

function assertProbeChannelNote() {
  const nodes = new Map();
  function createElement(tag) {
    const node = {
      tag,
      id: '',
      value: '',
      textContent: '',
      title: '',
      disabled: false,
      dataset: {},
      style: {},
      children: [],
      options: [],
      listeners: {},
      parentElement: null,
      classList: { add() {}, remove() {} },
      setAttribute() {},
      getAttribute() { return null; },
      addEventListener(type, fn) { (this.listeners[type] ||= []).push(fn); },
      removeEventListener() {},
      append(...kids) {
        this.children.push(...kids);
        this.options.push(...kids);
      },
      replaceChildren(...kids) {
        this.children = [...kids];
        this.options = [...kids];
      },
      getContext() { return this._ctx; },
      getBoundingClientRect() { return { width: 320, height: 180, left: 0, top: 0 }; },
      width: 320,
      height: 180,
      clientWidth: 320,
      clientHeight: 180,
    };
    return node;
  }
  function add(id, tag, extra = {}) {
    const node = createElement(tag);
    node.id = id;
    Object.assign(node, extra);
    nodes.set(id, node);
    return node;
  }
  const texts = [];
  const arcs = [];
  const ctx = {
    setTransform() {}, clearRect() {}, fillRect() {}, strokeRect() {},
    beginPath() {}, moveTo() {}, lineTo() {}, closePath() {},
    stroke() {}, fill() {}, save() {}, restore() {},
    arc(x, y, r) { arcs.push({ x, y, r }); },
    fillText(value) { texts.push(String(value)); },
  };
  for (const id of ['signal-state', 'signal-note', 'ebn0-out', 'ebn0-label', 'ebn0-unit', 'ebn0-caption', 'ber-readout']) {
    add(id, 'span');
  }
  add('scheme', 'select', { value: 'bpsk' });
  add('channel-source', 'select', { value: 'identity' });
  add('seed', 'input', { value: '1' });
  add('ebn0', 'input', { value: '6' });
  add('run-link', 'button');
  add('sweep-ber', 'button');
  for (const id of ['constellation', 'ber-plot', 'spectrum']) {
    const canvas = add(id, 'canvas');
    canvas._ctx = ctx;
    canvas.parentElement = createElement('div');
  }
  const previous = { document: globalThis.document, ResizeObserver: globalThis.ResizeObserver };
  globalThis.document = {
    getElementById: (id) => nodes.get(id) ?? null,
    createElement,
    querySelectorAll: () => [],
    addEventListener() {},
  };
  globalThis.ResizeObserver = class { observe() {} disconnect() {} };
  try {
    const bus = createBus();
    mountSignal(document, bus, { seed: 1, ebn0: 6 });
    bus.emit('impulse', { probes: [{ samples: [1, 0.2, 0] }] });
    const channel = nodes.get('channel-source');
    channel.value = '0';
    for (const fn of channel.listeners.change ?? []) fn();
    assert.ok(texts.includes('Probe channel: the line ignores echoes.'), texts.join(' | '));
    const bpskCircle = arcs.filter((arc) => arc.r > 2).at(-1);
    texts.length = 0;
    arcs.length = 0;
    const scheme = nodes.get('scheme');
    scheme.value = 'qpsk';
    for (const fn of scheme.listeners.change ?? []) fn();
    const qpskCircle = arcs.find((arc) => arc.r > 2);
    assert.ok(bpskCircle && qpskCircle, 'scheme switch did not draw a reference circle');
    const ratio = qpskCircle.r / bpskCircle.r;
    assert.ok(Math.abs(ratio - 1.2) < 1e-6, `QPSK/BPSK circle pixel ratio ${ratio}`);
  } finally {
    globalThis.document = previous.document;
    globalThis.ResizeObserver = previous.ResizeObserver;
  }
}

test('software renderers are named, and the perf note stays with field-core', () => {
  assert.equal(softwareBackendLabel('llvmpipe'), 'software renderer (llvmpipe)');
  assert.equal(softwareBackendLabel('softpipe Mesa'), 'software renderer (softpipe)');
  assert.equal(softwareBackendLabel('ANGLE (Google, SwiftShader)'), 'software renderer (SwiftShader)');
  assert.equal(softwareBackendLabel('Generic Adapter', { fallback: true }), 'software renderer (software)');
  assert.equal(softwareBackendLabel('NVIDIA'), '');
  assert.match(read('docs/CONTRACT.md'), /Field-core agent \(A\) owns `docs\/fdtd-perf\.md`/);
  assert.doesNotMatch(read('src/ui/field-panel.js'), /software \(SwiftShader\)/);
});

function linkStats(samples) {
  const h = Float64Array.from(samples);
  const bits = signal.bitsRandom(512, makeRng(20261121));
  const result = runChannel(signal, {
    bits,
    scheme: 'bpsk',
    sps: 8,
    h,
    ebn0dB: 6,
    rng: makeRng(7),
  });
  return { offset: result.offset, pRef: result.pRef };
}

function installFieldDom() {
  const ctx = {
    fillStyle: '',
    strokeStyle: '',
    font: '',
    lineWidth: 1,
    imageSmoothingEnabled: true,
    setTransform() {},
    clearRect() {},
    fillRect() {},
    strokeRect() {},
    beginPath() {},
    moveTo() {},
    lineTo() {},
    closePath() {},
    stroke() {},
    fill() {},
    arc() {},
    fillText() {},
    save() {},
    restore() {},
    setLineDash() {},
    drawImage() {},
    putImageData() {},
    createImageData(w, h) {
      return { data: new Uint8ClampedArray(Math.max(1, w) * Math.max(1, h) * 4), width: w, height: h };
    },
    getImageData() {
      return { data: new Uint8ClampedArray(4), width: 1, height: 1 };
    },
  };
  const nodes = new Map();
  function createElement(tag) {
    return {
      tag,
      id: '',
      dataset: {},
      style: {},
      disabled: false,
      value: '',
      textContent: '',
      title: '',
      children: [],
      attributes: {},
      listeners: {},
      classList: { add() {}, remove() {} },
      setAttribute(name, value) { this.attributes[name] = String(value); },
      getAttribute(name) { return this.attributes[name]; },
      addEventListener(type, fn) { (this.listeners[type] ||= []).push(fn); },
      removeEventListener() {},
      append(...kids) { this.children.push(...kids); },
      replaceChildren(...kids) { this.children = [...kids]; },
      setPointerCapture() {},
      getBoundingClientRect() { return { width: 320, height: 180, left: 0, top: 0, right: 320, bottom: 180 }; },
      getContext(kind) { return kind === 'webgl' || kind === 'webgl2' ? null : ctx; },
      width: 320,
      height: 180,
      clientWidth: 320,
      clientHeight: 180,
    };
  }
  for (const id of ['panel-field', 'field-canvas', 'field-state', 'field-readout', 'eps-r', 'play', 'pause', 'use-gpu', 'preset', 'freq', 'clear-geometry', 'reset', 'backend-label']) {
    const node = createElement(id === 'field-canvas' ? 'canvas' : 'button');
    node.id = id;
    if (id === 'eps-r' || id === 'freq') node.value = id === 'freq' ? '24' : '4';
    if (id === 'preset') node.value = 'free';
    nodes.set(id, node);
  }
  const previous = {
    document: globalThis.document,
    ResizeObserver: globalThis.ResizeObserver,
    requestAnimationFrame: globalThis.requestAnimationFrame,
  };
  globalThis.document = {
    getElementById: (id) => nodes.get(id) ?? null,
    createElement,
    querySelectorAll: () => [],
    addEventListener() {},
    removeEventListener() {},
  };
  globalThis.ResizeObserver = class { observe() {} disconnect() {} };
  globalThis.requestAnimationFrame = (fn) => setTimeout(fn, 0);
  return {
    restore() {
      globalThis.document = previous.document;
      globalThis.ResizeObserver = previous.ResizeObserver;
      globalThis.requestAnimationFrame = previous.requestAnimationFrame;
    },
  };
}

test('a GPU record waits for the backend switch and matches the CPU impulse', async () => {
  const dom = installFieldDom();
  const previousGpu = globalThis.navigator?.gpu;
  try {
    if (globalThis.navigator) globalThis.navigator.gpu = undefined;
    let gpuSteps = 0;
    const gpuNs = {
      async createGpuStepper(grid) {
        const twin = structuredClone(grid);
        if (twin.mode === 'tmz') twin.Ez = twin.field;
        else twin.u = twin.field;
        await new Promise((resolve) => setTimeout(resolve, 0));
        return {
          backend: 'webgpu',
          step(n = 1) {
            gpuSteps += n;
            fdtdLive.step(twin, n);
          },
          async readField() { return fdtdLive.snapshot(twin); },
          readProbes() { return twin.probes; },
          destroy() {},
        };
      },
    };
    const contract = {
      fdtd: { ready: true, ns: fdtdLive },
      gpu: { ready: false, ns: gpuNs },
    };
    const field = mountField(document, createBus(), { nx: 32, ny: 32, pml: 4 });
    await field.setContract(contract);
    const readoutEl = document.getElementById('field-readout');
    const readout = readoutEl.children.map((node) => node.textContent).join(' ');
    assert.match(readout, /Current mode: Electromagnetic mode/);
    assert.match(readout, /Last action:/);
    assert.match(readout, /Simulation speed:/);
    assert.doesNotMatch(readout, /\btmz\b/);
    const speedTip = readoutEl.children.find((node) => typeof node === 'object' && String(node.textContent).startsWith('Simulation speed'));
    assert.match(speedTip.title, /inside the solver step/);
    assert.doesNotMatch(speedTip.title, /real time/);
    await field.recordImpulse();
    const cpu = document.getElementById('field-canvas').__recordedProbes;
    assert.ok(cpu?.[0]?.samples?.length > 16, 'CPU record produced no impulse');
    assert.equal(gpuSteps, 0);

    contract.gpu.ready = true;
    gpuSteps = 0;
    document.getElementById('use-gpu').listeners.click[0]();
    await field.recordImpulse();
    const gpu = document.getElementById('field-canvas').__recordedProbes;
    assert.equal(gpuSteps, 64, `GPU stepper took ${gpuSteps} of 64 steps`);
    assert.equal(gpu[0].samples.length, cpu[0].samples.length);
    for (let i = 0; i < cpu[0].samples.length; i++) {
      assert.equal(gpu[0].samples[i], cpu[0].samples[i], `sample ${i}`);
    }
    const cpuLink = linkStats(cpu[0].samples);
    const gpuLink = linkStats(gpu[0].samples);
    assert.equal(gpuLink.offset, cpuLink.offset);
    assert.equal(gpuLink.pRef, cpuLink.pRef);
    await field.recordImpulse();
    const again = document.getElementById('field-canvas').__recordedProbes;
    const againLink = linkStats(again[0].samples);
    assert.equal(againLink.offset, cpuLink.offset);
    assert.equal(againLink.pRef, cpuLink.pRef);
  } finally {
    if (globalThis.navigator) globalThis.navigator.gpu = previousGpu;
    await new Promise((resolve) => setTimeout(resolve, 0));
    dom.restore();
  }
});
