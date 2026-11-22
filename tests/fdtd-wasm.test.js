import assert from 'node:assert/strict';
import test from 'node:test';
import { C0 } from '../src/fdtd.js';
import { createFDTD, loadWasmFDTD } from '../src/wasm/index.js';

function log(line) {
  console.log(line);
}

function relL2(ref, other) {
  let num = 0;
  let den = 0;
  for (let i = 0; i < ref.length; i++) {
    const d = other[i] - ref[i];
    num += d * d;
    den += ref[i] * ref[i];
  }
  return Math.sqrt(num / Math.max(den, 1e-300));
}

function dftMag(samples, dt, f0, i1) {
  let re = 0;
  let im = 0;
  const w = 2 * Math.PI * f0 * dt;
  for (let n = 0; n < i1; n++) {
    const phase = w * (n + 1);
    re += samples[n] * Math.cos(phase);
    im += samples[n] * Math.sin(phase);
  }
  return Math.hypot(re, im);
}

function discreteGamma(courant, cellsPerLambdaDielectric, epsR = 4) {
  const n = Math.sqrt(epsR);
  const nFree = cellsPerLambdaDielectric * n;
  const s = Math.sin((Math.PI * courant) / nFree);
  const sin1 = s / courant;
  const sin2 = s / (courant / n);
  const c1 = Math.sqrt(1 - sin1 * sin1);
  const c2 = Math.sqrt(1 - sin2 * sin2);
  return (c2 - n * c1) / (c2 + n * c1);
}

function lineReflect(api, { pec = false, epsR = 0, cellsDielectric = 25, cellsFree = 40 }) {
  const dx = 1;
  const nFree = epsR ? cellsDielectric * Math.sqrt(epsR) : cellsFree;
  const f0 = C0 / (nFree * dx);
  const pml = 16;
  const src = 180;
  const probe = 230;
  const wall = 290;
  const nx = 360;
  const ny = 4;
  const spread = 1.25 / f0;
  const t0 = 2.6 * spread;
  const nSteps = 700;
  const make = (on) => {
    const g = api.createGrid({ nx, ny, dx, dy: dx, mode: 'tmz', pml, periodicY: true });
    for (let j = 0; j < ny; j++) {
      api.addSource(g, { i: src, j, kind: 'gaussian', f0, t0, spread, amp: 1 });
    }
    if (on && pec) api.setMaterial(g, { x0: wall, y0: 0, x1: nx - 1, y1: ny - 1, pec: 1 });
    if (on && epsR) api.setMaterial(g, { x0: wall, y0: 0, x1: nx - 1, y1: ny - 1, epsR });
    api.addProbe(g, { i: probe, j: 0 });
    api.step(g, nSteps);
    return g;
  };
  const ref = make(false);
  const tot = make(true);
  const diff = ref.probes[0].samples.map((v, i) => tot.probes[0].samples[i] - v);
  const echoCells = src - pml + (probe - pml);
  const iEcho = Math.floor((t0 + (echoCells * dx) / C0) / ref.dt) - 8;
  const i1 = Math.max(8, Math.min(diff.length, iEcho));
  const mag = dftMag(diff, ref.dt, f0, i1) / dftMag(ref.probes[0].samples, ref.dt, f0, i1);
  return { mag, dt: ref.dt, nFree };
}

test('wasm loads, and a failed load falls back to the JS solver', async () => {
  const wasm = await loadWasmFDTD();
  assert.equal(wasm.backend, 'wasm');
  const failed = await createFDTD({
    load: async () => {
      throw new Error('csp blocked wasm');
    },
  });
  assert.equal(failed.backend, 'js');
  assert.match(failed.fallbackReason, /csp blocked wasm/);
  const grid = failed.createGrid({ nx: 12, ny: 10, dx: 1, dy: 1, pml: 2 });
  failed.addSource(grid, { i: 4, j: 4, kind: 'sine', f0: 1e6, amp: 1 });
  failed.step(grid, 3);
  assert.equal(grid.steps, 3);
  const ok = await createFDTD();
  assert.equal(ok.backend, 'wasm');
  assert.equal(ok.fallbackReason, null);
});

test('WASM field matches src/fdtd.js within 1e-6 after 100, 500 and 2000 steps', async () => {
  const js = await import('../src/fdtd.js');
  const wasm = await loadWasmFDTD();
  const scenes = [
    { mode: 'tmz', periodicY: false },
    { mode: 'tmz', periodicY: true },
    { mode: 'scalar', periodicY: false },
  ];
  for (const scene of scenes) {
    const build = (api) => {
      const g = api.createGrid({
        nx: 48, ny: 36, dx: 1, dy: 1.5, mode: scene.mode, pml: 6, periodicY: scene.periodicY,
      });
      api.setMaterial(g, { x0: 18, y0: 8, x1: 30, y1: 24, epsR: 4, sigma: 0.001 });
      api.setMaterial(g, { x0: 10, y0: 4, x1: 12, y1: 28, pec: 1 });
      const f0 = C0 / 16;
      api.addSource(g, { i: 8, j: 16, kind: 'gaussian', f0, t0: 1.2 / f0, spread: 0.5 / f0, amp: 1 });
      api.addSource(g, { i: 20, j: 10, kind: 'sine', f0: f0 / 2, amp: 0.3 });
      api.addProbe(g, { i: 22, j: 18 });
      return g;
    };
    const a = build(js);
    const b = build(wasm);
    assert.equal(a.dt, b.dt);
    let done = 0;
    for (const n of [100, 500, 2000]) {
      js.step(a, n - done);
      wasm.step(b, n - done);
      done = n;
      const field = relL2(a.field, b.field64());
      const probe = relL2(a.probes[0].samples, b.probes[0].samples);
      log(`${scene.mode}${scene.periodicY ? ' periodic' : ''} n=${n} relL2=${field.toExponential(3)} probe=${probe.toExponential(3)}`);
      assert.ok(field < 1e-6, `field relL2 ${field}`);
      assert.ok(probe < 1e-6, `probe relL2 ${probe}`);
    }
  }
});

test('A1 Courant dt on the WASM backend', async () => {
  const api = await loadWasmFDTD();
  const dx = 0.02;
  const dy = 0.05;
  const limit = 1 / (C0 * Math.sqrt(1 / dx ** 2 + 1 / dy ** 2));
  const dt = api.courantDt(dx, dy);
  assert.ok(dt <= limit);
  assert.ok(Math.abs(dt - 0.99 * limit) <= 1e-12 * limit);
  const grid = api.createGrid({ nx: 16, ny: 12, dx, dy, pml: 3 });
  assert.equal(grid.dt, api.courantDt(dx, dy));
  const s = (C0 * api.courantDt(1, 1)) / 1;
  assert.ok(s <= 1 / Math.sqrt(2) + 1e-15);
  log(`A1 WASM PASS S=${s.toFixed(6)}`);
});

test('A2 cylindrical spreading on the WASM backend', async () => {
  const api = await loadWasmFDTD();
  const dx = 1;
  const cellsPerLambda = 24;
  const f0 = C0 / (cellsPerLambda * dx);
  const pml = 12;
  const radius = 220;
  const nx = 2 * (radius + pml) + 1;
  const cx = (nx / 2) | 0;
  const grid = api.createGrid({ nx, ny: nx, dx, dy: dx, mode: 'tmz', pml });
  const spread = 1.6 / f0;
  const t0 = 2.8 * spread;
  api.addSource(grid, { i: cx, j: cx, kind: 'gaussian', f0, t0, spread, amp: 1 });
  const rs = [50, 80, 110, 140, 170, 200];
  for (const r of rs) api.addProbe(grid, { i: cx + r, j: cx });
  const travel = (r) => t0 + (r * dx) / C0;
  const nSteps = Math.ceil((travel(200) + 2.2 * spread) / grid.dt) + 5;
  api.step(grid, nSteps);
  const norms = rs.map((r, k) => {
    const samples = grid.probes[k].samples;
    const i0 = Math.max(0, Math.floor((travel(r) - 1.4 * spread) / grid.dt));
    const i1 = Math.min(samples.length, Math.ceil((travel(r) + 1.4 * spread) / grid.dt));
    let peak = 0;
    for (let n = i0; n < i1; n++) peak = Math.max(peak, Math.abs(samples[n]));
    return peak * Math.sqrt(r);
  });
  const mean = norms.reduce((a, b) => a + b, 0) / norms.length;
  const maxDev = Math.max(...norms.map((v) => Math.abs(v / mean - 1)));
  assert.ok(maxDev < 0.05, `spreading deviation ${(100 * maxDev).toFixed(2)}%`);

  const rho = 24;
  const margin = Math.ceil(10.5 * rho);
  const nt = 2 * (margin + 10) + 1;
  const c = (nt / 2) | 0;
  const tail = api.createGrid({ nx: nt, ny: nt, dx, dy: dx, mode: 'tmz', pml: 10 });
  const spreadT = 2.5 * tail.dt;
  const t0t = 4 * spreadT;
  api.addSource(tail, { i: c, j: c, kind: 'gaussian', f0: 0, t0: t0t, spread: spreadT, amp: 1 });
  api.addProbe(tail, { i: c + rho, j: c });
  const tArr = (rho * dx) / C0;
  api.step(tail, Math.ceil((18 * tArr + t0t) / tail.dt));
  const samples = tail.probes[0].samples;
  const xs = [];
  const ys = [];
  let pos = 0;
  let neg = 0;
  for (let n = 0; n < samples.length; n++) {
    const t = (n + 1) * tail.dt;
    if (t < 10 * tArr || t > 18 * tArr) continue;
    const v = samples[n];
    if (v > 0) pos++;
    else if (v < 0) neg++;
    if (v === 0) continue;
    xs.push(Math.log(t));
    ys.push(Math.log(Math.abs(v)));
  }
  const m = xs.length;
  const mx = xs.reduce((a, b) => a + b, 0) / m;
  const my = ys.reduce((a, b) => a + b, 0) / m;
  let num = 0;
  let den = 0;
  for (let i = 0; i < m; i++) {
    num += (xs[i] - mx) * (ys[i] - my);
    den += (xs[i] - mx) ** 2;
  }
  const slope = num / den;
  assert.ok(slope < -0.9 && slope > -1.15, `tail slope ${slope}`);
  assert.ok(pos > neg);
  log(`A2 WASM PASS max |dev|=${(100 * maxDev).toFixed(2)}% slope=${slope.toFixed(3)}`);
});

test('A3 PEC reflection on the WASM backend', async () => {
  const api = await loadWasmFDTD();
  const { mag, nFree } = lineReflect(api, { pec: true, cellsFree: 40 });
  const err = Math.abs(mag - 1);
  assert.ok(err <= 0.02, `|R|=${mag}`);
  log(`A3 WASM PASS |R|=${mag.toFixed(6)} err=${(100 * err).toFixed(4)}% at ${nFree} cells/λ`);
});

test('A4 PML energy on the WASM backend', async () => {
  const api = await loadWasmFDTD();
  const dx = 1;
  const cells = 20;
  const f0 = C0 / (cells * dx);
  const spread = 1.2 / f0;
  const t0 = 2.5 * spread;
  const run = (n, pml) => {
    const g = api.createGrid({ nx: n, ny: n, dx, dy: dx, mode: 'tmz', pml });
    const c = (n / 2) | 0;
    api.addSource(g, { i: c, j: c, kind: 'gaussian', f0, t0, spread, amp: 1 });
    api.addProbe(g, { i: c + 18, j: c });
    const dist = c - pml;
    const back = ((dist + (dist - 18)) * dx) / C0;
    api.step(g, Math.ceil((t0 + back + 2 * spread) / g.dt));
    return g.probes[0].samples;
  };
  const small = run(80, 10);
  const large = run(220, 10);
  const n = Math.min(small.length, large.length);
  let num = 0;
  let den = 0;
  for (let i = 0; i < n; i++) {
    const d = small[i] - large[i];
    num += d * d;
    den += large[i] * large[i];
  }
  const ratio = num / den;
  assert.ok(ratio < 0.01, `energy ratio ${ratio}`);
  log(`A4 WASM PASS reflected/incident energy=${ratio.toExponential(3)}`);
});

test('A5 dielectric slab on the WASM backend', async () => {
  const api = await loadWasmFDTD();
  const cellsDielectric = 25;
  const out = lineReflect(api, { epsR: 4, cellsDielectric });
  const third = 1 / 3;
  const relThird = Math.abs(out.mag - third) / third;
  const s = (C0 * out.dt) / 1;
  const gamma = discreteGamma(s, cellsDielectric, 4);
  const relDisc = Math.abs(out.mag - Math.abs(gamma)) / Math.abs(gamma);
  assert.ok(relThird <= 0.03, `|R|=${out.mag} vs 1/3`);
  assert.ok(relDisc <= 0.005, `|R|=${out.mag} vs discrete ${gamma}`);
  log(`A5 WASM PASS |R|=${out.mag.toFixed(6)} vs 1/3 err=${(100 * relThird).toFixed(3)}%`);
});

function median(values) {
  const sorted = [...values].sort((a, b) => a - b);
  const mid = sorted.length >> 1;
  return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
}

test('A9 steps/s JS vs WASM at 256^2 and 512^2', async () => {
  const js = await import('../src/fdtd.js');
  const wasm = await loadWasmFDTD();
  const stepsFor = (n) => (n >= 512 ? 40 : 120);
  const runs = 5;
  const summary = {};
  for (const n of [256, 512]) {
    const steps = stepsFor(n);
    const make = (api) => {
      const g = api.createGrid({ nx: n, ny: n, dx: 1, dy: 1, mode: 'tmz', pml: 12 });
      const f0 = C0 / 24;
      api.addSource(g, { i: (n / 2) | 0, j: (n / 2) | 0, kind: 'gaussian', f0, t0: 2 / f0, spread: 0.8 / f0, amp: 1 });
      api.step(g, 2);
      return g;
    };
    const time = (api, g) => {
      const t0 = performance.now();
      api.step(g, steps);
      const seconds = (performance.now() - t0) / 1000;
      return steps / seconds;
    };
    const ratios = [];
    const jsRates = [];
    const wasmRates = [];
    for (let run = 0; run < runs; run++) {
      const jsRate = time(js, make(js));
      const wasmRate = time(wasm, make(wasm));
      jsRates.push(jsRate);
      wasmRates.push(wasmRate);
      ratios.push(wasmRate / jsRate);
    }
    const ratioMed = median(ratios);
    const mean = ratios.reduce((sum, value) => sum + value, 0) / ratios.length;
    const variance = ratios.reduce((sum, value) => sum + (value - mean) ** 2, 0) / (ratios.length - 1);
    summary[n] = { js: median(jsRates), wasm: median(wasmRates), ratioMed, variance };
    log(
      `A9 ${n}^2 median JS=${summary[n].js.toFixed(1)} WASM=${summary[n].wasm.toFixed(1)} ratio=${ratioMed.toFixed(3)} variance=${variance.toExponential(3)} over ${runs} runs, Node ${process.version}`,
    );
    if (process.env.WAP_WASM_PERF === '1') {
      assert.ok(ratioMed >= 0.85, `${n} median wasm/js ${ratioMed}`);
    }
  }
  log(
    `A9 SUMMARY Node ${process.version} 256=${summary[256].js.toFixed(1)}/${summary[256].wasm.toFixed(1)} ratio=${summary[256].ratioMed.toFixed(3)} var=${summary[256].variance.toExponential(3)} 512=${summary[512].js.toFixed(1)}/${summary[512].wasm.toFixed(1)} ratio=${summary[512].ratioMed.toFixed(3)} var=${summary[512].variance.toExponential(3)}`,
  );
});

test('createGrid rejects a non-integer size and a grid past the cap', async () => {
  const api = await loadWasmFDTD();
  assert.throws(
    () => api.createGrid({ nx: 3.5, ny: 16, dx: 1, dy: 1, pml: 2 }),
    /integers/,
  );
  assert.throws(
    () => api.createGrid({ nx: 4096, ny: 4096, dx: 1, dy: 1, pml: 2 }),
    /exceeds the 2048 cap/,
  );
  await assert.rejects(
    Promise.resolve().then(() => api.createGrid({ nx: 1e7, ny: 1e7, dx: 1, dy: 1, pml: 2 })),
    (err) => {
      assert.match(String(err && err.message ? err.message : err), /exceeds the 2048 cap/);
      assert.doesNotMatch(String(err && err.message ? err.message : err), /unreachable/);
      return true;
    },
  );
});
