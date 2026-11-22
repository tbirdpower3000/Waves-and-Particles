import assert from "node:assert/strict";
import test from "node:test";
import {
  C0,
  courantDt,
  createGrid,
  setMaterial,
  addSource,
  addProbe,
  step,
} from "../src/fdtd.js";

function log(line) {
  console.log(line);
}

function relMetrics(ref, other) {
  let num = 0;
  let den = 0;
  let peak = 0;
  let maxAbs = 0;
  for (let i = 0; i < ref.length; i++) {
    const d = other[i] - ref[i];
    num += d * d;
    den += ref[i] * ref[i];
    peak = Math.max(peak, Math.abs(ref[i]));
    maxAbs = Math.max(maxAbs, Math.abs(d));
  }
  return {
    relL2: Math.sqrt(num / Math.max(den, 1e-300)),
    maxOverPeak: maxAbs / Math.max(peak, 1e-300),
  };
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

// Schneider eq. 7.72, abrupt interface (permittivity jumps between E nodes).
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

function lineReflect({ pec = false, epsR = 0, cellsDielectric = 25, cellsFree = 40 }) {
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
    const g = createGrid({ nx, ny, dx, dy: dx, mode: "tmz", pml, periodicY: true });
    for (let j = 0; j < ny; j++) {
      addSource(g, { i: src, j, kind: "gaussian", f0, t0, spread, amp: 1 });
    }
    if (on && pec) setMaterial(g, { x0: wall, y0: 0, x1: nx - 1, y1: ny - 1, pec: 1 });
    if (on && epsR) setMaterial(g, { x0: wall, y0: 0, x1: nx - 1, y1: ny - 1, epsR });
    addProbe(g, { i: probe, j: 0 });
    step(g, nSteps);
    return g;
  };
  const ref = make(false);
  const tot = make(true);
  const diff = ref.probes[0].samples.map((v, i) => tot.probes[0].samples[i] - v);
  const echoCells = src - pml + (probe - pml);
  const iEcho = Math.floor((t0 + (echoCells * dx) / C0) / ref.dt) - 8;
  const i1 = Math.max(8, Math.min(diff.length, iEcho));
  const mag = dftMag(diff, ref.dt, f0, i1) / dftMag(ref.probes[0].samples, ref.dt, f0, i1);
  return { mag, dt: ref.dt, nFree, i1, nSteps };
}

test("A1 Courant dt formula", () => {
  const dx = 0.02;
  const dy = 0.05;
  const limit = 1 / (C0 * Math.sqrt(1 / dx ** 2 + 1 / dy ** 2));
  const dt = courantDt(dx, dy);
  assert.ok(dt <= limit);
  assert.ok(Math.abs(dt - 0.99 * limit) <= 1e-12 * limit);
  const grid = createGrid({ nx: 16, ny: 12, dx, dy, pml: 3 });
  assert.equal(grid.dt, courantDt(dx, dy));
  const s = (C0 * courantDt(1, 1)) / 1;
  assert.ok(s <= 1 / Math.sqrt(2) + 1e-15);
  log(
    `A1 PASS dt = 0.99/(c*sqrt(1/dx^2+1/dy^2)); square-grid S=${s.toFixed(6)} <= 1/sqrt(2)`,
  );
});

test("A2 cylindrical r^-1/2 within 5% and late-time 1/t tail", () => {
  const dx = 1;
  const cellsPerLambda = 24;
  const f0 = C0 / (cellsPerLambda * dx);
  const pml = 12;
  const radius = 220;
  const nx = 2 * (radius + pml) + 1;
  const cx = (nx / 2) | 0;
  const grid = createGrid({ nx, ny: nx, dx, dy: dx, mode: "tmz", pml });
  const spread = 1.6 / f0;
  const t0 = 2.8 * spread;
  addSource(grid, { i: cx, j: cx, kind: "gaussian", f0, t0, spread, amp: 1 });
  const rs = [50, 80, 110, 140, 170, 200];
  for (const r of rs) addProbe(grid, { i: cx + r, j: cx });
  const travel = (r) => t0 + (r * dx) / C0;
  const nSteps = Math.ceil((travel(200) + 2.2 * spread) / grid.dt) + 5;
  step(grid, nSteps);
  const norms = rs.map((r, k) => {
    const samples = grid.probes[k].samples;
    const i0 = Math.max(0, Math.floor((travel(r) - 1.4 * spread) / grid.dt));
    const i1 = Math.min(samples.length, Math.ceil((travel(r) + 1.4 * spread) / grid.dt));
    let peak = 0;
    for (let n = i0; n < i1; n++) peak = Math.max(peak, Math.abs(samples[n]));
    return peak * Math.sqrt(r);
  });
  const mean = norms.reduce((a, b) => a + b, 0) / norms.length;
  const dev = norms.map((v) => Math.abs(v / mean - 1));
  const maxDev = Math.max(...dev);
  assert.ok(maxDev < 0.05, `spreading deviation ${(100 * maxDev).toFixed(2)}%`);

  const rho = 24;
  const margin = Math.ceil(10.5 * rho);
  const nt = 2 * (margin + 10) + 1;
  const c = (nt / 2) | 0;
  const tail = createGrid({ nx: nt, ny: nt, dx, dy: dx, mode: "tmz", pml: 10 });
  const spreadT = 2.5 * tail.dt;
  const t0t = 4 * spreadT;
  addSource(tail, { i: c, j: c, kind: "gaussian", f0: 0, t0: t0t, spread: spreadT, amp: 1 });
  addProbe(tail, { i: c + rho, j: c });
  const tArr = (rho * dx) / C0;
  step(tail, Math.ceil((18 * tArr + t0t) / tail.dt));
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
  assert.ok(pos > neg, "late tail should keep the sign of the Green's function");
  log(
    `A2 PASS r^-1/2 max |dev|=${(100 * maxDev).toFixed(2)}% over r=50..200 at ${cellsPerLambda} cells/λ (edge ~20); tail slope ${slope.toFixed(3)} on log|G| vs log t for t in [10,18] ρ/c`,
  );
});

test("A3 PEC |R|=1 within 2%", () => {
  const { mag, nFree } = lineReflect({ pec: true, cellsFree: 40 });
  const err = Math.abs(mag - 1);
  assert.ok(err <= 0.02, `|R|=${mag}`);
  log(`A3 PASS |R|=${mag.toFixed(6)} err=${(100 * err).toFixed(4)}% at ${nFree} cells/λ (normal incidence)`);
});

test("A4 boundary reflection energy < 1%", () => {
  const dx = 1;
  const cells = 20;
  const f0 = C0 / (cells * dx);
  const spread = 1.2 / f0;
  const t0 = 2.5 * spread;
  const run = (n, pml) => {
    const g = createGrid({ nx: n, ny: n, dx, dy: dx, mode: "tmz", pml });
    const c = (n / 2) | 0;
    addSource(g, { i: c, j: c, kind: "gaussian", f0, t0, spread, amp: 1 });
    addProbe(g, { i: c + 18, j: c });
    const dist = c - pml;
    const back = ((dist + (dist - 18)) * dx) / C0;
    step(g, Math.ceil((t0 + back + 2 * spread) / g.dt));
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
  log(
    `A4 PASS reflected/incident energy=${ratio.toExponential(3)} (reference domain 220 vs PML domain 80, 10 cells, m=3) at ${cells} cells/λ`,
  );
});

test("A5 dielectric eps_r=4 |R| within 3% of 1/3", () => {
  const cellsDielectric = 25;
  const out = lineReflect({ epsR: 4, cellsDielectric });
  const third = 1 / 3;
  const relThird = Math.abs(out.mag - third) / third;
  const s = (C0 * out.dt) / 1;
  const gamma = discreteGamma(s, cellsDielectric, 4);
  const relDisc = Math.abs(out.mag - Math.abs(gamma)) / Math.abs(gamma);
  assert.ok(relThird <= 0.03, `|R|=${out.mag} vs 1/3`);
  assert.ok(relDisc <= 0.005, `|R|=${out.mag} vs discrete ${gamma}`);
  log(
    `A5 PASS |R|=${out.mag.toFixed(6)} vs 1/3 err=${(100 * relThird).toFixed(3)}% vs discrete Γ̃=${gamma.toFixed(6)} err=${(100 * relDisc).toFixed(3)}% at ${cellsDielectric} cells/λ in the dielectric (${out.nFree} in free space), first reflection, ABC excluded`,
  );
});

test("float32 storage vs float64 CPU", () => {
  const make = (storage) => {
    const g = createGrid({ nx: 40, ny: 32, dx: 1, dy: 1, pml: 6, storage });
    setMaterial(g, { x0: 18, y0: 6, x1: 30, y1: 24, epsR: 4 });
    setMaterial(g, { x0: 10, y0: 4, x1: 12, y1: 26, pec: 1 });
    const f0 = C0 / 20;
    addSource(g, { i: 8, j: 16, kind: "sine", f0, amp: 1 });
    return g;
  };
  const hi = make("float64");
  const lo = make("float32");
  const marks = [100, 500, 2000];
  let done = 0;
  for (const n of marks) {
    step(hi, n - done);
    step(lo, n - done);
    done = n;
    const m = relMetrics(hi.field, lo.field);
    const limit = 1e-4 * Math.max(1, n / 500);
    assert.ok(Number.isFinite(m.relL2));
    assert.ok(m.relL2 <= limit, `n=${n} relL2=${m.relL2} limit=${limit}`);
    log(
      `float32 n=${n} relL2=${m.relL2.toExponential(3)} max|d|/peak=${m.maxOverPeak.toExponential(3)} limit=${limit.toExponential(3)}`,
    );
  }
});

test("scalar mode propagates at c", () => {
  const g = createGrid({ nx: 140, ny: 48, dx: 1, dy: 1, mode: "scalar", pml: 8 });
  const cells = 30;
  const f0 = C0 / cells;
  addSource(g, { i: 24, j: 24, kind: "gaussian", f0, t0: 2 / f0, spread: 0.8 / f0, amp: 1 });
  addProbe(g, { i: 40, j: 24 });
  addProbe(g, { i: 70, j: 24 });
  step(g, 280);
  const argmax = (samples) => {
    let k = 0;
    for (let n = 1; n < samples.length; n++) {
      if (Math.abs(samples[n]) > Math.abs(samples[k])) k = n;
    }
    return k;
  };
  const k0 = argmax(g.probes[0].samples);
  const k1 = argmax(g.probes[1].samples);
  const s = (C0 * g.dt) / g.dx;
  const traveled = (k1 - k0) * s;
  assert.ok(Math.abs(traveled - 30) < 4, `traveled ${traveled} cells`);
  assert.ok(g.u && !g.Ez);
  log(`scalar PASS peak delay ${traveled.toFixed(2)} cells over a 30-cell path (c/sqrt(epsR))`);
});
