/**
 * DEV MOCK. Not a physics solver and not the signal chain.
 * The page does not import this file. It lives under tests/ for the smoke tests.
 */

export const C0 = 299792458;

export function courantDt(dx, dy, c = C0, safety = 0.99) {
  return (safety * c) / Math.sqrt(1 / (dx * dx) + 1 / (dy * dy));
}

export function createGrid({ nx, ny, dx, dy, mode, pml }) {
  const cells = nx * ny;
  const field = new Float64Array(cells);
  const grid = {
    nx, ny, dx, dy, mode, pml,
    dt: courantDt(dx, dy),
    field,
    Ez: field,
    u: field,
    Hx: new Float64Array(cells),
    Hy: new Float64Array(cells),
    epsR: new Float64Array(cells).fill(1),
    sigma: new Float64Array(cells),
    pec: new Uint8Array(cells),
    sources: [],
    probes: [],
    stepCount: 0,
  };
  return grid;
}

export function setMaterial(grid, { x0, y0, x1, y1, epsR = 1, sigma = 0, pec = false }) {
  for (let j = y0; j <= y1; j++) {
    for (let i = x0; i <= x1; i++) {
      if (i < 0 || j < 0 || i >= grid.nx || j >= grid.ny) continue;
      const at = j * grid.nx + i;
      grid.epsR[at] = epsR;
      grid.sigma[at] = sigma;
      grid.pec[at] = pec ? 1 : 0;
    }
  }
}

export function addSource(grid, source) {
  grid.sources.push({ ...source });
}

export function addProbe(grid, probe) {
  const id = grid.probes.length;
  grid.probes.push({ ...probe, samples: [] });
  return id;
}

export function step(grid, n = 1) {
  for (let k = 0; k < n; k++) {
    grid.stepCount += 1;
    const t = grid.stepCount * grid.dt;
    for (const source of grid.sources) {
      const at = source.j * grid.nx + source.i;
      const value = source.kind === 'gaussian'
        ? Math.exp(-((t - source.t0) ** 2) / (2 * source.spread * source.spread))
        : Math.sin(2 * Math.PI * source.f0 * t);
      if (at >= 0 && at < grid.field.length) grid.field[at] = value;
    }
    for (const probe of grid.probes) {
      const at = probe.j * grid.nx + probe.i;
      probe.samples.push(grid.field[at] || 0);
    }
  }
}

export function snapshot(grid) {
  return Float32Array.from(grid.field);
}

export async function createGpuStepper(grid) {
  return {
    backend: 'dev-mock',
    step(n = 1) { step(grid, n); },
    async readField() { return snapshot(grid); },
    readProbes() { return grid.probes; },
    destroy() {},
  };
}

export const fdtd = {
  C0, courantDt, createGrid, setMaterial, addSource, addProbe, step, snapshot,
};

export const gpu = { createGpuStepper };

export const signal = {
  bitsRandom(n) { return new Uint8Array(Math.max(0, n | 0)); },
  modulate() { return { i: new Float64Array(0), q: new Float64Array(0) }; },
  chirp() { return { i: new Float64Array(0), q: new Float64Array(0) }; },
  convolveChannel(sig) {
    return { i: Float64Array.from(sig?.i ?? []), q: Float64Array.from(sig?.q ?? []) };
  },
  awgn(sig) {
    return { i: Float64Array.from(sig?.i ?? []), q: Float64Array.from(sig?.q ?? []) };
  },
  demodulate() { return new Uint8Array(0); },
  ber() { return 0; },
  berTheory() { return 0; },
  fftMag() { return new Float64Array(0); },
};

export function makeRng() {
  return { uniform: () => 0.5, normal: () => 0 };
}

export const rng = { makeRng };
