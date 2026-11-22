/**
 * 2D FDTD reference (float64 storage by default).
 *
 * TMz Yee grid: Ez at integer nodes, Hx at (i, j+1/2), Hy at (i+1/2, j).
 * Scalar mode stores the same stagger in `u` with μ = 1 and ε = epsR / c²,
 * so a free-space cell propagates at c and epsR slows it by sqrt(epsR).
 *
 * Boundaries: convolutional PML (Roden & Gedney, Microwave Opt. Technol. Lett.
 * 27, 2000) with κ = 1 and α = 0, cubic grading m = 3, and
 * σ_max = 0.8 (m+1) / (η Δ). α = 0 keeps the E and H recursions identical
 * (matched σ_m = σ μ/ε). The outermost index is not updated, which terminates
 * the PML on a PEC wall. Conductivity and epsR stay in the ordinary Ca, Cb
 * update, so the stretch does not depend on the host medium.
 *
 * setMaterial uses inclusive integer cell indices.
 */

export const C0 = 299792458;

const MU0 = 4e-7 * Math.PI;
const EPS0 = 1 / (MU0 * C0 * C0);
const PML_M = 3;

export function courantDt(dx, dy, c = C0, safety = 0.99) {
  if (!(dx > 0) || !(dy > 0) || !(c > 0)) {
    throw new Error("courantDt requires positive dx, dy, and c");
  }
  const s = safety === undefined ? 0.99 : safety;
  return s / (c * Math.sqrt(1 / (dx * dx) + 1 / (dy * dy)));
}

function arrayType(storage) {
  return storage === "float32" ? Float32Array : Float64Array;
}

function zeros(storage, n) {
  return new (arrayType(storage))(n);
}

/** Soft-source waveform. Gaussian with f0 > 0 is a sine carrier under the envelope. */
export function sampleSource(src, t) {
  const amp = src.amp ?? 1;
  if (src.kind === "sine") {
    return amp * Math.sin(2 * Math.PI * (src.f0 ?? 0) * t);
  }
  if (src.kind !== "gaussian") {
    throw new Error(`unknown source kind ${src.kind}`);
  }
  const t0 = src.t0 ?? 0;
  const spread = src.spread ?? 0;
  if (!(spread > 0)) return 0;
  const x = (t - t0) / spread;
  const env = Math.exp(-x * x);
  const f0 = src.f0 ?? 0;
  if (f0) return amp * env * Math.sin(2 * Math.PI * f0 * (t - t0));
  return amp * env;
}

/**
 * Running sum of the waveform. Adding this sum into the field makes the
 * leapfrog second difference equal the waveform, so the waveform is the
 * source term of the wave equation.
 */
export function accumulateSource(src, t) {
  src.acc = (src.acc ?? 0) + sampleSource(src, t);
  return src.acc;
}

function sigmaNormMax(delta) {
  // σ/ε0 = 0.8 (m+1) c / Δ, from σ_max = 0.8 (m+1) / (η0 Δ) and η0 ε0 c = 1.
  return (0.8 * (PML_M + 1) * C0) / delta;
}

function sigmaAt(x, length, thickness, peak) {
  let depth = 0;
  if (x < thickness) depth = thickness - x;
  else if (x > length - thickness) depth = x - (length - thickness);
  if (depth <= 0 || thickness <= 0) return 0;
  const rho = depth / thickness;
  return peak * rho ** PML_M;
}

function abOf(sigmaNorm, dt) {
  if (!(sigmaNorm > 0)) return [0, 1];
  const b = Math.exp(-sigmaNorm * dt);
  // κ = 1, α = 0 ⇒ a = (b - 1).
  return [b - 1, b];
}

function fillAxis(n, delta, pmlCells, dt, storage) {
  const thickness = pmlCells * delta;
  const length = n * delta;
  const peak = sigmaNormMax(delta);
  const aE = zeros(storage, n);
  const bE = zeros(storage, n);
  const aH = zeros(storage, Math.max(0, n - 1));
  const bH = zeros(storage, Math.max(0, n - 1));
  for (let i = 0; i < n; i++) {
    const [a, b] = abOf(sigmaAt((i + 0.5) * delta, length, thickness, peak), dt);
    aE[i] = a;
    bE[i] = b;
  }
  for (let i = 0; i < n - 1; i++) {
    const [a, b] = abOf(sigmaAt((i + 1) * delta, length, thickness, peak), dt);
    aH[i] = a;
    bH[i] = b;
  }
  let h0 = 0;
  while (h0 < n - 1 && aH[h0] !== 0) h0++;
  let h1 = n - 1;
  while (h1 > h0 && aH[h1 - 1] !== 0) h1--;
  let e0 = 0;
  while (e0 < n && aE[e0] !== 0) e0++;
  let e1 = n - 1;
  while (e1 >= e0 && aE[e1] !== 0) e1--;
  return { aE, bE, aH, bH, h0, h1, e0, e1 };
}

function epsOf(grid, epsR) {
  return grid.epsBase * epsR;
}

function writeCoeff(grid, id) {
  if (grid.pec[id]) {
    grid.ca[id] = 0;
    grid.cb[id] = 0;
    return;
  }
  const eps = epsOf(grid, grid.epsR[id]);
  const loss = (grid.sigma[id] * grid.dt) / (2 * eps);
  const den = 1 + loss;
  grid.ca[id] = (1 - loss) / den;
  grid.cb[id] = grid.dt / eps / den;
}

function rebuildPecList(grid) {
  const { pec, nx, ny } = grid;
  const list = [];
  for (let id = 0; id < nx * ny; id++) if (pec[id]) list.push(id);
  grid.pecList = Int32Array.from(list);
}

export function createGrid({
  nx,
  ny,
  dx,
  dy,
  mode = "tmz",
  pml = 10,
  storage = "float64",
  periodicY = false,
}) {
  if (mode !== "tmz" && mode !== "scalar") {
    throw new Error(`mode must be 'tmz' or 'scalar', got ${mode}`);
  }
  if (storage !== "float64" && storage !== "float32") {
    throw new Error(`storage must be 'float64' or 'float32', got ${storage}`);
  }
  const minNy = periodicY ? 2 : 3;
  if (!Number.isInteger(nx) || !Number.isInteger(ny) || nx < 3 || ny < minNy) {
    throw new Error(`nx and ny must be integers (nx >= 3, ny >= ${minNy})`);
  }
  if (dy === undefined) dy = dx;
  if (!(dx > 0) || !(dy > 0)) throw new Error("dx and dy must be positive");
  const pmlCells = pml | 0;
  // periodicY is an optional y-periodic boundary (no y PML). It is off by default.
  if (pmlCells < 0 || pmlCells * 2 >= nx || (!periodicY && pmlCells * 2 >= ny)) {
    throw new Error("pml thickness leaves no interior");
  }

  const dt = courantDt(dx, dy);
  const n = nx * ny;
  const field = zeros(storage, n);
  const grid = {
    nx,
    ny,
    dx,
    dy,
    dt,
    mode,
    pml: pmlCells,
    storage,
    field,
    Hx: zeros(storage, n),
    Hy: zeros(storage, n),
    epsR: new Float64Array(n).fill(1),
    sigma: new Float64Array(n),
    pec: new Uint8Array(n),
    ca: zeros(storage, n),
    cb: zeros(storage, n),
    psiEzx: zeros(storage, n),
    psiEzy: zeros(storage, n),
    psiHyx: zeros(storage, n),
    psiHxy: zeros(storage, n),
    sources: [],
    probes: [],
    time: 0,
    steps: 0,
    periodicY: !!periodicY,
    mu: mode === "tmz" ? MU0 : 1,
    epsBase: mode === "tmz" ? EPS0 : 1 / (C0 * C0),
  };
  grid.ch = dt / grid.mu;
  grid.invDx = 1 / dx;
  grid.invDy = 1 / dy;
  if (mode === "tmz") grid.Ez = field;
  else grid.u = field;

  const xAxis = fillAxis(nx, dx, pmlCells, dt, storage);
  const yAxis = fillAxis(ny, dy, periodicY ? 0 : pmlCells, dt, storage);
  grid.aEx = xAxis.aE;
  grid.bEx = xAxis.bE;
  grid.aHx = xAxis.aH;
  grid.bHx = xAxis.bH;
  grid.hy0 = xAxis.h0;
  grid.hy1 = xAxis.h1;
  grid.aEy = yAxis.aE;
  grid.bEy = yAxis.bE;
  grid.aHy = yAxis.aH;
  grid.bHy = yAxis.bH;
  grid.hx0 = yAxis.h0;
  grid.hx1 = yAxis.h1;

  grid.iInt0 = Math.max(1, xAxis.e0);
  grid.iInt1 = Math.min(nx - 2, xAxis.e1);
  grid.jInt0 = Math.max(1, yAxis.e0);
  grid.jInt1 = Math.min(ny - 2, yAxis.e1);

  for (let id = 0; id < n; id++) writeCoeff(grid, id);
  grid.pecList = new Int32Array(0);
  return grid;
}

export function setMaterial(grid, { x0, y0, x1, y1, epsR, sigma, pec }) {
  const { nx, ny } = grid;
  const i0 = Math.max(0, Math.min(nx - 1, Math.floor(x0)));
  const i1 = Math.max(0, Math.min(nx - 1, Math.floor(x1)));
  const j0 = Math.max(0, Math.min(ny - 1, Math.floor(y0)));
  const j1 = Math.max(0, Math.min(ny - 1, Math.floor(y1)));
  const loI = Math.min(i0, i1);
  const hiI = Math.max(i0, i1);
  const loJ = Math.min(j0, j1);
  const hiJ = Math.max(j0, j1);
  const setE = epsR !== undefined;
  const setS = sigma !== undefined;
  const setP = pec !== undefined;
  for (let j = loJ; j <= hiJ; j++) {
    const row = j * nx;
    for (let i = loI; i <= hiI; i++) {
      const id = row + i;
      if (setE) grid.epsR[id] = epsR;
      if (setS) grid.sigma[id] = sigma;
      if (setP) grid.pec[id] = pec ? 1 : 0;
      writeCoeff(grid, id);
    }
  }
  if (setP) rebuildPecList(grid);
  return grid;
}

export function addSource(grid, spec) {
  const { nx, ny } = grid;
  const i = spec.i | 0;
  const j = spec.j | 0;
  if (i < 0 || j < 0 || i >= nx || j >= ny) {
    throw new Error(`source (${i},${j}) is outside the grid`);
  }
  if (spec.kind !== "gaussian" && spec.kind !== "sine") {
    throw new Error(`unknown source kind ${spec.kind}`);
  }
  grid.sources.push({
    i,
    j,
    kind: spec.kind,
    f0: spec.f0 ?? 0,
    t0: spec.t0 ?? 0,
    spread: spec.spread ?? 0,
    amp: spec.amp ?? 1,
    acc: 0,
  });
  return grid.sources.length - 1;
}

export function addProbe(grid, { i, j }) {
  const ii = i | 0;
  const jj = j | 0;
  if (ii < 0 || jj < 0 || ii >= grid.nx || jj >= grid.ny) {
    throw new Error(`probe (${ii},${jj}) is outside the grid`);
  }
  const id = grid.probes.length;
  grid.probes.push({ i: ii, j: jj, samples: [] });
  return id;
}

function updateHx(grid) {
  const { nx, ny, field, Hx, psiHxy, aHy, bHy, ch, invDy, hx0, hx1, periodicY } = grid;
  const jTop = periodicY ? ny : ny - 1;
  for (let j = 0; j < jTop; j++) {
    const up = j + 1 < ny ? j + 1 : 0;
    const row = j * nx;
    const rowU = up * nx;
    const pmlRow = !periodicY && (j < hx0 || j >= hx1);
    if (!pmlRow) {
      for (let i = 0; i < nx; i++) {
        Hx[row + i] -= ch * (field[rowU + i] - field[row + i]) * invDy;
      }
      continue;
    }
    const a = aHy[j];
    const b = bHy[j];
    for (let i = 0; i < nx; i++) {
      const id = row + i;
      const dEdy = (field[rowU + i] - field[id]) * invDy;
      const psi = b * psiHxy[id] + a * dEdy;
      psiHxy[id] = psi;
      Hx[id] -= ch * (dEdy + psi);
    }
  }
}

function updateHy(grid) {
  const { nx, ny, field, Hy, psiHyx, aHx, bHx, ch, invDx, hy0, hy1 } = grid;
  for (let j = 0; j < ny; j++) {
    const row = j * nx;
    for (let i = 0; i < hy0; i++) {
      const id = row + i;
      const dEdx = (field[id + 1] - field[id]) * invDx;
      const psi = bHx[i] * psiHyx[id] + aHx[i] * dEdx;
      psiHyx[id] = psi;
      Hy[id] += ch * (dEdx + psi);
    }
    for (let i = hy0; i < hy1; i++) {
      const id = row + i;
      Hy[id] += ch * (field[id + 1] - field[id]) * invDx;
    }
    for (let i = hy1; i < nx - 1; i++) {
      const id = row + i;
      const dEdx = (field[id + 1] - field[id]) * invDx;
      const psi = bHx[i] * psiHyx[id] + aHx[i] * dEdx;
      psiHyx[id] = psi;
      Hy[id] += ch * (dEdx + psi);
    }
  }
}

function updateEzAt(grid, i, j, withPsi) {
  const { nx, field, Hx, Hy, ca, cb, pec, psiEzx, psiEzy, aEx, bEx, aEy, bEy, invDx, invDy } =
    grid;
  const id = j * nx + i;
  if (pec[id]) {
    field[id] = 0;
    return;
  }
  const dHy = (Hy[id] - Hy[id - 1]) * invDx;
  const dHx = (Hx[id] - Hx[id - nx]) * invDy;
  if (!withPsi) {
    field[id] = ca[id] * field[id] + cb[id] * (dHy - dHx);
    return;
  }
  const px = bEx[i] * psiEzx[id] + aEx[i] * dHy;
  const py = bEy[j] * psiEzy[id] + aEy[j] * dHx;
  psiEzx[id] = px;
  psiEzy[id] = py;
  field[id] = ca[id] * field[id] + cb[id] * (dHy + px - (dHx + py));
}

function updateEzInterior(grid, j0, j1) {
  const { nx, field, Hx, Hy, ca, cb, pec, invDx, invDy, iInt0, iInt1 } = grid;
  for (let j = j0; j <= j1; j++) {
    const row = j * nx;
    const rowD = row - nx;
    for (let i = iInt0; i <= iInt1; i++) {
      const id = row + i;
      if (pec[id]) {
        field[id] = 0;
        continue;
      }
      const dHy = (Hy[id] - Hy[id - 1]) * invDx;
      const dHx = (Hx[id] - Hx[rowD + i]) * invDy;
      field[id] = ca[id] * field[id] + cb[id] * (dHy - dHx);
    }
  }
}

function updateEz(grid) {
  const { nx, ny, iInt0, iInt1, jInt0, jInt1, periodicY } = grid;
  if (!periodicY) {
    if (jInt0 <= jInt1 && iInt0 <= iInt1) updateEzInterior(grid, jInt0, jInt1);
    for (let j = 1; j < ny - 1; j++) {
      const interiorRow = j >= jInt0 && j <= jInt1;
      if (interiorRow) {
        for (let i = 1; i < iInt0; i++) updateEzAt(grid, i, j, true);
        for (let i = iInt1 + 1; i < nx - 1; i++) updateEzAt(grid, i, j, true);
      } else {
        for (let i = 1; i < nx - 1; i++) updateEzAt(grid, i, j, true);
      }
    }
    return;
  }
  for (let j = 0; j < ny; j++) {
    const below = (j - 1 + ny) % ny;
    for (let i = 1; i < nx - 1; i++) {
      const withPsi = i < iInt0 || i > iInt1;
      updateEzPeriodic(grid, i, j, below, withPsi);
    }
  }
}

function updateEzPeriodic(grid, i, j, below, withPsi) {
  const { nx, field, Hx, Hy, ca, cb, pec, psiEzx, psiEzy, aEx, bEx, aEy, bEy, invDx, invDy } =
    grid;
  const id = j * nx + i;
  if (pec[id]) {
    field[id] = 0;
    return;
  }
  const dHy = (Hy[id] - Hy[id - 1]) * invDx;
  const dHx = (Hx[id] - Hx[below * nx + i]) * invDy;
  if (!withPsi) {
    field[id] = ca[id] * field[id] + cb[id] * (dHy - dHx);
    return;
  }
  const px = bEx[i] * psiEzx[id] + aEx[i] * dHy;
  const py = bEy[j] * psiEzy[id] + aEy[j] * dHx;
  psiEzx[id] = px;
  psiEzy[id] = py;
  field[id] = ca[id] * field[id] + cb[id] * (dHy + px - (dHx + py));
}

export function step(grid, n = 1) {
  const { field, pecList, sources, probes, nx } = grid;
  const count = n | 0;
  for (let s = 0; s < count; s++) {
    updateHx(grid);
    updateHy(grid);
    updateEz(grid);
    grid.time += grid.dt;
    const t = grid.time;
    for (let q = 0; q < sources.length; q++) {
      const src = sources[q];
      const id = src.j * nx + src.i;
      if (grid.pec[id]) continue;
      field[id] += accumulateSource(src, t);
    }
    for (let p = 0; p < pecList.length; p++) field[pecList[p]] = 0;
    for (let q = 0; q < probes.length; q++) {
      const pr = probes[q];
      pr.samples.push(field[pr.j * nx + pr.i]);
    }
    grid.steps += 1;
  }
  return grid;
}

export function snapshot(grid) {
  const out = new Float32Array(grid.field.length);
  out.set(grid.field);
  return out;
}
