import { gpuBufferGate } from './params.js';

export const PROBE_CAP = 4;
export const PROBE_COLORS = ['#2ee6c7', '#ffb020', '#8b7cff', '#ff5d6c'];

export function createScene(params = {}) {
  return {
    mode: params.mode === 'scalar' ? 'scalar' : 'tmz',
    nx: params.nx ?? 512,
    ny: params.ny ?? 512,
    dx: 1,
    dy: 1,
    pml: params.pml ?? 16,
    epsR: params.eps ?? 4,
    materials: [],
    source: null,
    probes: [],
  };
}

export function clampCell(i, j, nx, ny) {
  return {
    i: Math.min(nx - 1, Math.max(0, i)),
    j: Math.min(ny - 1, Math.max(0, j)),
  };
}

export function cellFromOffset(offsetX, offsetY, width, height, nx, ny) {
  if (!(width > 0) || !(height > 0)) return null;
  const u = offsetX / width;
  const v = offsetY / height;
  if (u < 0 || v < 0 || u > 1 || v > 1) return null;
  return {
    i: Math.min(nx - 1, Math.max(0, Math.floor(Math.min(u, 0.999999) * nx))),
    j: Math.min(ny - 1, Math.max(0, Math.floor(Math.min(v, 0.999999) * ny))),
  };
}

export function rectFromCells(a, b) {
  return {
    x0: Math.min(a.i, b.i),
    y0: Math.min(a.j, b.j),
    x1: Math.max(a.i, b.i),
    y1: Math.max(a.j, b.j),
  };
}

export function materialFromRect(rect, tool, epsR) {
  if (tool === 'metal') {
    return { ...rect, epsR: 1, sigma: 0, pec: true };
  }
  const eps = Number(epsR);
  return {
    ...rect,
    epsR: Number.isFinite(eps) ? eps : 4,
    sigma: 0,
    pec: false,
  };
}

export function addProbe(scene, i, j) {
  if (scene.probes.length >= PROBE_CAP) {
    return { ok: false, reason: '4 receiver probes maximum' };
  }
  if (scene.probes.some((probe) => probe.i === i && probe.j === j)) {
    return { ok: false, reason: 'a probe is already on that cell' };
  }
  scene.probes.push({ i, j });
  return { ok: true, reason: '' };
}

export function sourceTiming(grid, courantDt, c0, kind) {
  const dx = Number.isFinite(grid.dx) && grid.dx > 0 ? grid.dx : 1;
  const dy = Number.isFinite(grid.dy) && grid.dy > 0 ? grid.dy : dx;
  let dt = grid.dt;
  if (!(Number.isFinite(dt) && dt > 0)) {
    dt = typeof courantDt === 'function' ? courantDt(dx, dy) : 1;
  }
  if (!(Number.isFinite(dt) && dt > 0)) dt = 1;
  const speed = Number.isFinite(c0) && c0 > 0 ? c0 : 1;
  const f0 = speed / (24 * dx);
  if (kind === 'gaussian') {
    const spread = 12 * dt;
    return { kind: 'gaussian', f0, spread, t0: 4 * spread, amp: 1 };
  }
  return { kind: 'sine', f0, spread: dt, t0: 0, amp: 1 };
}

export function rebuildGrid(scene, fdtd, sourceKind) {
  gpuBufferGate(scene.nx, scene.ny);
  if (!Number.isFinite(scene.dx) || !Number.isFinite(scene.dy) || scene.dx <= 0 || scene.dy <= 0) {
    throw new Error('dx and dy must be finite and positive');
  }
  const grid = fdtd.createGrid({
    nx: scene.nx,
    ny: scene.ny,
    dx: scene.dx,
    dy: scene.dy,
    mode: scene.mode,
    pml: scene.pml,
  });
  for (const material of scene.materials) {
    fdtd.setMaterial(grid, { ...material });
  }
  if (scene.source) {
    const kind = sourceKind || scene.source.kind || 'sine';
    const timing = sourceTiming(grid, fdtd.courantDt, fdtd.C0, kind);
    fdtd.addSource(grid, {
      i: scene.source.i,
      j: scene.source.j,
      kind: timing.kind,
      f0: timing.f0,
      t0: timing.t0,
      spread: timing.spread,
      amp: timing.amp,
    });
  }
  scene.probes.forEach((probe, index) => {
    const id = fdtd.addProbe(grid, { i: probe.i, j: probe.j });
    probe.id = typeof id === 'number' ? id : index;
  });
  return grid;
}
