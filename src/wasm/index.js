/**
 * WASM FDTD with a fallback to src/fdtd.js.
 *
 * The committed package in ./pkg is what the static site loads. Node passes
 * the bytes straight to WebAssembly.instantiate. The browser glue fetches the
 * .wasm file; a Content-Security-Policy of script-src 'self' blocks that
 * instantiate, and this module then uses the JS solver.
 */

import * as js from '../fdtd.js';

let wasmApi = null;
let wasmError = null;

async function loadWasm() {
  if (wasmApi) return wasmApi;
  if (wasmError) throw wasmError;
  try {
    const glue = await import('./pkg/fdtd_wasm.js');
    const init = glue.default;
    if (typeof process !== 'undefined' && process.versions && process.versions.node) {
      const { readFileSync } = await import('node:fs');
      const bytes = readFileSync(new URL('./pkg/fdtd_wasm_bg.wasm', import.meta.url));
      await init({ module_or_path: bytes });
    } else {
      await init();
    }
    wasmApi = glue;
    return glue;
  } catch (err) {
    wasmError = err;
    throw err;
  }
}

class WasmGrid {
  constructor(raw, opts) {
    this.raw = raw;
    this.nx = opts.nx;
    this.ny = opts.ny;
    this.dx = opts.dx;
    this.dy = opts.dy;
    this.dt = raw.dt();
    this.mode = opts.mode;
    this.pml = opts.pml;
    this.storage = 'float64';
    this.periodicY = opts.periodicY;
    this.probes = [];
    this.sources = [];
    if (opts.mode === 'tmz') this.Ez = this;
    else this.u = this;
  }

  step(n = 1) {
    this.raw.step(n | 0);
    this._syncProbes();
    return this;
  }

  field64() {
    return this.raw.field();
  }

  snapshot() {
    const src = this.raw.field();
    const out = new Float32Array(src.length);
    out.set(src);
    return out;
  }

  _syncProbes() {
    for (let id = 0; id < this.probes.length; id++) {
      this.probes[id].samples = Array.from(this.raw.probe_samples(id));
    }
  }

  get steps() {
    return this.raw.steps();
  }
}

function bindWasm(glue) {
  return {
    backend: 'wasm',
    C0: glue.c0(),
    courantDt(dx, dy, c, safety) {
      return glue.courant_dt(dx, dy, c, safety);
    },
    createGrid(opts) {
      if (opts.storage === 'float32') {
        throw new Error('wasm solver is float64 only');
      }
      const periodicY = !!opts.periodicY;
      const minNy = periodicY ? 2 : 3;
      const nx = opts.nx;
      const ny = opts.ny;
      if (!Number.isInteger(nx) || !Number.isInteger(ny) || nx < 3 || ny < minNy) {
        throw new Error(`nx and ny must be integers (nx >= 3, ny >= ${minNy})`);
      }
      const cap = 2048;
      if (nx > cap || ny > cap) {
        throw new Error(`grid ${nx}x${ny} exceeds the ${cap} cap`);
      }
      const dx = opts.dx;
      const dy = opts.dy === undefined ? dx : opts.dy;
      if (!(dx > 0) || !(dy > 0)) throw new Error('dx and dy must be positive');
      const mode = opts.mode || 'tmz';
      const pml = opts.pml === undefined ? 10 : opts.pml;
      if (!Number.isInteger(pml) || pml < 0) {
        throw new Error(`pml must be an integer >= 0, got ${pml}`);
      }
      const raw = glue.create_grid(nx, ny, dx, dy, mode, pml, periodicY);
      return new WasmGrid(raw, { nx, ny, dx, dy, mode, pml, periodicY });
    },
    setMaterial(grid, spec) {
      grid.raw.set_material(
        spec.x0, spec.y0, spec.x1, spec.y1,
        spec.epsR, spec.sigma, spec.pec === undefined ? undefined : (spec.pec ? 1 : 0),
      );
      return grid;
    },
    addSource(grid, spec) {
      const id = grid.raw.add_source(
        spec.i | 0,
        spec.j | 0,
        spec.kind,
        spec.f0 ?? 0,
        spec.t0 ?? 0,
        spec.spread ?? 0,
        spec.amp ?? 1,
      );
      grid.sources.push(spec);
      return id;
    },
    addProbe(grid, { i, j }) {
      const id = grid.raw.add_probe(i | 0, j | 0);
      grid.probes.push({ i: i | 0, j: j | 0, samples: [] });
      return id;
    },
    step(grid, n = 1) {
      return grid.step(n);
    },
    snapshot(grid) {
      return grid.snapshot();
    },
  };
}

function bindJs() {
  return {
    backend: 'js',
    C0: js.C0,
    courantDt: js.courantDt,
    createGrid: js.createGrid,
    setMaterial: js.setMaterial,
    addSource: js.addSource,
    addProbe: js.addProbe,
    step: js.step,
    snapshot: js.snapshot,
  };
}

/** Throws if the wasm module does not instantiate. */
export async function loadWasmFDTD() {
  return bindWasm(await loadWasm());
}

/**
 * Wasm when it loads, otherwise src/fdtd.js.
 * `fallbackReason` is null on the wasm path. `load` is a test seam.
 */
export async function createFDTD({ load = loadWasm } = {}) {
  try {
    const api = bindWasm(await load());
    return { ...api, fallbackReason: null };
  } catch (err) {
    return { ...bindJs(), fallbackReason: String(err && err.message ? err.message : err) };
  }
}
