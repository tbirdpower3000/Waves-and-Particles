import { step as stepCpu } from "../fdtd.js";
import { fieldToImage } from "./color.js";
import { createWebGPUStepper } from "./webgpu.js";
import { createWebGL2Stepper } from "./webgl2.js";

/**
 * GPU stepper with the same update as the CPU reference in src/fdtd.js.
 * `step` returns a promise; await it before readProbes(). readField waits
 * for any stepped work that has not finished yet.
 *
 * @param {object} grid from createGrid (materials and sources already set)
 * @param {{ backend?: 'webgpu' | 'webgl2' | 'auto' }} [opts]
 */
export async function createGpuStepper(grid, opts = {}) {
  const backend = opts.backend ?? "auto";
  if (backend !== "webgpu" && backend !== "webgl2" && backend !== "auto") {
    throw new Error(`backend must be 'webgpu', 'webgl2', or 'auto', got ${backend}`);
  }
  /** @type {unknown[]} */
  const failures = [];
  if (backend === "webgpu" || backend === "auto") {
    let gpu = null;
    try {
      gpu = await createWebGPUStepper(grid);
    } catch (err) {
      // WebGPU was present enough to start init. A shader or init exception
      // must fail auto mode instead of silently becoming a WebGL2 success.
      failures.push(err);
      if (backend === "webgpu" || backend === "auto") throw err;
    }
    if (gpu) return gpu;
    const reason = "no WebGPU adapter";
    if (backend === "webgpu") throw new Error("UNKNOWN: no GPU backend");
    console.log(`GPU auto-fallback: ${reason}; trying WebGL2`);
  }
  if (backend === "webgl2" || backend === "auto") {
    try {
      const gl = createWebGL2Stepper(grid);
      if (gl) {
        if (backend === "auto") gl.fallbackReason = "no WebGPU adapter";
        return gl;
      }
    } catch (err) {
      failures.push(err);
      if (backend === "webgl2" || backend === "auto") throw err;
    }
  }
  if (failures.length) {
    const error = new Error(failures.map((err) => err?.message || String(err)).join("\n"));
    error.cause = failures;
    throw error;
  }
  throw new Error("UNKNOWN: no GPU backend");
}

function cpuView(canvas, grid, opts = {}) {
  const stepsPerFrame = opts.stepsPerFrame ?? 1;
  const scale = opts.scale ?? 1;
  canvas.width = grid.nx;
  canvas.height = grid.ny;
  let last = fieldToImage(grid.field, grid.nx, grid.ny, scale);
  const paint = () => {
    last = fieldToImage(grid.field, grid.nx, grid.ny, scale);
    const ctx = canvas.getContext?.("2d");
    if (ctx && typeof ctx.putImageData === "function") {
      const image =
        typeof ImageData === "function"
          ? new ImageData(last, grid.nx, grid.ny)
          : { data: last, width: grid.nx, height: grid.ny };
      ctx.putImageData(image, 0, 0);
    }
  };
  return {
    backend: "cpu",
    adapter: null,
    probeStream: null,
    fullFieldReads: 0,
    materialBytesUploaded: 0,
    canvas,
    stepsPerFrame,
    scale,
    frame(count) {
      stepCpu(grid, count == null ? stepsPerFrame : count | 0);
      paint();
    },
    idle() {
      return Promise.resolve();
    },
    readDrawn() {
      return Promise.resolve(last.slice());
    },
    readProbes: () => grid.probes.map((p) => ({ i: p.i, j: p.j, samples: p.samples.slice() })),
    syncMaterials() {
      return 0;
    },
    destroy() {},
  };
}

/**
 * Draw the field from GPU memory. `frame` submits steps and a colormap
 * pass and does not read the field back. With no adapter, the same calls
 * step the CPU grid and paint it.
 *
 * @param {object} canvas
 * @param {object} grid from createGrid
 * @param {{ backend?: 'webgpu' | 'webgl2' | 'auto', stepsPerFrame?: number, scale?: number }} [opts]
 */
export async function createGpuView(canvas, grid, opts = {}) {
  let stepper = null;
  try {
    stepper = await createGpuStepper(grid, { backend: opts.backend });
  } catch (err) {
    const message = String(err?.message || err);
    if (!message.includes("UNKNOWN: no GPU backend")) throw err;
    console.log(`GPU auto-fallback: ${message}; drawing on the CPU`);
  }
  if (!stepper?.createView) return cpuView(canvas, grid, opts);
  const view = stepper.createView(canvas, opts);
  view.adapter = stepper.adapter;
  view.probeStream = stepper.probeStream ?? null;
  view.syncMaterials = (next) => stepper.syncMaterials(next ?? grid);
  Object.defineProperty(view, "materialBytesUploaded", {
    get: () => stepper.materialBytesUploaded,
  });
  return view;
}
