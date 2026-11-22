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
