# A9 performance (field core)

Reported, not gated. Live numbers are printed by `node --test`.

CPU figures are the float64 reference in `src/fdtd.js` (sine source, 8-cell CPML, 5 warmup steps excluded). GPU figures are `createGpuStepper` on the same grid, 2 warmup steps excluded, timed through one field readback so the queue is idle at the end.

This VM has no discrete GPU. The WebGPU adapter is vendor `google`, architecture `swiftshader`, and `adapter.info.isFallbackAdapter` is true. That is a software adapter, so A7 is UNKNOWN for real hardware. The software comparison still met the 1e-3 field tolerance (rel L2 1.86e-5 after 500 steps). WebGL2's masked `RENDERER` string is `WebKit WebGL`; the unmasked renderer from `WEBGL_debug_renderer_info` is `ANGLE ... SwiftShader`.

## Node v22.23.4

| Grid | Backend | steps/s | Notes |
|---|---|---|---|
| 256² | CPU | 1327 | 80 steps |
| 512² | CPU | 399 | 40 steps |
| 256² | WebGPU SwiftShader | 126 | software, 40 steps |
| 512² | WebGPU SwiftShader | 38 | software, 20 steps |

## Node v20.20.2

| Grid | Backend | steps/s | Notes |
|---|---|---|---|
| 256² | CPU | 1408 | 80 steps |
| 512² | CPU | 381 | 40 steps |
| 256² | WebGPU SwiftShader | 96 | software, 40 steps |
| 512² | WebGPU SwiftShader | 36 | software, 20 steps |

GPU rates move around between runs because the grids are small and the adapter is software. The test log is the measurement for that process.
