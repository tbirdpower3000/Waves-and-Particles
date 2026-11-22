# A9 performance (field core)

Reported, not gated. Live numbers are printed by `node --test`. Every GPU figure below is SwiftShader software. None of them is a hardware pass.

CPU figures are the float64 reference in `src/fdtd.js` (sine source, 8-cell CPML, 5 warmup steps excluded). GPU figures are `createGpuStepper` on the same grid. The timer starts after warmup and `idle()`, then `kick(steps)` plus one queue fence. It does not read the field back.

The resident path keeps fields, materials, and CPML on the GPU. A geometry edit uploads only the cells whose coefficients changed. Probe samples sit in a GPU ring and come back through double-buffered `mapAsync` copies, so `frame` does not wait on that map.

## Before the resident path

Same machine, previous stepper. Each step uploaded a full-grid source, and the timer included one full-field readback. Node v22.23.4 only. SwiftShader software.

| Grid | steps/s | Steps |
|---|---|---|
| 256² | 169.13 | 40 |
| 512² | 53.21 | 20 |
| 1024² | 12.42 | 8 |

## After, Node v22.23.4

| Grid | Backend | steps/s | Notes |
|---|---|---|---|
| 256² | CPU | 1342.33 | 80 steps |
| 512² | CPU | 391.52 | 40 steps |
| 1024² | CPU | 100.93 | 12 steps |
| 256² | WebGPU SwiftShader | 157.42 | software, 40 steps |
| 512² | WebGPU SwiftShader | 46.75 | software, 20 steps |
| 1024² | WebGPU SwiftShader | 12.11 | software, 8 steps |

## After, Node v20.20.2

| Grid | Backend | steps/s | Notes |
|---|---|---|---|
| 256² | CPU | 1357.60 | 80 steps |
| 512² | CPU | 359.62 | 40 steps |
| 1024² | CPU | 89.49 | 12 steps |
| 256² | WebGPU SwiftShader | 135.96 | software, 40 steps |
| 512² | WebGPU SwiftShader | 38.21 | software, 20 steps |
| 1024² | WebGPU SwiftShader | 8.34 | software, 8 steps |

This VM has no discrete GPU. The WebGPU adapter is vendor `google`, architecture `swiftshader`, and `adapter.info.isFallbackAdapter` is true. WebGL2's unmasked renderer is `ANGLE ... SwiftShader`, so that path is software too. A masked renderer (`WebKit WebGL`, no `WEBGL_debug_renderer_info`) is UNKNOWN, not hardware. GPU rates move around between runs because the grids are small and the adapter is software. The test log is the measurement for that process.
