# Waves-and-Particles module contract (WAP-001 r1)

No build step. Browser loads ES modules from `src/` via `<script type="module">`; Node (>=20) imports the same files in tests. No runtime npm deps; dev deps none required (tests use `node:test` + `node:assert`).

## File ownership (one writer per file during r1)
| Path | Owner |
|---|---|
| `src/fdtd.js`, `src/gpu/*` , `tests/fdtd*.test.js` | Field-core agent |
| `src/signal.js`, `src/rng.js`, `tests/signal*.test.js`, `tests/rng*.test.js` | Signal agent |
| `index.html`, `src/ui/*`, `.github/workflows/ci.yml`, `README.md`, `package.json`, `tests/smoke*.test.js` | UI/CI agent |
| `docs/*` | scaffold only (read-only afterwards) |

## src/fdtd.js (pure JS, CPU reference; float64)
```js
export const C0 = 299792458;
export function courantDt(dx, dy, c = C0, safety = 0.99)            // returns dt
export function createGrid({ nx, ny, dx, dy, mode /* 'tmz' | 'scalar' */, pml /* cells */ })
  // -> { nx, ny, dx, dy, dt, mode, Ez|u, Hx, Hy, epsR: Float64Array, sigma: Float64Array, pec: Uint8Array, ... }
export function setMaterial(grid, { x0, y0, x1, y1, epsR, sigma, pec })   // axis-aligned rect, cell units
export function addSource(grid, { i, j, kind /* 'gaussian'|'sine' */, f0, t0, spread, amp })
export function addProbe(grid, { i, j }) // -> probe id; grid.probes[id].samples: number[]
export function step(grid, n = 1)
export function snapshot(grid) // -> Float32Array of main field, length nx*ny
```
## src/gpu/ (Field-core)
`export async function createGpuStepper(grid, { backend /* 'webgpu'|'webgl2'|'auto' */ })` -> `{ backend, step(n), readField(): Promise<Float32Array>, readProbes(), destroy() }`. Same semantics as CPU `step`.

## src/signal.js (float64)
```js
export function bitsRandom(n, rng)
export function modulate(bits, { scheme /* 'bpsk'|'qpsk' */, sps }) // -> { i: Float64Array, q: Float64Array }
export function chirp({ f0, f1, T, fs })                            // -> { i, q }
export function convolveChannel(sig, h /* Float64Array real impulse response */) // -> { i, q }
export function awgn(sig, { ebn0dB, bitsPerSymbol, sps, rng })      // -> { i, q }
export function demodulate(sig, { scheme, sps })                    // -> bits
export function ber(txBits, rxBits)
export function berTheory(scheme, ebn0dB)
export function fftMag(sig)                                         // -> Float64Array
```
## src/rng.js
`export function makeRng(seed)` -> `{ uniform(), normal() }`, deterministic in Node and browser.

## UI
`index.html` imports the three modules only through the exports above. If an export is missing at runtime (parallel build), UI shows a clear "module not ready" state instead of crashing.

## Amendment r1.1 (2026-11-21, Code Lab, after QC on PR #1/#2)
- `demodulate(rx, scheme, sps, opts?)`: `opts.offset` (sample index, default 0), `opts.nSymbols` (decode exactly this many). Callers using any non-identity channel MUST pass both.
- `awgn(x, ebn0dB, k, L, opts?)`: `opts.pRef` overrides buffer mean power. Callers MUST pass signal-region power when the buffer contains padding/tails.
- `ber(txBits, rxBits)` throws on length mismatch.
- `src/rng.js` additionally exports `splitmix32`, `sfc32` (non-cryptographic; never for tokens/IDs).
- UI (owner C) MUST always pass offset, nSymbols and pRef through the channel path.
- Test command: `node --test` from repo root (Node 20+); `package.json` has `"type":"module"`.
