# Waves and Particles

*2D waves on a grid, then the bits they carry.*

**Live app:** it runs at **[https://tbirdpower3000.github.io/Waves-and-Particles/](https://tbirdpower3000.github.io/Waves-and-Particles/)** (GitHub Pages, served from the `main` branch).

A browser GPU lab for 2D FDTD EM/RF and scalar waves, channel impulse responses, and an RF signal chain (BPSK/QPSK/chirp, AWGN, BER, spectrum).

Open `index.html` over HTTP. Paths are relative to the repo root, which is how GitHub Pages serves this folder. WebGPU is needed for the GPU path, with a CPU fallback.

```
python3 -m http.server 8000
# then visit http://localhost:8000
```

## What it does

- **Field:** a 512×512 grid in TMz or scalar mode, metal walls, dielectric rectangles, one transmitter, and up to four receiver probes. Play, pause, and reset step the CPU reference. The footer reports steps per second and `navigator.hardwareConcurrency`. The GPU button calls `createGpuStepper` when `src/gpu/` is present.
- **Channel:** each probe's impulse response, from a Gaussian pulse on the field grid.
- **Signal:** BPSK, QPSK, or a linear chirp, convolved with the identity channel or a frozen probe, then AWGN, a constellation, BER against the theory curve, and a spectrum. Demodulation uses the main tap of the impulse response as the symbol cursor, and the noise power is taken from that symbol region.
- **Field, for now:** `src/fdtd.js` is not in main yet. The field and channel panels say "coming soon". The page does not request `src/fdtd.js` or `src/gpu/`, because a missing file is a console error. Set `request: true` on those entries in `src/ui/load-contract.js` once the files are in the tree. A signal export that is absent still says "module not ready".

## Accuracy checks

| Test | Result | Reference |
|---|---|---|
| BPSK, identity channel, Eb/N0 = 6 dB, 1e6 bits | 2406 errors (BER 2.406e-3) | 2.3883e-3 |
| Courant time step | pending CI | dt <= 1/(c sqrt(1/dx^2 + 1/dy^2)) |
| Free-space spreading | pending CI | amplitude falls as r^-1/2 within 5%, 50-200 cells |
| PEC wall, normal incidence | pending CI | reflection coefficient 1 within 2% |
| PML boundary | pending CI | reflected energy under 1% (-20 dB) |
| Dielectric slab, eps_r = 4 | pending CI | Fresnel reflection 1/3 within 3% |
| GPU field vs CPU | pending CI | relative L2 under 1e-3 after 500 steps |

The 1e6-bit count is the measured BPSK run from the merged signal chain (seed 20261121, identity channel). `tests/signal.test.js` checks that run against the reference within ±20%. Field rows stay pending CI until the field solver's tests record a number.

`node --test`

Run that from the repo root with Node 20 or Node 22. `npm test` is the same command. Do not pass a directory: `node --test tests/` fails on Node 21 and newer because the runner treats the directory as a file. `node tests/*.test.js` without `--test` executes only the first file. GitHub Actions runs `node --test` on Node 20 and Node 22 for pull requests and for pushes to `main`, then a headless Chromium job at 1100×800.

- `tests/rng.test.js` checks the seeded generator: uniform in (0, 1), normal mean and variance, and the brief's splitmix32 and sfc32 vectors.
- `tests/signal.test.js` checks modulation, convolution, AWGN scaling, demodulation on the channel cursor, BER including the 1e6-bit BPSK case, the FFT, and the chirp.
- `tests/smoke.test.js` checks the package, the page, the contract loader, the channel cursor arguments, URL bounds, and that this file names every test.
- `tests/smoke-browser.test.js` checks startup in headless Chromium at 1100×800: no console errors, a first field frame, and no scrolling. It runs when `WAP_BROWSER_SMOKE=1`.

## Roadmap

GPR mode (layered soil, buried target, B-scan radargram, benchmarked against gprMax), a validation harness exporting PNG plus an error table, then PIC particles.

`src/ui/dev-mocks.js` is a throwaway stand-in for local experiments. The page does not import it. The on-page BER preview uses 1024 bits so the tab stays responsive; the 1e6-bit gate is `tests/signal.test.js`. Steps per second are shown in the field footer and are not a CI gate.

no license yet
