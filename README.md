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

- **Field:** a 512×512 TMz or scalar grid. Draw metal and dielectric, or pick a preset: free space, a PEC wall, or an εr=4 slab. Set the source in cells per wavelength. Play, pause, and reset call `step`. The GPU button calls `createGpuStepper` and falls back to the CPU reference. The label names the active backend, including `software (SwiftShader)` when the adapter is a software fallback. The footer shows frames/s for the view and solver step/s for the time spent inside `step`. A9 is recorded in [docs/fdtd-perf.md](docs/fdtd-perf.md), not in that footer.
- **Channel:** each probe's samples from `addProbe`. Record impulse runs a Gaussian pulse and freezes that trace for the signal chain. If `src/fdtd.js` is missing, the panel says "coming soon".
- **Signal:** BPSK, QPSK, or a linear chirp, convolved with the identity channel or a frozen probe, then AWGN, a constellation, BER against the theory curve, and a spectrum. Demodulation uses the main tap of the impulse response as the symbol cursor, and the noise power is taken from that symbol region. A zero error count is shown as `< 1/N`.

## Accuracy checks

| Test | Result | Reference |
|---|---|---|
| BPSK, identity channel, Eb/N0 = 6 dB, 1e6 bits | 2406 errors (BER 2.406e-3) | 2.3883e-3 |
| Courant time step | S=0.700036 | S <= 1/sqrt(2); dt = 0.99/(c sqrt(1/dx^2 + 1/dy^2)) |
| Free-space spreading | 0.28% | within 5% of r^-1/2, r=50 to 200 |
| PEC wall, normal incidence | \|R\|=0.999999 | 1 within 2% |
| PML boundary | 2.058e-9 | reflected energy under 1% (-20 dB) |
| Dielectric slab, eps_r = 4 | \|R\|=0.335932 | 1/3 within 3% |
| GPU field vs CPU (A7) | UNKNOWN on SwiftShader | rel L2 under 1e-3 after 500 steps on hardware |
| Steps/s at 512^2 (A9) | see [docs/fdtd-perf.md](docs/fdtd-perf.md) | reported there, not gated |

The 1e6-bit count is the measured BPSK run from the merged signal chain (seed 20261121, identity channel). `tests/signal.test.js` checks that run against the reference within ±20%. The FDTD rows are the numbers printed by `tests/fdtd.test.js` on main. A7 is UNKNOWN on SwiftShader. A9 is the table in [docs/fdtd-perf.md](docs/fdtd-perf.md).

`node --test`

Run that from the repo root with Node 20 or Node 22. `npm test` is the same command. Do not pass a directory: `node --test tests/` fails on Node 21 and newer because the runner treats the directory as a file. `node tests/*.test.js` without `--test` executes only the first file. GitHub Actions runs `node --test` on Node 20 and Node 22 for pull requests and for pushes to `main`, then a headless Chromium job at 1100×800.

- `tests/fdtd.test.js` checks the CPU reference: Courant dt, cylindrical spreading, PEC reflection, PML energy, and the εr=4 Fresnel reflection.
- `tests/fdtd-gpu.test.js` checks A7, GPU field against the CPU reference. It is UNKNOWN when only SwiftShader is available.
- `tests/rng.test.js` checks the seeded generator: uniform in (0, 1), normal mean and variance, and the brief's splitmix32 and sfc32 vectors.
- `tests/signal.test.js` checks modulation, convolution, AWGN scaling, demodulation on the channel cursor, BER including the 1e6-bit BPSK case, the FFT, and the chirp.
- `tests/smoke.test.js` checks the package, the page, the contract loader, the channel cursor arguments, URL bounds, and that this file names every test.
- `tests/smoke-browser.test.js` checks startup in headless Chromium at 1100×800: no console errors, a first field frame without scrolling, and that Play advances the step count. It runs when `WAP_BROWSER_SMOKE=1`.

## Roadmap

GPR mode (layered soil, buried target, B-scan radargram, benchmarked against gprMax), a validation harness exporting PNG plus an error table, then PIC particles.

`tests/dev-mocks.js` is a throwaway stand-in for the smoke tests. The page does not import it. The on-page BER preview uses 8192 bits and prints `< 1/N` when that run sees no errors. The 1e6-bit gate is `tests/signal.test.js`. A9 lives in [docs/fdtd-perf.md](docs/fdtd-perf.md).

no license yet
