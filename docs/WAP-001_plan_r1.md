# WAP-001 r1 — Waves-and-Particles MVP (plan, not approved)

- Repo: https://github.com/tbirdpower3000/Waves-and-Particles (public, empty, main, created 2026-11-21)
- Owner: Code Lab (delivery). Builder: one Cursor cloud agent. Checker: Quality Control.
- Approver: Brian (launch, spend, merge, publish).

## Scope (MVP)
Single `index.html` (no build step), WebGPU compute with WebGL2 fallback, plus `tests/` run by Node in CI.
1. 2D FDTD Maxwell (TMz), 512x512 default, PML absorbing edges, draw walls (metal / dielectric eps_r), place TX source + up to 4 RX probes. Live field view.
2. Scalar wave mode (sound/water) on the same grid stepper.
3. Channel: each RX records impulse response (time series) from a Gaussian pulse.
4. Signal chain: BPSK / QPSK / linear chirp -> convolve with recorded channel -> AWGN -> demod. Show constellation, BER vs Eb/N0, spectrum (FFT).
Out of scope for r1: 3D, particles/PIC, atom mode, Unreal, GitHub Pages, accounts, analytics.

## Acceptance criteria (frozen for r1)
A1 Courant: dt <= 1/(c*sqrt(1/dx^2+1/dy^2)); test asserts the app's dt formula.
A2 Free-space spreading: 2D cylindrical wave amplitude falls ~ r^-1/2 within 5% over 50-200 cells (CPU reference stepper in tests).
A3 PEC wall reflection: normal-incidence reflection coefficient |R| = 1 within 2%.
A4 PML: reflected energy from boundary < 1% of incident (-20 dB).
A5 Dielectric slab: Fresnel |R| for eps_r=4 equals 1/3 within 3%.
A6 Signal chain: BPSK through identity channel, BER at Eb/N0 = 6 dB within ±20% of theory 2.4e-3 (>=1e6 bits).
A7 GPU path matches CPU reference field to rel. L2 < 1e-3 after 500 steps (headless check where WebGPU/WebGL available; else marked UNKNOWN, not pass).
A8 Startup: page loads with no console errors in headless Chromium at 1100x800; first frame without scrolling.
A9 Perf (reported, not gated): steps/s at 512^2 on the box GPU or software fallback, stated with hardware.
A10 CI: GitHub Actions runs `node tests/*.test.js` on every PR; README lists every test.

## Permissions
- Allowed: branch + PR writes on this repo only; read-only reference to Atomic-Modeling.
- Not allowed: merge, Pages publish, visibility change, new repos, secrets, purchases, other repos.

## Budget
One cloud agent run, Brian sets cap before launch. Max 2 repair cycles after QC.

## Handoff
Builder returns: PR URL, head SHA, actual test output. Code Lab -> QC REPORT BACK on that SHA. Brian accepts/merges.
