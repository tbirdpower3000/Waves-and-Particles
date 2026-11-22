import assert from 'node:assert/strict';
import { test } from 'node:test';
import { runGpuSignalCheck } from './gpu-signal-harness.js';
import { makeRng } from '../src/rng.js';
import {
  awgn,
  ber,
  berTheory,
  bitsRandom,
  chirp,
  convolveChannel,
  demodulate,
  fftMag,
  modulate,
} from '../src/signal.js';

// Q(sqrt(2*10^(dB/10))) from a libm erfc. berTheory must land on these.
const THEORY = {
  6: 2.3882907809328075e-3,
  9: 3.3627228419617485e-5,
  10: 3.8721082155220370e-6,
};

// Fixed-seed CI link. Counts are for this generator, I-then-Q noise, V8.
// Binomial band at 6 dB, N = 200_000, false-fail ~9e-5: [395, 565].
const SEED = 20261121;
const N_CI = 200_000;
const BAND_LO = 395;
const BAND_HI = 565;
const GOLDEN = {
  bpskL1: 468,
  bpskL8: 483,
  qpskL1: 467,
  qpskL8: 502,
};

const UNIT = new Float64Array([1]);
// Single tap. ||h||^2 = 2 exactly in intent; Math.SQRT2**2 is 2 + 1 ulp.
const GAIN = new Float64Array([Math.SQRT2]);

// chi^2 acceptance for mean(noise^2) / sigma^2 at 1e5 real samples.
const CHI_LO = 0.9827;
const CHI_HI = 1.0175;

function runLink({ scheme, sps, nBits, seed, ebn0dB = 6, h = UNIT, noise = true }) {
  const rng = makeRng(seed);
  const tx = bitsRandom(nBits, rng);
  const ch = convolveChannel(modulate(tx, { scheme, sps }), h);
  const wave = noise
    ? awgn(ch, { ebn0dB, bitsPerSymbol: scheme === 'qpsk' ? 2 : 1, sps, rng })
    : ch;
  const rx = demodulate(wave, { scheme, sps });
  let errors = 0;
  const m = Math.min(tx.length, rx.length);
  for (let i = 0; i < m; i++) {
    if ((tx[i] ? 1 : 0) !== (rx[i] ? 1 : 0)) errors++;
  }
  errors += Math.abs(tx.length - rx.length);
  return { errors, tx, rx };
}

function assertBand(errors) {
  assert.ok(errors >= BAND_LO && errors <= BAND_HI, `errors ${errors} outside [${BAND_LO}, ${BAND_HI}]`);
}

function meanPower(sig) {
  let energy = 0;
  for (let i = 0; i < sig.i.length; i++) {
    energy += sig.i[i] * sig.i[i] + sig.q[i] * sig.q[i];
  }
  return energy / sig.i.length;
}

function regionPower(sig, offset, len) {
  let energy = 0;
  for (let i = 0; i < len; i++) {
    const at = offset + i;
    energy += sig.i[at] * sig.i[at] + sig.q[at] * sig.q[at];
  }
  return energy / len;
}

function normalizedTaps(taps) {
  let energy = 0;
  for (const tap of taps) energy += tap * tap;
  const scale = Math.sqrt(energy);
  return Float64Array.from(taps, (tap) => tap / scale);
}

// BPSK, sps 8, N = 200k, 6 dB. offset is the main-tap index.
// pRef is the power of the samples demodulate reads, not the padded buffer.
function errorsAligned({ h, offset }) {
  const sps = 8;
  const nBits = N_CI;
  const rng = makeRng(SEED);
  const tx = bitsRandom(nBits, rng);
  const ch = convolveChannel(modulate(tx, { scheme: 'bpsk', sps }), h);
  const span = nBits * sps;
  const pRef = regionPower(ch, offset, span);
  const noisy = awgn(ch, { ebn0dB: 6, bitsPerSymbol: 1, sps, rng, pRef });
  const rx = demodulate(noisy, { scheme: 'bpsk', sps, offset, nSymbols: nBits });
  let errors = 0;
  for (let i = 0; i < tx.length; i++) if (tx[i] !== rx[i]) errors++;
  return { errors, pRef, fullPower: meanPower(ch) };
}

// Ratio of measured per-rail noise power to Pbar*L / (2*k*gamma).
function noiseRatios(sig, { ebn0dB, bitsPerSymbol, sps, seed }) {
  const y = awgn(sig, { ebn0dB, bitsPerSymbol, sps, rng: makeRng(seed) });
  const n = sig.i.length;
  let accI = 0;
  let accQ = 0;
  for (let i = 0; i < n; i++) {
    const di = y.i[i] - sig.i[i];
    const dq = y.q[i] - sig.q[i];
    accI += di * di;
    accQ += dq * dq;
  }
  const expected = (meanPower(sig) * sps) / (2 * bitsPerSymbol * 10 ** (ebn0dB / 10));
  return { i: (accI / n) / expected, q: (accQ / n) / expected };
}

function assertChi(ratios, label) {
  assert.ok(ratios.i >= CHI_LO && ratios.i <= CHI_HI, `${label} I ${ratios.i}`);
  assert.ok(ratios.q >= CHI_LO && ratios.q <= CHI_HI, `${label} Q ${ratios.q}`);
}

test('theoretical BER matches BPSK and Gray QPSK at 6, 9 and 10 dB', () => {
  for (const db of [6, 9, 10]) {
    const bpsk = berTheory('bpsk', db);
    const qpsk = berTheory('qpsk', db);
    assert.equal(bpsk, qpsk);
    const rel = Math.abs(bpsk - THEORY[db]) / THEORY[db];
    assert.ok(rel < 1e-6, `${db} dB rel ${rel} got ${bpsk}`);
  }
});

test('ber() counts disagreements and throws on a length mismatch', () => {
  assert.equal(ber([0, 1, 0, 1], [0, 1, 0, 1]), 0);
  assert.equal(ber([0, 1, 0, 1], [0, 1, 1, 1]), 0.25);
  assert.equal(ber([0, 0], [1, 1]), 1);
  assert.equal(ber([], []), 0);
  assert.throws(() => ber([0, 1, 0], [0, 1]), /bit length mismatch: 3 vs 2/);
});

test('bitsRandom is reproducible and binary', () => {
  const a = bitsRandom(1000, makeRng(5));
  const b = bitsRandom(1000, makeRng(5));
  assert.equal(a.length, 1000);
  assert.deepEqual(a, b);
  for (let i = 0; i < a.length; i++) assert.ok(a[i] === 0 || a[i] === 1);
});

test('noiseless round trip is error-free for BPSK and QPSK', () => {
  for (const scheme of ['bpsk', 'qpsk']) {
    for (const sps of [1, 4, 8]) {
      const { errors } = runLink({ scheme, sps, nBits: 8192, seed: 7, noise: false });
      assert.equal(errors, 0, `${scheme} sps=${sps}`);
    }
  }
  // Odd QPSK length: the trailing bit is not a symbol.
  const tx = Uint8Array.of(0, 1, 1);
  const rx = demodulate(modulate(tx, { scheme: 'qpsk', sps: 2 }), { scheme: 'qpsk', sps: 2 });
  assert.deepEqual([...rx], [0, 1]);
});

test('convolution with a unit impulse is identity', () => {
  const tx = bitsRandom(64, makeRng(3));
  const sig = modulate(tx, { scheme: 'qpsk', sps: 4 });
  const i0 = sig.i[0];
  const q0 = sig.q[3];
  const y = convolveChannel(sig, UNIT);
  assert.equal(sig.i[0], i0);
  assert.equal(sig.q[3], q0);
  assert.equal(y.i.length, sig.i.length);
  assert.equal(y.q.length, sig.q.length);
  for (let i = 0; i < sig.i.length; i++) {
    assert.equal(y.i[i], sig.i[i]);
    assert.equal(y.q[i], sig.q[i]);
  }
  const known = convolveChannel(
    { i: Float64Array.of(1, 2, 3), q: Float64Array.of(0, -1, 4) },
    Float64Array.of(1, -1),
  );
  assert.deepEqual([...known.i], [1, 1, 1, -3]);
  assert.deepEqual([...known.q], [0, -1, 5, -4]);
});

test('A6 BPSK identity channel at 6 dB is within ±20% of theory', () => {
  const nBits = 1_000_000;
  const { tx, rx } = runLink({ scheme: 'bpsk', sps: 1, nBits, seed: SEED });
  assert.equal(tx.length, nBits);
  const rate = ber(tx, rx);
  const rel = Math.abs(rate - THEORY[6]) / THEORY[6];
  assert.ok(rel <= 0.20, `BER ${rate} rel ${rel}`);
});

test('QPSK at 6 dB matches BPSK BER per bit', () => {
  const nBits = 1_000_000;
  const { tx, rx } = runLink({ scheme: 'qpsk', sps: 1, nBits, seed: SEED });
  const rate = ber(tx, rx);
  const rel = Math.abs(rate - THEORY[6]) / THEORY[6];
  assert.ok(rel <= 0.20, `BER ${rate} rel ${rel}`);
  assert.equal(berTheory('qpsk', 6), berTheory('bpsk', 6));
});

test('BPSK L=1 N=200k golden error count and binomial band', () => {
  const { errors, tx, rx } = runLink({ scheme: 'bpsk', sps: 1, nBits: N_CI, seed: SEED });
  assert.equal(errors, GOLDEN.bpskL1);
  assertBand(errors);
  assert.equal(ber(tx, rx), errors / N_CI);
});

test('BPSK L=8 N=200k golden error count and binomial band', () => {
  // Ignoring L makes sigma too small by sqrt(8) (~+9 dB) and the count falls to ~0.
  const { errors } = runLink({ scheme: 'bpsk', sps: 8, nBits: N_CI, seed: SEED });
  assert.equal(errors, GOLDEN.bpskL8);
  assertBand(errors);
});

test('QPSK L=1 N=200k golden error count and binomial band', () => {
  // Using Es/N0 as Eb/N0 (k omitted) is -3 dB and the count jumps to ~4600.
  const { errors } = runLink({ scheme: 'qpsk', sps: 1, nBits: N_CI, seed: SEED });
  assert.equal(errors, GOLDEN.qpskL1);
  assertBand(errors);
});

test('QPSK L=8 N=200k golden error count and binomial band', () => {
  const { errors } = runLink({ scheme: 'qpsk', sps: 8, nBits: N_CI, seed: SEED });
  assert.equal(errors, GOLDEN.qpskL8);
  assertBand(errors);
});

test('trap: N0/2 per rail and 10^(dB/10), not N0 or 10^(dB/20)', () => {
  // 1e5 samples per rail. Missing /2 lands near ratio 2.
  // 10^(9/20) instead of 10^(9/10) lands near 10^0.45 ≈ 2.82.
  const n = 100_000;
  const bits = new Uint8Array(n);
  for (let i = 0; i < n; i++) bits[i] = i & 1;
  const sig = modulate(bits, { scheme: 'bpsk', sps: 1 });
  const ratios = noiseRatios(sig, { ebn0dB: 9, bitsPerSymbol: 1, sps: 1, seed: SEED });
  assertChi(ratios, 'bpsk 9 dB');
});

test('trap: QPSK uses k = 2 so Eb is Es/2', () => {
  // k omitted doubles sigma^2 (ratio ~2), outside the chi^2 band.
  const nSym = 100_000;
  const bits = new Uint8Array(nSym * 2);
  for (let i = 0; i < bits.length; i++) bits[i] = i & 1;
  const sig = modulate(bits, { scheme: 'qpsk', sps: 1 });
  assert.ok(Math.abs(meanPower(sig) - 2) < 1e-12);
  const ratios = noiseRatios(sig, { ebn0dB: 6, bitsPerSymbol: 2, sps: 1, seed: SEED });
  assertChi(ratios, 'qpsk k=2');
});

test('trap: oversampling L multiplies symbol energy', () => {
  // Dropping L divides sigma^2 by 8 here (ratio ~0.125).
  const nSym = 100_000 / 8;
  const bits = new Uint8Array(nSym);
  for (let i = 0; i < nSym; i++) bits[i] = i & 1;
  const sig = modulate(bits, { scheme: 'bpsk', sps: 8 });
  assert.equal(sig.i.length, 100_000);
  const ratios = noiseRatios(sig, { ebn0dB: 6, bitsPerSymbol: 1, sps: 8, seed: SEED });
  assertChi(ratios, 'bpsk L=8');
});

test('trap: channel gain ||h||^2 = 2 is absorbed by received Pbar', () => {
  // Noise sized for the pre-channel waveform is +3 dB at the receiver (~5 errors).
  const link = { scheme: 'bpsk', sps: 1, nBits: N_CI, seed: SEED, ebn0dB: 6 };
  const gained = runLink({ ...link, h: GAIN });
  assert.equal(gained.errors, GOLDEN.bpskL1);
  assertBand(gained.errors);

  const n = 100_000;
  const bits = new Uint8Array(n);
  for (let i = 0; i < n; i++) bits[i] = i & 1;
  const rx = convolveChannel(modulate(bits, { scheme: 'bpsk', sps: 1 }), GAIN);
  assert.ok(Math.abs(meanPower(rx) - 2) < 1e-9, `Pbar ${meanPower(rx)}`);
  const ratios = noiseRatios(rx, { ebn0dB: 6, bitsPerSymbol: 1, sps: 1, seed: 11 });
  assertChi(ratios, 'gain');
});

test('BPSK 6 dB stays in band through a 3-sample delay', () => {
  const h = new Float64Array([0, 0, 0, 1]);
  const { errors } = errorsAligned({ h, offset: 3 });
  assertBand(errors);
});

test('BPSK 6 dB stays in band through a 2-tap unit-energy channel', () => {
  // Main cursor at tap 0. The second tap is real ISI; a half-amplitude
  // second tap (592 errors) leaves the uncoded AWGN band, a quarter stays in it.
  const h = normalizedTaps([1, 0.25]);
  let energy = 0;
  for (const tap of h) energy += tap * tap;
  assert.ok(Math.abs(energy - 1) < 1e-12, `||h||^2 ${energy}`);
  assert.ok(h[0] > h[1] && h[1] > 0);
  const { errors } = errorsAligned({ h, offset: 0 });
  assertBand(errors);
});

test('BPSK 6 dB stays in band with 1999 trailing zero taps', () => {
  const h = new Float64Array(2000);
  h[0] = 1;
  const { errors, pRef, fullPower } = errorsAligned({ h, offset: 0 });
  assert.ok(fullPower < pRef, `full ${fullPower} pRef ${pRef}`);
  assertBand(errors);
});

test('pRef is the signal-region power, not the zero-padded buffer mean', () => {
  const n = 100_000;
  const i = new Float64Array(n * 2);
  const q = new Float64Array(n * 2);
  for (let k = 0; k < n; k++) i[k] = 1;
  const sig = { i, q };
  const y = awgn(sig, { ebn0dB: 9, bitsPerSymbol: 1, sps: 1, rng: makeRng(SEED), pRef: 1 });
  let accI = 0;
  let accQ = 0;
  for (let k = 0; k < n; k++) {
    const di = y.i[k] - sig.i[k];
    const dq = y.q[k] - sig.q[k];
    accI += di * di;
    accQ += dq * dq;
  }
  const expected = 1 / (2 * 10 ** 0.9);
  assertChi({ i: (accI / n) / expected, q: (accQ / n) / expected }, 'pRef');
});

test('FFT of a pure tone peaks at that bin', () => {
  const n = 64;
  const k0 = 7;
  const i = new Float64Array(n);
  const q = new Float64Array(n);
  for (let t = 0; t < n; t++) i[t] = Math.cos((2 * Math.PI * k0 * t) / n);
  const mag = fftMag({ i, q });
  assert.equal(mag.length, n);
  let peak = 0;
  for (let k = 1; k < n; k++) if (mag[k] > mag[peak]) peak = k;
  assert.ok(peak === k0 || peak === n - k0, `peak bin ${peak}`);
  assert.ok(Math.abs(mag[k0] - mag[n - k0]) / mag[k0] < 1e-9);
  assert.ok(Math.abs(mag[k0] - n / 2) / (n / 2) < 1e-9);
  for (let k = 0; k < n; k++) {
    if (k === k0 || k === n - k0) continue;
    assert.ok(mag[k] < 1e-8 * mag[k0], `leak bin ${k} ${mag[k]}`);
  }

  const ic = new Float64Array(n);
  const qc = new Float64Array(n);
  const k1 = 5;
  for (let t = 0; t < n; t++) {
    const th = (2 * Math.PI * k1 * t) / n;
    ic[t] = Math.cos(th);
    qc[t] = Math.sin(th);
  }
  const magc = fftMag({ i: ic, q: qc });
  let peakC = 0;
  for (let k = 1; k < n; k++) if (magc[k] > magc[peakC]) peakC = k;
  assert.equal(peakC, k1);
  assert.ok(Math.abs(magc[k1] - n) / n < 1e-9);
  for (let k = 0; k < n; k++) {
    if (k === k1) continue;
    assert.ok(magc[k] < 1e-8 * magc[k1]);
  }
  assert.equal(i[0], 1);
});

test('linear chirp sweeps from f0 to f1 at unit amplitude', () => {
  const fs = 48_000;
  const f0 = 1000;
  const f1 = 5000;
  const T = 0.05;
  const sig = chirp({ f0, f1, T, fs });
  const n = sig.i.length;
  assert.equal(n, Math.round(T * fs));
  assert.equal(sig.i[0], 1);
  assert.equal(sig.q[0], 0);
  for (let t = 0; t < n; t++) {
    const power = sig.i[t] * sig.i[t] + sig.q[t] * sig.q[t];
    assert.ok(Math.abs(power - 1) < 1e-10);
  }
  const rate = (f1 - f0) / T;
  const freqAt = (sample) => {
    const cross = sig.i[sample] * sig.q[sample + 1] - sig.q[sample] * sig.i[sample + 1];
    const dot = sig.i[sample] * sig.i[sample + 1] + sig.q[sample] * sig.q[sample + 1];
    return (Math.atan2(cross, dot) * fs) / (2 * Math.PI);
  };
  // Finite difference of a linear chirp is the frequency at the midpoint of the step.
  const expectAt = (sample) => f0 + rate * ((sample + 0.5) / fs);
  assert.ok(Math.abs(freqAt(0) - expectAt(0)) < 1e-4);
  assert.ok(Math.abs(freqAt(n - 2) - expectAt(n - 2)) < 1e-3);
  assert.ok(freqAt(n - 2) > freqAt(0));
});

test('GPU signal chain matches the CPU reference', { timeout: 180_000 }, async (t) => {
  const result = await runGpuSignalCheck();
  if (!result || result.unknown) {
    console.log(`UNKNOWN: no GPU adapter${result?.reason ? ` (${result.reason})` : ''}`);
    t.skip('UNKNOWN: no GPU adapter');
    return;
  }
  if (result.error) assert.fail(result.error);
  const adapter = result.adapter;
  const kind = adapter.software ? 'software' : 'hardware';
  console.log(
    `adapter: ${adapter.description} vendor=${adapter.vendor || 'n/a'} device=${adapter.device || 'n/a'} arch=${adapter.architecture || 'n/a'} ${kind} fallback=${adapter.fallback}`,
  );
  const bpskRate = result.bpskErrors / result.nBits;
  const qpskRate = result.qpskErrors / result.nBits;
  const bpskRel = Math.abs(bpskRate - THEORY[6]) / THEORY[6];
  const qpskRel = Math.abs(qpskRate - THEORY[6]) / THEORY[6];
  const gpuBits = result.nBits * 2;
  console.log(
    `GPU BPSK N=${result.nBits} errors=${result.bpskErrors} BER=${bpskRate.toExponential(4)} rel=${(bpskRel * 100).toFixed(2)}%`,
  );
  console.log(
    `GPU QPSK N=${result.nBits} errors=${result.qpskErrors} BER=${qpskRate.toExponential(4)} rel=${(qpskRel * 100).toFixed(2)}%`,
  );
  console.log(
    `bits/s GPU=${(gpuBits / (result.gpuMs / 1000)).toExponential(3)} over ${result.gpuMs.toFixed(0)} ms (BPSK+QPSK)  CPU=${(result.nBits / (result.cpuMs / 1000)).toExponential(3)} over ${result.cpuMs.toFixed(0)} ms (BPSK)`,
  );
  console.log(`FFT relL2=${result.fftRel.toExponential(3)}  FIR relL2=${result.firRel.toExponential(3)}  modem mismatches=${result.roundtripMismatches}  CPU errors=${result.cpuErrors}`);
  assert.equal(result.roundtripMismatches, 0);
  assert.ok(result.nBits >= 1_000_000);
  assert.ok(bpskRel <= 0.20, `BPSK BER ${bpskRate}`);
  assert.ok(qpskRel <= 0.20, `QPSK BER ${qpskRate}`);
  assert.ok(result.fftRel < 1e-4, `FFT rel ${result.fftRel}`);
  assert.ok(result.firRel < 1e-4, `FIR rel ${result.firRel}`);
  assert.equal(result.limitBits, true);
  assert.equal(result.limitMod, true);
  assert.equal(result.limitSweep, true);
  assert.equal(result.badOffset, true);
  assert.equal(result.badSymbols, true);
  assert.equal(result.badK, true);
  const halfRate = result.halfErrors / result.halfBits;
  const loudRate = result.loudErrors / result.loudBits;
  assert.ok(Math.abs(halfRate - THEORY[6]) / THEORY[6] <= 0.25, `h=0.5 BER ${halfRate}`);
  assert.ok(Math.abs(loudRate - THEORY[6]) / THEORY[6] <= 0.25, `h=2 BER ${loudRate}`);
  assert.ok(result.swapped < 8, `seed^1 swapped I/Q on ${result.swapped} samples`);
  assert.ok(result.randRel < 1e-4, `random FFT rel ${result.randRel}`);
  assert.equal(result.liveBefore, 0);
  assert.equal(result.liveAfter, 0);
  assert.equal(typeof result.afterDestroy, 'number');
  console.log(
    `limits=${result.limitBits}/${result.limitMod}/${result.limitSweep} offset=${result.badOffset} nSymbols=${result.badSymbols} k=${result.badK} h=0.5 errors=${result.halfErrors}/${result.halfBits} h=2 errors=${result.loudErrors}/${result.loudBits} swapped=${result.swapped} randFFT=${result.randRel.toExponential(3)}`,
  );
  if (adapter.software) {
    console.log('UNKNOWN: software adapter (adapter.info SwiftShader or fallback); not a real-hardware pass');
    t.skip('UNKNOWN: software GPU adapter');
  }
});
