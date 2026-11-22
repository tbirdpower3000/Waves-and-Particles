/**
 * Baseband signal chain: random bits, BPSK/QPSK, linear chirp, real-channel
 * convolution, AWGN, integrate-and-dump demod, BER, and FFT magnitude.
 * Float64 samples. No runtime dependencies. Same module in the browser and Node.
 *
 * Bits are Uint8Array of 0 and 1. Waveforms are { i: Float64Array, q: Float64Array }.
 */

function requireSps(sps) {
  if (!Number.isInteger(sps) || sps < 1) {
    throw new Error(`sps must be a positive integer, got ${sps}`);
  }
}

function requireScheme(scheme) {
  if (scheme !== 'bpsk' && scheme !== 'qpsk') {
    throw new Error(`unknown scheme: ${scheme}`);
  }
}

function requirePair(sig) {
  if (!sig || sig.i == null || sig.q == null) {
    throw new Error('signal must be { i, q }');
  }
  if (sig.i.length !== sig.q.length) {
    throw new Error(`I/Q length mismatch: ${sig.i.length} vs ${sig.q.length}`);
  }
  return sig.i.length;
}

/**
 * @param {number} n
 * @param {{ uniform: () => number }} rng
 * @returns {Uint8Array}
 */
export function bitsRandom(n, rng) {
  if (!Number.isInteger(n) || n < 0) {
    throw new Error(`n must be a non-negative integer, got ${n}`);
  }
  const bits = new Uint8Array(n);
  for (let i = 0; i < n; i++) bits[i] = rng.uniform() < 0.5 ? 0 : 1;
  return bits;
}

/**
 * Rectangular pulse shaping. BPSK: 0 -> +1, 1 -> -1 on I, Q = 0.
 * QPSK: bit pairs (even, odd) drive I and Q the same way, which is Gray
 * labeling, so each bit has the BPSK error rate. A trailing odd QPSK bit
 * is not transmitted.
 *
 * @param {ArrayLike<number>} bits
 * @param {{ scheme: 'bpsk' | 'qpsk', sps: number }} opts
 * @returns {{ i: Float64Array, q: Float64Array }}
 */
export function modulate(bits, { scheme, sps }) {
  requireScheme(scheme);
  requireSps(sps);
  const k = scheme === 'qpsk' ? 2 : 1;
  const nSym = Math.floor(bits.length / k);
  const n = nSym * sps;
  const i = new Float64Array(n);
  const q = new Float64Array(n);
  if (scheme === 'bpsk') {
    for (let s = 0; s < nSym; s++) {
      const amp = bits[s] ? -1 : 1;
      const off = s * sps;
      for (let t = 0; t < sps; t++) i[off + t] = amp;
    }
  } else {
    for (let s = 0; s < nSym; s++) {
      const ampI = bits[2 * s] ? -1 : 1;
      const ampQ = bits[2 * s + 1] ? -1 : 1;
      const off = s * sps;
      for (let t = 0; t < sps; t++) {
        i[off + t] = ampI;
        q[off + t] = ampQ;
      }
    }
  }
  return { i, q };
}

/**
 * Complex baseband linear chirp, unit amplitude.
 * Instantaneous frequency runs from f0 at t = 0 to f1 at t = T.
 * Sample count is round(T * fs).
 *
 * @param {{ f0: number, f1: number, T: number, fs: number }} opts
 * @returns {{ i: Float64Array, q: Float64Array }}
 */
export function chirp({ f0, f1, T, fs }) {
  if (!(fs > 0) || !(T > 0) || !Number.isFinite(f0) || !Number.isFinite(f1)) {
    throw new Error('chirp requires finite f0, f1 and positive T, fs');
  }
  const n = Math.max(1, Math.round(T * fs));
  const i = new Float64Array(n);
  const q = new Float64Array(n);
  const rate = (f1 - f0) / T;
  for (let nIdx = 0; nIdx < n; nIdx++) {
    const t = nIdx / fs;
    const phase = 2 * Math.PI * (f0 * t + 0.5 * rate * t * t);
    i[nIdx] = Math.cos(phase);
    q[nIdx] = Math.sin(phase);
  }
  return { i, q };
}

/**
 * Full linear convolution of a complex baseband waveform with a real
 * impulse response. Output length is sig.length + h.length - 1.
 * h is not energy-normalized; awgn measures whatever power arrives.
 *
 * @param {{ i: ArrayLike<number>, q: ArrayLike<number> }} sig
 * @param {ArrayLike<number>} h
 * @returns {{ i: Float64Array, q: Float64Array }}
 */
export function convolveChannel(sig, h) {
  const n = requirePair(sig);
  const m = h.length;
  if (n === 0 || m === 0) {
    return { i: new Float64Array(0), q: new Float64Array(0) };
  }
  const outLen = n + m - 1;
  const i = new Float64Array(outLen);
  const q = new Float64Array(outLen);
  const srcI = sig.i;
  const srcQ = sig.q;
  for (let k = 0; k < m; k++) {
    const hk = h[k];
    if (hk === 0) continue;
    for (let nIdx = 0; nIdx < n; nIdx++) {
      const at = nIdx + k;
      i[at] += hk * srcI[nIdx];
      q[at] += hk * srcQ[nIdx];
    }
  }
  return { i, q };
}

/**
 * Add complex AWGN at a requested Eb/N0.
 *
 * Discrete-time complex baseband, sample period normalized to 1.
 * Pbar = mean_n (i[n]^2 + q[n]^2) is measured on this buffer, so Eb is the
 * energy at the receiver (after the channel), not a unit-power assumption.
 * A rectangular symbol occupies L = sps samples:
 *   Es = Pbar * L,   Eb = Es / k,   k = bitsPerSymbol.
 * White noise has two-sided density N0/2 on each real axis. At fs = 1 that
 * density is the per-sample variance, so each of I and Q gets N(0, sigma^2)
 * with sigma^2 = N0/2. gamma = Eb/N0 = 10^(ebn0dB/10) is a power ratio
 * (not 10^(dB/20)):
 *   N0 = Eb / gamma
 *   sigma^2 = Pbar * L / (2 * k * gamma).
 * The same sigma is used on I and on Q.
 *
 * Checks. BPSK with i = ±1, q = 0 (Pbar = 1, k = 1): sigma^2 = L / (2 gamma).
 * Integrate-and-dump sums L samples, signal ±L, noise variance L*sigma^2,
 * decision SNR = L/sigma^2 = 2 gamma, so Pb = Q(sqrt(2 gamma)).
 * QPSK with i = ±1, q = ±1 (Pbar = 2, k = 2): sigma^2 = L / (2 gamma) as well.
 * Each rail is an independent BPSK bit (Gray labeling), so the BER per bit
 * equals BPSK.
 *
 * Traps this rejects: dropping the 2 (sigma^2 = N0 per rail) is -3 dB;
 * omitting k uses Es/N0 as Eb/N0 and is -3 dB for QPSK; omitting L understates
 * sigma by sqrt(L) (+10 log10(L) dB, about +9 dB at L = 8). A pure channel
 * gain scales Pbar, so sigma tracks the received amplitude when pRef is omitted.
 * Multipath is not equalized; the Q(·) curve assumes one cursor, no ISI.
 *
 * pRef, when passed, replaces the measured buffer mean. Pass the mean power of
 * the symbol region demodulate will read (offset through nSymbols * sps
 * samples). A convolution tail of zeros, or a run of leading zeros from a
 * delay, lowers the mean of the whole buffer and would otherwise shrink sigma.
 *
 * @param {{ i: ArrayLike<number>, q: ArrayLike<number> }} sig
 * @param {{ ebn0dB: number, bitsPerSymbol: number, sps: number, rng: { normal: () => number }, pRef?: number }} opts
 * @returns {{ i: Float64Array, q: Float64Array }}
 */
export function awgn(sig, { ebn0dB, bitsPerSymbol, sps, rng, pRef }) {
  const n = requirePair(sig);
  requireSps(sps);
  if (!Number.isFinite(ebn0dB)) {
    throw new Error(`ebn0dB must be finite, got ${ebn0dB}`);
  }
  if (!(bitsPerSymbol > 0) || !Number.isFinite(bitsPerSymbol)) {
    throw new Error(`bitsPerSymbol must be positive, got ${bitsPerSymbol}`);
  }
  const i = new Float64Array(n);
  const q = new Float64Array(n);
  if (n === 0) return { i, q };

  const srcI = sig.i;
  const srcQ = sig.q;
  let pBar;
  if (pRef !== undefined) {
    if (typeof pRef !== 'number' || !Number.isFinite(pRef) || pRef < 0) {
      throw new Error(`pRef must be a finite number >= 0, got ${pRef}`);
    }
    pBar = pRef;
  } else {
    let energy = 0;
    for (let nIdx = 0; nIdx < n; nIdx++) {
      energy += srcI[nIdx] * srcI[nIdx] + srcQ[nIdx] * srcQ[nIdx];
    }
    pBar = energy / n;
  }
  const gamma = 10 ** (ebn0dB / 10);
  const sigma = Math.sqrt((pBar * sps) / (2 * bitsPerSymbol * gamma));
  for (let nIdx = 0; nIdx < n; nIdx++) {
    i[nIdx] = srcI[nIdx] + sigma * rng.normal();
    q[nIdx] = srcQ[nIdx] + sigma * rng.normal();
  }
  return { i, q };
}

/**
 * Integrate-and-dump. BPSK reads I. QPSK reads I then Q, inverse of modulate().
 * `offset` is the first sample of symbol 0 (default 0). `nSymbols`, when
 * passed, is how many symbols to decode; the convolution tail after that
 * window is ignored. When omitted, every full symbol that fits from `offset`
 * to the end of the buffer is decoded.
 *
 * @param {{ i: ArrayLike<number>, q: ArrayLike<number> }} sig
 * @param {{ scheme: 'bpsk' | 'qpsk', sps: number, offset?: number, nSymbols?: number }} opts
 * @returns {Uint8Array}
 */
export function demodulate(sig, { scheme, sps, offset = 0, nSymbols } = {}) {
  requireScheme(scheme);
  requireSps(sps);
  const n = requirePair(sig);
  if (!Number.isInteger(offset) || offset < 0 || offset > n) {
    throw new Error(`offset must be an integer in [0, ${n}], got ${offset}`);
  }
  const room = Math.floor((n - offset) / sps);
  if (nSymbols !== undefined && (!Number.isInteger(nSymbols) || nSymbols < 0)) {
    throw new Error(`nSymbols must be a non-negative integer, got ${nSymbols}`);
  }
  const nSym = nSymbols === undefined ? room : nSymbols;
  if (nSym > room) {
    throw new Error(`nSymbols ${nSym} does not fit at offset ${offset} (room for ${room})`);
  }
  const srcI = sig.i;
  const srcQ = sig.q;
  if (scheme === 'bpsk') {
    const bits = new Uint8Array(nSym);
    for (let s = 0; s < nSym; s++) {
      let acc = 0;
      const off = offset + s * sps;
      for (let t = 0; t < sps; t++) acc += srcI[off + t];
      bits[s] = acc >= 0 ? 0 : 1;
    }
    return bits;
  }
  const bits = new Uint8Array(nSym * 2);
  for (let s = 0; s < nSym; s++) {
    let accI = 0;
    let accQ = 0;
    const off = offset + s * sps;
    for (let t = 0; t < sps; t++) {
      accI += srcI[off + t];
      accQ += srcQ[off + t];
    }
    bits[2 * s] = accI >= 0 ? 0 : 1;
    bits[2 * s + 1] = accQ >= 0 ? 0 : 1;
  }
  return bits;
}

/**
 * Bit error rate. Lengths must match: a demodulator that emitted the
 * convolution tail as extra bits is a caller error, not a BER.
 * @param {ArrayLike<number>} txBits
 * @param {ArrayLike<number>} rxBits
 * @returns {number}
 */
export function ber(txBits, rxBits) {
  if (txBits.length !== rxBits.length) {
    throw new Error(`bit length mismatch: ${txBits.length} vs ${rxBits.length}`);
  }
  const n = txBits.length;
  if (n === 0) return 0;
  let errors = 0;
  for (let i = 0; i < n; i++) {
    if ((txBits[i] ? 1 : 0) !== (rxBits[i] ? 1 : 0)) errors++;
  }
  return errors / n;
}

// Numerical Recipes erfc, fractional error ~1e-7. Direct tail, so high
// Eb/N0 does not cancel through 1 - erf(x).
function erfc(x) {
  const z = Math.abs(x);
  const t = 1 / (1 + 0.5 * z);
  const ans = t * Math.exp(
    -z * z - 1.26551223 +
    t * (1.00002368 +
    t * (0.37409196 +
    t * (0.09678418 +
    t * (-0.18628806 +
    t * (0.27886807 +
    t * (-1.13520398 +
    t * (1.48851587 +
    t * (-0.82215223 +
    t * 0.17087277))))))))
  );
  return x >= 0 ? ans : 2 - ans;
}

/**
 * Uncoded AWGN BER per bit. Gray QPSK equals BPSK:
 * Pb = Q(sqrt(2 Eb/N0)) = 0.5 * erfc(sqrt(10^(ebn0dB/10))).
 *
 * @param {'bpsk' | 'qpsk'} scheme
 * @param {number} ebn0dB
 * @returns {number}
 */
export function berTheory(scheme, ebn0dB) {
  requireScheme(scheme);
  if (!Number.isFinite(ebn0dB)) {
    throw new Error(`ebn0dB must be finite, got ${ebn0dB}`);
  }
  const gamma = 10 ** (ebn0dB / 10);
  return 0.5 * erfc(Math.sqrt(gamma));
}

function nextPow2(n) {
  let p = 1;
  while (p < n) {
    p *= 2;
    if (p > 2 ** 30) throw new Error('fft length exceeds 2^30');
  }
  return p;
}

// In-place radix-2 DIT FFT, X[k] = sum_n x[n] exp(-j 2π k n / N).
function fft(re, im) {
  const n = re.length;
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >>> 1;
    for (; j & bit; bit >>>= 1) j ^= bit;
    j ^= bit;
    if (i < j) {
      const tr = re[i];
      re[i] = re[j];
      re[j] = tr;
      const ti = im[i];
      im[i] = im[j];
      im[j] = ti;
    }
  }
  for (let len = 2; len <= n; len *= 2) {
    const half = len >>> 1;
    const ang = -2 * Math.PI / len;
    const wRe = Math.cos(ang);
    const wIm = Math.sin(ang);
    for (let i = 0; i < n; i += len) {
      let curRe = 1;
      let curIm = 0;
      for (let k = 0; k < half; k++) {
        const oddRe = re[i + k + half] * curRe - im[i + k + half] * curIm;
        const oddIm = re[i + k + half] * curIm + im[i + k + half] * curRe;
        const evenRe = re[i + k];
        const evenIm = im[i + k];
        re[i + k] = evenRe + oddRe;
        im[i + k] = evenIm + oddIm;
        re[i + k + half] = evenRe - oddRe;
        im[i + k + half] = evenIm - oddIm;
        const nextRe = curRe * wRe - curIm * wIm;
        curIm = curRe * wIm + curIm * wRe;
        curRe = nextRe;
      }
    }
  }
}

/**
 * Unnormalized radix-2 DFT magnitude. Zero-pads to the next power of two.
 * A complex exponential exp(+j 2π k n / N) peaks in bin k.
 *
 * @param {{ i: ArrayLike<number>, q: ArrayLike<number> }} sig
 * @returns {Float64Array}
 */
export function fftMag(sig) {
  const n0 = requirePair(sig);
  if (n0 === 0) return new Float64Array(0);
  const n = nextPow2(n0);
  const re = new Float64Array(n);
  const im = new Float64Array(n);
  re.set(sig.i);
  im.set(sig.q);
  fft(re, im);
  const mag = new Float64Array(n);
  for (let k = 0; k < n; k++) mag[k] = Math.hypot(re[k], im[k]);
  return mag;
}
