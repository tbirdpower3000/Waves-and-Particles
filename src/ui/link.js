// Channel path for contract r1.1.
// demodulate(rx, { scheme, sps, offset, nSymbols }) with offset = main-tap
// index and nSymbols = transmitted symbols. awgn(..., { pRef }) uses the
// mean power of that symbol region, not the zero-padded buffer.

function tapParts(value) {
  if (value != null && typeof value === 'object') {
    const re = Number(value.re ?? value.i ?? 0);
    const im = Number(value.im ?? value.q ?? 0);
    return {
      re: Number.isFinite(re) ? re : 0,
      im: Number.isFinite(im) ? im : 0,
    };
  }
  const re = Number(value);
  return { re: Number.isFinite(re) ? re : 0, im: 0 };
}

// Largest tap by magnitude. The returned phasor keeps the tap's sign and,
// for a complex tap, its phase, so a negative cursor does not invert the bits.
export function mainCursor(h) {
  if (!h || h.length === 0) return { index: 0, re: 1, im: 0 };
  let best = 0;
  let mag = 0;
  let bestRe = 0;
  let bestIm = 0;
  for (let i = 0; i < h.length; i++) {
    const part = tapParts(h[i]);
    const size = Math.hypot(part.re, part.im);
    if (size > mag) {
      mag = size;
      best = i;
      bestRe = part.re;
      bestIm = part.im;
    }
  }
  if (!(mag > 0)) return { index: best, re: 1, im: 0 };
  return { index: best, re: bestRe / mag, im: bestIm / mag };
}

export function mainTap(h) {
  return mainCursor(h).index;
}

// Multiply by the conjugate of the main-tap phasor. A real negative tap is a
// phase of pi, which flips BPSK and both QPSK rails before the decision.
export function alignToCursor(sig, cursor) {
  const re = cursor?.re ?? 1;
  const im = cursor?.im ?? 0;
  const n = sig.i.length;
  const i = new Float64Array(n);
  const q = new Float64Array(n);
  for (let k = 0; k < n; k++) {
    const x = sig.i[k];
    const y = sig.q[k];
    i[k] = re * x + im * y;
    q[k] = re * y - im * x;
  }
  return { i, q };
}

export function regionPower(sig, offset, len) {
  if (!sig || sig.i == null || sig.q == null) throw new Error('signal must be { i, q }');
  if (!Number.isInteger(offset) || offset < 0) {
    throw new Error(`offset must be an integer >= 0, got ${offset}`);
  }
  if (!Number.isInteger(len) || len < 1) {
    throw new Error(`signal region length must be a positive integer, got ${len}`);
  }
  if (offset + len > sig.i.length || offset + len > sig.q.length) {
    throw new Error(`signal region [${offset}, ${offset + len}) does not fit in ${sig.i.length} samples`);
  }
  let energy = 0;
  for (let i = 0; i < len; i++) {
    const iv = sig.i[offset + i];
    const qv = sig.q[offset + i];
    if (!Number.isFinite(iv) || !Number.isFinite(qv)) {
      throw new Error('non-finite sample in the signal region');
    }
    energy += iv * iv + qv * qv;
  }
  return energy / len;
}

export function symbolPoints(i, q, sps, offset, nSymbols, limit = 800) {
  const points = [];
  const count = Math.min(nSymbols, limit);
  for (let s = 0; s < count; s++) {
    const off = offset + s * sps;
    if (off + sps > i.length || off + sps > q.length) break;
    let si = 0;
    let sq = 0;
    for (let t = 0; t < sps; t++) {
      si += i[off + t];
      sq += q[off + t];
    }
    points.push({ i: si / sps, q: sq / sps });
  }
  return points;
}

export function trajectory(i, q, offset, span, limit = 1500) {
  const end = Math.min(i.length, q.length, offset + span);
  const start = Math.max(0, Math.min(offset, end));
  const n = end - start;
  if (n <= 0) return [];
  const step = Math.max(1, Math.ceil(n / limit));
  const points = [];
  for (let at = start; at < end; at += step) {
    points.push({ i: i[at], q: q[at] });
  }
  return points;
}

export function spectrumDb(mag) {
  const n = mag?.length ?? 0;
  const half = Math.max(0, Math.ceil(n / 2));
  const out = new Float64Array(half);
  if (half === 0) return out;
  let peak = 0;
  for (let k = 0; k < half; k++) {
    const m = Number(mag[k]);
    if (Number.isFinite(m) && m > peak) peak = m;
  }
  const norm = peak > 0 ? peak : 1;
  for (let k = 0; k < half; k++) {
    const m = Number(mag[k]);
    out[k] = 20 * Math.log10((Number.isFinite(m) ? m : 0) / norm + 1e-12);
  }
  return out;
}

export function formatBer(rate, bits) {
  if (!Number.isFinite(rate)) return '—';
  if (rate === 0) {
    const n = Number.isFinite(bits) && bits > 0 ? Math.trunc(bits) : 0;
    return n > 0 ? `< 1/${n}` : '< 1/N';
  }
  return rate.toExponential(2);
}

function tapsOf(h) {
  return h && h.length ? h : new Float64Array([1]);
}

export function runChannel(api, { bits, scheme, sps, h, ebn0dB, rng }) {
  if (scheme !== 'bpsk' && scheme !== 'qpsk') {
    throw new Error(`demodulate is only defined for BPSK and QPSK, got ${scheme}`);
  }
  if (!Number.isInteger(sps) || sps < 1) throw new Error(`sps must be a positive integer, got ${sps}`);
  const taps = tapsOf(h);
  const k = scheme === 'qpsk' ? 2 : 1;
  const nSymbols = Math.floor(bits.length / k);
  if (nSymbols < 1) throw new Error('not enough bits for one symbol');
  const txBits = bits.subarray(0, nSymbols * k);
  const modulated = api.modulate(txBits, { scheme, sps });
  const clean = api.convolveChannel(modulated, taps);
  const cursor = mainCursor(taps);
  const offset = cursor.index;
  const span = nSymbols * sps;
  const pRef = regionPower(clean, offset, span);
  const noisy = api.awgn(clean, {
    ebn0dB,
    bitsPerSymbol: k,
    sps,
    rng,
    pRef,
  });
  const aligned = alignToCursor(noisy, cursor);
  const rxBits = api.demodulate(aligned, {
    scheme,
    sps,
    offset,
    nSymbols,
  });
  const errorRate = api.ber(txBits, rxBits);
  return {
    clean, noisy: aligned, txBits, rxBits, offset, nSymbols, pRef, errorRate, h: taps, span, cursor,
  };
}

export function runChirp(api, { h, ebn0dB, rng, f0, f1, T, fs }) {
  const taps = tapsOf(h);
  const tone = api.chirp({ f0, f1, T, fs });
  const clean = api.convolveChannel(tone, taps);
  const offset = mainCursor(taps).index;
  const span = tone.i.length;
  const pRef = regionPower(clean, offset, span);
  const noisy = api.awgn(clean, {
    ebn0dB,
    bitsPerSymbol: 1,
    sps: 1,
    rng,
    pRef,
  });
  return { noisy, offset, nSymbols: 0, pRef, span, h: taps };
}

export function runLinkSafely(api, opts) {
  try {
    if (opts.scheme === 'chirp') {
      return { ok: true, chirp: true, result: runChirp(api, opts) };
    }
    return { ok: true, chirp: false, result: runChannel(api, opts) };
  } catch (err) {
    return { ok: false, message: err instanceof Error ? err.message : String(err) };
  }
}
