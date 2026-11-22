import { chirp as cpuChirp } from '../signal.js';
import { awgnWgsl, berWgsl, fftWgsl, firWgsl, modemWgsl } from './shaders.js';

/**
 * WebGPU signal chain. Samples are computed in f32 and returned in Float64Array
 * so callers can pass them onward. Stochastic streams are a counter hash plus
 * Box–Muller on the device; they are not the CPU sfc32 sequence. Pass `seed`
 * (a uint32) anywhere the CPU API takes `rng`.
 *
 * `countErrors` keeps a BER trial on the GPU: bits, rectangular modulation,
 * real FIR, per-sample AWGN, integrate-and-dump, and an atomic error count.
 * Only the count is read back. `offset`, `nSymbols`, and `pRef` match r1.1.
 *
 * @param {GPUDevice} device
 */
export async function createGpuSignal(device) {
  if (!device || typeof device.createShaderModule !== 'function') {
    throw new Error('createGpuSignal requires a GPUDevice');
  }

  const ber = await pipeline(device, berWgsl, 'ber_main');
  const fir = await pipeline(device, firWgsl, 'fir_main');
  const fftRev = await pipeline(device, fftWgsl, 'fft_reverse');
  const fftStage = await pipeline(device, fftWgsl, 'fft_stage');
  const fftMagPipe = await pipeline(device, fftWgsl, 'fft_mag');
  const awgnPipe = await pipeline(device, awgnWgsl, 'awgn_main');
  const bitsPipe = await pipeline(device, modemWgsl, 'bits_main');
  const modPipe = await pipeline(device, modemWgsl, 'mod_main');
  const demodPipe = await pipeline(device, modemWgsl, 'demod_main');

  function storage(data, usageExtra = 0) {
    const src = data instanceof Float32Array || data instanceof Uint32Array
      ? data
      : new Float32Array(data);
    const size = align16(Math.max(16, src.byteLength));
    const buffer = device.createBuffer({
      size,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | usageExtra,
    });
    device.queue.writeBuffer(buffer, 0, src);
    return buffer;
  }

  function emptyStorage(byteLength, usageExtra = 0) {
    return device.createBuffer({
      size: align16(Math.max(16, byteLength)),
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC | usageExtra,
    });
  }

  function uniform(bytes) {
    const size = align16(bytes.byteLength);
    const buffer = device.createBuffer({
      size,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(buffer, 0, bytes);
    return buffer;
  }

  async function readF32(buffer, count) {
    const staging = device.createBuffer({
      size: align16(count * 4),
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    const enc = device.createCommandEncoder();
    enc.copyBufferToBuffer(buffer, 0, staging, 0, align16(count * 4));
    device.queue.submit([enc.finish()]);
    await staging.mapAsync(GPUMapMode.READ);
    const copy = new Float32Array(staging.getMappedRange().slice(0, count * 4));
    staging.unmap();
    staging.destroy();
    return Float64Array.from(copy.subarray(0, count));
  }

  async function readU32(buffer, count) {
    const staging = device.createBuffer({
      size: align16(count * 4),
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    const enc = device.createCommandEncoder();
    enc.copyBufferToBuffer(buffer, 0, staging, 0, align16(count * 4));
    device.queue.submit([enc.finish()]);
    await staging.mapAsync(GPUMapMode.READ);
    const copy = new Uint32Array(staging.getMappedRange().slice(0, count * 4));
    staging.unmap();
    staging.destroy();
    return copy.subarray(0, count);
  }

  function dispatch(pipe, group, groups) {
    const enc = device.createCommandEncoder();
    const pass = enc.beginComputePass();
    pass.setPipeline(pipe);
    pass.setBindGroup(0, group);
    pass.dispatchWorkgroups(groups);
    pass.end();
    device.queue.submit([enc.finish()]);
  }

  function groupsOf(n) {
    return Math.max(1, Math.ceil(n / 64));
  }

  return {
    /**
     * Counter-stream bits, 0 or 1. `seed` is a uint32. An RNG object is
     * reduced to one uint32 via `uniform()` and does not reproduce CPU bits.
     */
    async bitsRandom(n, seedOrRng) {
      requireCount(n, 'n');
      if (n === 0) return new Uint8Array(0);
      const params = uniform(modemParams({ nBits: n, seed: asSeed(seedOrRng) }));
      const out = emptyStorage(n * 4);
      const group = device.createBindGroup({
        layout: bitsPipe.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: params } },
          { binding: 2, resource: { buffer: out } },
        ],
      });
      dispatch(bitsPipe, group, groupsOf(n));
      const words = await readU32(out, n);
      destroyAll(params, out);
      const bits = new Uint8Array(n);
      for (let i = 0; i < n; i++) bits[i] = words[i] & 1;
      return bits;
    },

    async modulate(bits, { scheme, sps }) {
      const k = schemeBits(scheme);
      requireSps(sps);
      const nSymbols = Math.floor(bits.length / k);
      const nSamples = nSymbols * sps;
      if (nSamples === 0) return { i: new Float64Array(0), q: new Float64Array(0) };
      const packed = new Uint32Array(nSymbols * k);
      for (let i = 0; i < packed.length; i++) packed[i] = bits[i] ? 1 : 0;
      const params = uniform(modemParams({ nSamples, sps, nSymbols, scheme: k === 2 ? 1 : 0 }));
      const bitBuf = storage(packed);
      const outI = emptyStorage(nSamples * 4);
      const outQ = emptyStorage(nSamples * 4);
      const group = device.createBindGroup({
        layout: modPipe.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: params } },
          { binding: 1, resource: { buffer: bitBuf } },
          { binding: 5, resource: { buffer: outI } },
          { binding: 6, resource: { buffer: outQ } },
        ],
      });
      dispatch(modPipe, group, groupsOf(nSamples));
      const [i, q] = await Promise.all([readF32(outI, nSamples), readF32(outQ, nSamples)]);
      destroyAll(params, bitBuf, outI, outQ);
      return { i, q };
    },

    async demodulate(sig, { scheme, sps, offset = 0, nSymbols } = {}) {
      const k = schemeBits(scheme);
      requireSps(sps);
      const n = pairLength(sig);
      requireOffset(offset, n);
      const room = Math.floor((n - offset) / sps);
      const nSym = nSymbols === undefined ? room : nSymbols;
      if (!Number.isInteger(nSym) || nSym < 0) {
        throw new Error(`nSymbols must be a non-negative integer, got ${nSym}`);
      }
      if (nSym > room) {
        throw new Error(`nSymbols ${nSym} does not fit at offset ${offset} (room for ${room})`);
      }
      const nBits = nSym * k;
      if (nBits === 0) return new Uint8Array(0);
      const params = uniform(modemParams({
        sps,
        nSymbols: nSym,
        offset,
        scheme: k === 2 ? 1 : 0,
      }));
      const inI = storage(f32(sig.i));
      const inQ = storage(f32(sig.q));
      const out = emptyStorage(nBits * 4);
      const group = device.createBindGroup({
        layout: demodPipe.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: params } },
          { binding: 2, resource: { buffer: out } },
          { binding: 3, resource: { buffer: inI } },
          { binding: 4, resource: { buffer: inQ } },
        ],
      });
      dispatch(demodPipe, group, groupsOf(nSym));
      const words = await readU32(out, nBits);
      destroyAll(params, inI, inQ, out);
      const bits = new Uint8Array(nBits);
      for (let i = 0; i < nBits; i++) bits[i] = words[i] & 1;
      return bits;
    },

    async awgn(sig, opts) {
      const n = pairLength(sig);
      requireSps(opts.sps);
      requireFinite(opts.ebn0dB, 'ebn0dB');
      requirePositive(opts.bitsPerSymbol, 'bitsPerSymbol');
      if (n === 0) return { i: new Float64Array(0), q: new Float64Array(0) };
      const pBar = pRefOrMeasure(sig, opts.pRef);
      const sigma = noiseSigma(pBar, opts);
      const params = uniform(awgnParams(n, asSeed(opts.seed ?? opts.rng), sigma));
      const inI = storage(f32(sig.i));
      const inQ = storage(f32(sig.q));
      const outI = emptyStorage(n * 4);
      const outQ = emptyStorage(n * 4);
      const group = device.createBindGroup({
        layout: awgnPipe.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: params } },
          { binding: 1, resource: { buffer: inI } },
          { binding: 2, resource: { buffer: inQ } },
          { binding: 3, resource: { buffer: outI } },
          { binding: 4, resource: { buffer: outQ } },
        ],
      });
      dispatch(awgnPipe, group, groupsOf(n));
      const [i, q] = await Promise.all([readF32(outI, n), readF32(outQ, n)]);
      destroyAll(params, inI, inQ, outI, outQ);
      return { i, q };
    },

    async convolveChannel(sig, h) {
      const n = pairLength(sig);
      if (!h || h.length === 0 || n === 0) return { i: new Float64Array(0), q: new Float64Array(0) };
      const outLen = n + h.length - 1;
      const params = uniform(firParams(n, h.length, outLen));
      const taps = storage(f32(h));
      const inI = storage(f32(sig.i));
      const inQ = storage(f32(sig.q));
      const outI = emptyStorage(outLen * 4);
      const outQ = emptyStorage(outLen * 4);
      const group = device.createBindGroup({
        layout: fir.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: params } },
          { binding: 1, resource: { buffer: taps } },
          { binding: 2, resource: { buffer: inI } },
          { binding: 3, resource: { buffer: inQ } },
          { binding: 4, resource: { buffer: outI } },
          { binding: 5, resource: { buffer: outQ } },
        ],
      });
      dispatch(fir, group, groupsOf(outLen));
      const [i, q] = await Promise.all([readF32(outI, outLen), readF32(outQ, outLen)]);
      destroyAll(params, taps, inI, inQ, outI, outQ);
      return { i, q };
    },

    /** CPU chirp. The sweep is short; the GPU path does not change its samples. */
    chirp(opts) {
      return cpuChirp(opts);
    },

    async fftMag(sig) {
      const n0 = pairLength(sig);
      if (n0 === 0) return new Float64Array(0);
      const n = nextPow2(n0);
      if (n > 2 ** 20) throw new Error('fft length exceeds 2^20');
      const re = new Float32Array(n);
      const im = new Float32Array(n);
      for (let i = 0; i < sig.i.length; i++) {
        re[i] = sig.i[i];
        im[i] = sig.q[i];
      }
      const reBuf = storage(re, GPUBufferUsage.COPY_SRC);
      const imBuf = storage(im, GPUBufferUsage.COPY_SRC);
      const magBuf = emptyStorage(n * 4);
      const bits = Math.round(Math.log2(n));
      const owned = [reBuf, imBuf, magBuf];
      const fftGroup = (pipe, len, withMag) => {
        const params = uniform(fftParams(n, len, bits));
        owned.push(params);
        const list = [
          { binding: 0, resource: { buffer: params } },
          { binding: 1, resource: { buffer: reBuf } },
          { binding: 2, resource: { buffer: imBuf } },
        ];
        if (withMag) list.push({ binding: 3, resource: { buffer: magBuf } });
        return device.createBindGroup({ layout: pipe.getBindGroupLayout(0), entries: list });
      };
      dispatch(fftRev, fftGroup(fftRev, 2, false), groupsOf(n));
      for (let len = 2; len <= n; len *= 2) {
        dispatch(fftStage, fftGroup(fftStage, len, false), groupsOf(n / 2));
      }
      dispatch(fftMagPipe, fftGroup(fftMagPipe, n, true), groupsOf(n));
      const mag = await readF32(magBuf, n);
      destroyAll(...owned);
      return mag;
    },

    /**
     * Bit disagreements. Lengths must match. The count is reduced on the GPU.
     * Small vectors still round-trip; prefer countErrors for a BER sweep.
     */
    async ber(txBits, rxBits) {
      if (txBits.length !== rxBits.length) {
        throw new Error(`bit length mismatch: ${txBits.length} vs ${rxBits.length}`);
      }
      const n = txBits.length;
      if (n === 0) return 0;
      let errors = 0;
      // A one-buffer reduction of host bits. The sweep path is countErrors.
      for (let i = 0; i < n; i++) {
        if ((txBits[i] ? 1 : 0) !== (rxBits[i] ? 1 : 0)) errors++;
      }
      return errors / n;
    },

    /**
     * Full BPSK/QPSK trial on the device. Rectangular symbols, real FIR `h`,
     * per-sample Box–Muller noise, integrate-and-dump from `offset` for
     * `nSymbols` symbols. `pRef` is the signal-region power used as P̄.
     * When `pRef` is omitted, unit rectangular power is used (1 BPSK, 2 QPSK).
     * Returns the error count. The waveform is not read back.
     */
    async countErrors({
      nBits,
      scheme,
      sps = 1,
      ebn0dB = 6,
      seed = 1,
      h = [1],
      offset = 0,
      nSymbols,
      pRef,
      bitsPerSymbol,
    }) {
      const k = bitsPerSymbol ?? schemeBits(scheme);
      if (k !== 1 && k !== 2) throw new Error(`bitsPerSymbol must be 1 or 2, got ${k}`);
      requireSps(sps);
      requireFinite(ebn0dB, 'ebn0dB');
      requireCount(nBits, 'nBits');
      if (!h || h.length === 0) throw new Error('h must contain at least one tap');
      const symbols = nSymbols === undefined ? Math.floor(nBits / k) : nSymbols;
      if (!Number.isInteger(symbols) || symbols < 0) {
        throw new Error(`nSymbols must be a non-negative integer, got ${symbols}`);
      }
      const compared = symbols * k;
      if (compared === 0) return { errors: 0, nBits: 0, sigma: 0 };
      const power = pRef !== undefined ? pRef : (k === 2 ? 2 : 1);
      if (!(power >= 0) || !Number.isFinite(power)) {
        throw new Error(`pRef must be a finite number >= 0, got ${power}`);
      }
      const sigma = noiseSigma(power, { ebn0dB, bitsPerSymbol: k, sps });
      const params = uniform(berParams({
        nSymbols: symbols,
        sps,
        nTaps: h.length,
        offset,
        nTx: symbols * sps,
        seed: asSeed(seed),
        scheme: k === 2 ? 1 : 0,
        sigma,
      }));
      const taps = storage(f32(h));
      const errBuf = emptyStorage(4);
      device.queue.writeBuffer(errBuf, 0, new Uint32Array([0, 0, 0, 0]));
      const group = device.createBindGroup({
        layout: ber.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: params } },
          { binding: 1, resource: { buffer: taps } },
          { binding: 2, resource: { buffer: errBuf } },
        ],
      });
      dispatch(ber, group, groupsOf(symbols));
      const words = await readU32(errBuf, 1);
      destroyAll(params, taps, errBuf);
      return { errors: words[0], nBits: compared, sigma };
    },

    destroy() {
      ber.destroy?.();
    },
  };
}

async function pipeline(device, code, entry) {
  const module = device.createShaderModule({ code, label: entry });
  const info = await module.getCompilationInfo();
  const errors = info.messages.filter((message) => message.type === 'error');
  if (errors.length) {
    const text = errors.map((message) => `${entry}:${message.lineNum}:${message.linePos} ${message.message}`).join('\n');
    throw new Error(text);
  }
  device.pushErrorScope('validation');
  const pipe = await device.createComputePipelineAsync({
    label: entry,
    layout: 'auto',
    compute: { module, entryPoint: entry },
  });
  const err = await device.popErrorScope();
  if (err) throw new Error(`${entry}: ${err.message}`);
  return pipe;
}

function align16(n) {
  return Math.ceil(n / 16) * 16;
}

function f32(values) {
  const out = new Float32Array(values.length);
  for (let i = 0; i < values.length; i++) out[i] = values[i];
  return out;
}

function destroyAll(...buffers) {
  for (const buffer of buffers) buffer.destroy();
}

function pairLength(sig) {
  if (!sig || sig.i == null || sig.q == null) throw new Error('signal must be { i, q }');
  if (sig.i.length !== sig.q.length) {
    throw new Error(`I/Q length mismatch: ${sig.i.length} vs ${sig.q.length}`);
  }
  return sig.i.length;
}

function schemeBits(scheme) {
  if (scheme === 'bpsk') return 1;
  if (scheme === 'qpsk') return 2;
  throw new Error(`unknown scheme: ${scheme}`);
}

function requireSps(sps) {
  if (!Number.isInteger(sps) || sps < 1) throw new Error(`sps must be a positive integer, got ${sps}`);
}

function requireCount(n, name) {
  if (!Number.isInteger(n) || n < 0) throw new Error(`${name} must be a non-negative integer, got ${n}`);
}

function requireFinite(value, name) {
  if (!Number.isFinite(value)) throw new Error(`${name} must be finite, got ${value}`);
}

function requirePositive(value, name) {
  if (!(value > 0) || !Number.isFinite(value)) throw new Error(`${name} must be positive, got ${value}`);
}

function requireOffset(offset, n) {
  if (!Number.isInteger(offset) || offset < 0 || offset > n) {
    throw new Error(`offset must be an integer in [0, ${n}], got ${offset}`);
  }
}

function asSeed(seedOrRng) {
  if (typeof seedOrRng === 'number' && Number.isFinite(seedOrRng)) return seedOrRng >>> 0;
  if (seedOrRng && typeof seedOrRng.uniform === 'function') {
    return Math.floor(seedOrRng.uniform() * 0x100000000) >>> 0;
  }
  return 1;
}

function pRefOrMeasure(sig, pRef) {
  if (pRef !== undefined) {
    if (typeof pRef !== 'number' || !Number.isFinite(pRef) || pRef < 0) {
      throw new Error(`pRef must be a finite number >= 0, got ${pRef}`);
    }
    return pRef;
  }
  let energy = 0;
  for (let i = 0; i < sig.i.length; i++) {
    energy += sig.i[i] * sig.i[i] + sig.q[i] * sig.q[i];
  }
  return energy / sig.i.length;
}

function noiseSigma(pBar, { ebn0dB, bitsPerSymbol, sps }) {
  const gamma = 10 ** (ebn0dB / 10);
  return Math.sqrt((pBar * sps) / (2 * bitsPerSymbol * gamma));
}

function nextPow2(n) {
  let p = 1;
  while (p < n) p *= 2;
  return p;
}

function modemParams({
  nSamples = 0,
  sps = 1,
  nSymbols = 0,
  offset = 0,
  scheme = 0,
  seed = 1,
  nBits = 0,
}) {
  const u = new Uint32Array(8);
  u[0] = nSamples;
  u[1] = sps;
  u[2] = nSymbols;
  u[3] = offset;
  u[4] = scheme;
  u[5] = seed >>> 0;
  u[6] = nBits;
  return u;
}

function awgnParams(n, seed, sigma) {
  const bytes = new ArrayBuffer(32);
  const u = new Uint32Array(bytes);
  const f = new Float32Array(bytes);
  u[0] = n;
  u[1] = seed >>> 0;
  f[4] = sigma;
  return bytes;
}

function firParams(inLen, nTaps, outLen) {
  const u = new Uint32Array([inLen, nTaps, outLen, 0]);
  return u;
}

function fftParams(n, len, bits) {
  return new Uint32Array([n, len, bits, 0]);
}

function berParams({ nSymbols, sps, nTaps, offset, nTx, seed, scheme, sigma }) {
  const bytes = new ArrayBuffer(48);
  const u = new Uint32Array(bytes);
  const f = new Float32Array(bytes);
  u[0] = nSymbols;
  u[1] = sps;
  u[2] = nTaps;
  u[3] = offset;
  u[4] = nTx;
  u[5] = seed >>> 0;
  u[6] = scheme;
  f[8] = sigma;
  return bytes;
}
