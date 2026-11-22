// WGSL compute for the signal chain. f32. Counter hash + Box–Muller, not the CPU sfc32 stream.

export const berWgsl = /* wgsl */ `
struct BerParams {
  nSymbols: u32,
  sps: u32,
  nTaps: u32,
  offset: u32,
  nTx: u32,
  seed: u32,
  scheme: u32,
  pad0: u32,
  sigma: f32,
  pad1: f32,
  pad2: f32,
  pad3: f32,
}

@group(0) @binding(0) var<uniform> p: BerParams;
@group(0) @binding(1) var<storage, read> h: array<f32>;
@group(0) @binding(2) var<storage, read_write> errCount: array<atomic<u32>>;

var<workgroup> red: array<u32, 64>;

fn mix(x: u32) -> u32 {
  var z = (x ^ (x >> 16u)) * 0x7feb352du;
  z = (z ^ (z >> 15u)) * 0x846ca68bu;
  return z ^ (z >> 16u);
}

fn u01(n: u32) -> f32 {
  let bits = mix(n);
  return (f32(bits >> 8u) + 0.5) / 16777216.0;
}

fn gauss(idx: u32) -> f32 {
  let u1 = u01(idx * 2u);
  let u2 = u01(idx * 2u + 1u);
  let mag = sqrt(-2.0 * log(u1));
  return mag * cos(6.283185307179586 * u2);
}

fn bitOf(sym: u32, rail: u32) -> u32 {
  return mix(sym + p.seed * 0x9e3779b9u + rail * 0x85ebca6bu) & 1u;
}

fn tx(sample: i32, rail: u32) -> f32 {
  if (p.scheme == 0u && rail == 1u) { return 0.0; }
  if (sample < 0 || u32(sample) >= p.nTx) { return 0.0; }
  let sym = u32(sample) / p.sps;
  let bit = bitOf(sym, rail);
  return select(-1.0, 1.0, bit == 0u);
}

fn filtered(sample: i32, rail: u32) -> f32 {
  var acc = 0.0;
  for (var k = 0u; k < p.nTaps; k++) {
    acc += h[k] * tx(sample - i32(k), rail);
  }
  return acc;
}

fn errorsAt(sym: u32) -> u32 {
  var accI = 0.0;
  var accQ = 0.0;
  for (var t = 0u; t < p.sps; t++) {
    let n = i32(p.offset + sym * p.sps + t);
    let base = u32(max(n, 0)) * 2u;
    accI += filtered(n, 0u) + p.sigma * gauss(base ^ p.seed);
    accQ += filtered(n, 1u) + p.sigma * gauss((base + 1u) ^ p.seed);
  }
  var errs = 0u;
  let bitI = bitOf(sym, 0u);
  let decidedI = select(1u, 0u, accI >= 0.0);
  if (decidedI != bitI) { errs += 1u; }
  if (p.scheme == 1u) {
    let bitQ = bitOf(sym, 1u);
    let decidedQ = select(1u, 0u, accQ >= 0.0);
    if (decidedQ != bitQ) { errs += 1u; }
  }
  return errs;
}

@compute @workgroup_size(64)
fn ber_main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
  var localErr = 0u;
  if (gid.x < p.nSymbols) {
    localErr = errorsAt(gid.x);
  }
  red[lid] = localErr;
  workgroupBarrier();
  var stride = 32u;
  loop {
    if (stride == 0u) { break; }
    if (lid < stride) {
      red[lid] += red[lid + stride];
    }
    workgroupBarrier();
    stride = stride >> 1u;
  }
  if (lid == 0u) {
    atomicAdd(&errCount[0], red[0]);
  }
}
`;

export const firWgsl = /* wgsl */ `
struct FirParams {
  inLen: u32,
  nTaps: u32,
  outLen: u32,
  pad: u32,
}

@group(0) @binding(0) var<uniform> p: FirParams;
@group(0) @binding(1) var<storage, read> h: array<f32>;
@group(0) @binding(2) var<storage, read> inI: array<f32>;
@group(0) @binding(3) var<storage, read> inQ: array<f32>;
@group(0) @binding(4) var<storage, read_write> outI: array<f32>;
@group(0) @binding(5) var<storage, read_write> outQ: array<f32>;

@compute @workgroup_size(64)
fn fir_main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let n = gid.x;
  if (n >= p.outLen) { return; }
  var accI = 0.0;
  var accQ = 0.0;
  for (var k = 0u; k < p.nTaps; k++) {
    let src = i32(n) - i32(k);
    if (src >= 0 && u32(src) < p.inLen) {
      accI += h[k] * inI[u32(src)];
      accQ += h[k] * inQ[u32(src)];
    }
  }
  outI[n] = accI;
  outQ[n] = accQ;
}
`;

export const fftWgsl = /* wgsl */ `
struct FftParams {
  n: u32,
  len: u32,
  bits: u32,
  pad: u32,
}

@group(0) @binding(0) var<uniform> p: FftParams;
@group(0) @binding(1) var<storage, read_write> re: array<f32>;
@group(0) @binding(2) var<storage, read_write> im: array<f32>;
@group(0) @binding(3) var<storage, read_write> mag: array<f32>;

fn reverseBits(x: u32) -> u32 {
  var v = x;
  var r = 0u;
  for (var i = 0u; i < p.bits; i++) {
    r = (r << 1u) | (v & 1u);
    v = v >> 1u;
  }
  return r;
}

@compute @workgroup_size(64)
fn fft_reverse(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= p.n) { return; }
  let j = reverseBits(i);
  if (i < j) {
    let tr = re[i];
    let ti = im[i];
    re[i] = re[j];
    im[i] = im[j];
    re[j] = tr;
    im[j] = ti;
  }
}

@compute @workgroup_size(64)
fn fft_stage(@builtin(global_invocation_id) gid: vec3<u32>) {
  let half = p.len >> 1u;
  let nButterfly = p.n >> 1u;
  let id = gid.x;
  if (id >= nButterfly) { return; }
  let group = id / half;
  let k = id - group * half;
  let i0 = group * p.len + k;
  let i1 = i0 + half;
  let ang = -6.283185307179586 * f32(k) / f32(p.len);
  let wRe = cos(ang);
  let wIm = sin(ang);
  let uRe = re[i0];
  let uIm = im[i0];
  let vRe = re[i1] * wRe - im[i1] * wIm;
  let vIm = re[i1] * wIm + im[i1] * wRe;
  re[i0] = uRe + vRe;
  im[i0] = uIm + vIm;
  re[i1] = uRe - vRe;
  im[i1] = uIm - vIm;
}

@compute @workgroup_size(64)
fn fft_mag(@builtin(global_invocation_id) gid: vec3<u32>) {
  let k = gid.x;
  if (k >= p.n) { return; }
  mag[k] = sqrt(re[k] * re[k] + im[k] * im[k]);
}
`;

export const awgnWgsl = /* wgsl */ `
struct AwgnParams {
  n: u32,
  seed: u32,
  pad0: u32,
  pad1: u32,
  sigma: f32,
  pad2: f32,
  pad3: f32,
  pad4: f32,
}

@group(0) @binding(0) var<uniform> p: AwgnParams;
@group(0) @binding(1) var<storage, read> inI: array<f32>;
@group(0) @binding(2) var<storage, read> inQ: array<f32>;
@group(0) @binding(3) var<storage, read_write> outI: array<f32>;
@group(0) @binding(4) var<storage, read_write> outQ: array<f32>;

fn mix(x: u32) -> u32 {
  var z = (x ^ (x >> 16u)) * 0x7feb352du;
  z = (z ^ (z >> 15u)) * 0x846ca68bu;
  return z ^ (z >> 16u);
}

fn u01(n: u32) -> f32 {
  let bits = mix(n);
  return (f32(bits >> 8u) + 0.5) / 16777216.0;
}

fn gauss(idx: u32) -> f32 {
  let u1 = u01(idx * 2u);
  let u2 = u01(idx * 2u + 1u);
  let mag = sqrt(-2.0 * log(u1));
  return mag * cos(6.283185307179586 * u2);
}

@compute @workgroup_size(64)
fn awgn_main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= p.n) { return; }
  let base = i * 2u;
  outI[i] = inI[i] + p.sigma * gauss(base ^ p.seed);
  outQ[i] = inQ[i] + p.sigma * gauss((base + 1u) ^ p.seed);
}
`;

export const modemWgsl = /* wgsl */ `
struct ModemParams {
  nSamples: u32,
  sps: u32,
  nSymbols: u32,
  offset: u32,
  scheme: u32,
  seed: u32,
  nBits: u32,
  pad: u32,
}

@group(0) @binding(0) var<uniform> p: ModemParams;
@group(0) @binding(1) var<storage, read> bitsIn: array<u32>;
@group(0) @binding(2) var<storage, read_write> bitsOut: array<u32>;
@group(0) @binding(3) var<storage, read> inI: array<f32>;
@group(0) @binding(4) var<storage, read> inQ: array<f32>;
@group(0) @binding(5) var<storage, read_write> outI: array<f32>;
@group(0) @binding(6) var<storage, read_write> outQ: array<f32>;

fn mix(x: u32) -> u32 {
  var z = (x ^ (x >> 16u)) * 0x7feb352du;
  z = (z ^ (z >> 15u)) * 0x846ca68bu;
  return z ^ (z >> 16u);
}

@compute @workgroup_size(64)
fn bits_main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= p.nBits) { return; }
  bitsOut[i] = mix(i + p.seed * 0x9e3779b9u) & 1u;
}

@compute @workgroup_size(64)
fn mod_main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let sample = gid.x;
  if (sample >= p.nSamples) { return; }
  let sym = sample / p.sps;
  if (p.scheme == 0u) {
    let bit = bitsIn[sym];
    outI[sample] = select(-1.0, 1.0, bit == 0u);
    outQ[sample] = 0.0;
  } else {
    let bitI = bitsIn[sym * 2u];
    let bitQ = bitsIn[sym * 2u + 1u];
    outI[sample] = select(-1.0, 1.0, bitI == 0u);
    outQ[sample] = select(-1.0, 1.0, bitQ == 0u);
  }
}

@compute @workgroup_size(64)
fn demod_main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let sym = gid.x;
  if (sym >= p.nSymbols) { return; }
  var accI = 0.0;
  var accQ = 0.0;
  for (var t = 0u; t < p.sps; t++) {
    let at = p.offset + sym * p.sps + t;
    accI += inI[at];
    accQ += inQ[at];
  }
  if (p.scheme == 0u) {
    bitsOut[sym] = select(1u, 0u, accI >= 0.0);
  } else {
    bitsOut[sym * 2u] = select(1u, 0u, accI >= 0.0);
    bitsOut[sym * 2u + 1u] = select(1u, 0u, accQ >= 0.0);
  }
}
`;
