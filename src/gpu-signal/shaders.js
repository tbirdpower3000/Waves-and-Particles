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

fn gauss_at(n: u32, stream: u32) -> f32 {
  // splitmix-style mix of the index, the seed, and the stream. Adding the
  // seed into the index left neighboring samples correlated.
  var z = mix(n);
  z = mix(z ^ mix(p.seed));
  z = mix(z ^ mix(stream * 0x9e3779b9u + 0x85ebca6bu));
  let u1 = u01(z);
  let u2 = u01(z ^ 0x6c62272eu);
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
    accI += filtered(n, 0u) + p.sigma * gauss_at(u32(max(n, 0)), 0u);
    accQ += filtered(n, 1u) + p.sigma * gauss_at(u32(max(n, 0)), 1u);
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

var<workgroup> energyRed: array<f32, 64>;

@group(0) @binding(3) var<storage, read_write> partialEnergy: array<f32>;

@compute @workgroup_size(64)
fn power_main(
  @builtin(global_invocation_id) gid: vec3<u32>,
  @builtin(local_invocation_index) lid: u32,
  @builtin(workgroup_id) wid: vec3<u32>,
) {
  var energy = 0.0;
  if (gid.x < p.nSymbols) {
    for (var t = 0u; t < p.sps; t++) {
      let n = i32(p.offset + gid.x * p.sps + t);
      let yi = filtered(n, 0u);
      let yq = filtered(n, 1u);
      energy += yi * yi + yq * yq;
    }
  }
  energyRed[lid] = energy;
  workgroupBarrier();
  var stride = 32u;
  loop {
    if (stride == 0u) { break; }
    if (lid < stride) {
      energyRed[lid] += energyRed[lid + stride];
    }
    workgroupBarrier();
    stride = stride >> 1u;
  }
  if (lid == 0u) {
    partialEnergy[wid.x] = energyRed[0];
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
@group(0) @binding(4) var<storage, read> twRe: array<f32>;
@group(0) @binding(5) var<storage, read> twIm: array<f32>;

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

// Host twiddles are exp(-j 2 pi k / N) in f64, stored as f32.
// After bit reversal, each radix-2 stage is one workgroup dispatch.
// A tile of 64 does the stages of length 2..64 in shared memory.
// Longer stages run one butterfly per thread. log2(N) stages in total.
var<workgroup> tileRe: array<f32, 64>;
var<workgroup> tileIm: array<f32, 64>;

fn butterfly_tile(i0: u32, i1: u32, tw: u32) {
  let wr = twRe[tw];
  let wi = twIm[tw];
  let oddRe = tileRe[i1] * wr - tileIm[i1] * wi;
  let oddIm = tileRe[i1] * wi + tileIm[i1] * wr;
  let ar = tileRe[i0];
  let ai = tileIm[i0];
  tileRe[i0] = ar + oddRe;
  tileIm[i0] = ai + oddIm;
  tileRe[i1] = ar - oddRe;
  tileIm[i1] = ai - oddIm;
}

fn butterfly_global(i0: u32, i1: u32, tw: u32) {
  let wr = twRe[tw];
  let wi = twIm[tw];
  let oddRe = re[i1] * wr - im[i1] * wi;
  let oddIm = re[i1] * wi + im[i1] * wr;
  let ar = re[i0];
  let ai = im[i0];
  re[i0] = ar + oddRe;
  im[i0] = ai + oddIm;
  re[i1] = ar - oddRe;
  im[i1] = ai - oddIm;
}

@compute @workgroup_size(64)
fn fft_tile(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
  let base = wid.x * 64u;
  if (base >= p.n) { return; }
  tileRe[lid] = re[base + lid];
  tileIm[lid] = im[base + lid];
  workgroupBarrier();
  for (var len = 2u; len <= 64u; len = len << 1u) {
    let half = len >> 1u;
    let stride = p.n / len;
    if (lid < 32u) {
      let group = lid / half;
      let k = lid - group * half;
      let i0 = group * len + k;
      butterfly_tile(i0, i0 + half, k * stride);
    }
    workgroupBarrier();
  }
  re[base + lid] = tileRe[lid];
  im[base + lid] = tileIm[lid];
}

@compute @workgroup_size(64)
fn fft_stage(@builtin(global_invocation_id) gid: vec3<u32>) {
  let half = p.len >> 1u;
  let butterflyId = gid.x;
  if (butterflyId >= (p.n >> 1u)) { return; }
  let group = butterflyId / half;
  let k = butterflyId - group * half;
  let i0 = group * p.len + k;
  butterfly_global(i0, i0 + half, k * (p.n / p.len));
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

fn gauss_at(n: u32, stream: u32) -> f32 {
  // splitmix-style mix of the index, the seed, and the stream. Adding the
  // seed into the index left neighboring samples correlated.
  var z = mix(n);
  z = mix(z ^ mix(p.seed));
  z = mix(z ^ mix(stream * 0x9e3779b9u + 0x85ebca6bu));
  let u1 = u01(z);
  let u2 = u01(z ^ 0x6c62272eu);
  let mag = sqrt(-2.0 * log(u1));
  return mag * cos(6.283185307179586 * u2);
}

@compute @workgroup_size(64)
fn awgn_main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= p.n) { return; }
  outI[i] = inI[i] + p.sigma * gauss_at(i, 0u);
  outQ[i] = inQ[i] + p.sigma * gauss_at(i, 1u);
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
