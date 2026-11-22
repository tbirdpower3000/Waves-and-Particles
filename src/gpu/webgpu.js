import { accumulateSource } from "../fdtd.js";

// One uniform plus at most seven storage buffers. Headless Chrome's default
// limit is eight storage buffers per stage, so PML profiles share one array
// and the four auxiliary ψ fields share a vec4.

const WGSL = /* wgsl */ `
struct Params {
  nx: u32,
  ny: u32,
  periodic: u32,
  span: u32,
  iInt0: i32,
  iInt1: i32,
  jInt0: i32,
  jInt1: i32,
  hx0: i32,
  hx1: i32,
  hy0: i32,
  hy1: i32,
  invDx: f32,
  invDy: f32,
  ch: f32,
  pad: f32,
}

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read_write> field: array<f32>;
@group(0) @binding(2) var<storage, read_write> hx: array<f32>;
@group(0) @binding(3) var<storage, read_write> hy: array<f32>;
@group(0) @binding(4) var<storage, read_write> psi: array<vec4f>;
@group(0) @binding(5) var<storage, read> mat: array<vec4f>;
@group(0) @binding(6) var<storage, read> src: array<f32>;
@group(0) @binding(7) var<storage, read> pml: array<f32>;

fn id(i: u32, j: u32) -> u32 { return j * p.nx + i; }
fn pmlAt(slot: u32, index: u32) -> f32 { return pml[slot * p.span + index]; }

@compute @workgroup_size(8, 8)
fn hxMain(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  let j = gid.y;
  if (i >= p.nx || j >= p.ny) { return; }
  if (p.periodic == 0u && j + 1u >= p.ny) { return; }
  var up = j + 1u;
  if (up >= p.ny) { up = 0u; }
  let here = id(i, j);
  let d = (field[id(i, up)] - field[here]) * p.invDy;
  let pmlRow = p.periodic == 0u && (i32(j) < p.hx0 || i32(j) >= p.hx1);
  if (!pmlRow) {
    hx[here] = hx[here] - p.ch * d;
    return;
  }
  var aux = psi[here];
  let next = pmlAt(7u, j) * aux.x + pmlAt(6u, j) * d;
  aux.x = next;
  psi[here] = aux;
  hx[here] = hx[here] - p.ch * (d + next);
}

@compute @workgroup_size(8, 8)
fn hyMain(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  let j = gid.y;
  if (i + 1u >= p.nx || j >= p.ny) { return; }
  let here = id(i, j);
  let d = (field[id(i + 1u, j)] - field[here]) * p.invDx;
  let pmlCol = i32(i) < p.hy0 || i32(i) >= p.hy1;
  if (!pmlCol) {
    hy[here] = hy[here] + p.ch * d;
    return;
  }
  var aux = psi[here];
  let next = pmlAt(5u, i) * aux.y + pmlAt(4u, i) * d;
  aux.y = next;
  psi[here] = aux;
  hy[here] = hy[here] + p.ch * (d + next);
}

@compute @workgroup_size(8, 8)
fn ezMain(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  let j = gid.y;
  if (i >= p.nx || j >= p.ny) { return; }
  let here = id(i, j);
  let m = mat[here];
  if (m.z > 0.5) {
    field[here] = 0.0;
    return;
  }
  var e = field[here];
  let interiorX = i > 0u && i + 1u < p.nx;
  let interiorY = p.periodic == 1u || (j > 0u && j + 1u < p.ny);
  if (interiorX && interiorY) {
    let below = select(j - 1u, p.ny - 1u, j == 0u);
    let dHy = (hy[here] - hy[id(i - 1u, j)]) * p.invDx;
    let dHx = (hx[here] - hx[id(i, below)]) * p.invDy;
    let edge = i32(i) < p.iInt0 || i32(i) > p.iInt1
      || (p.periodic == 0u && (i32(j) < p.jInt0 || i32(j) > p.jInt1));
    if (!edge) {
      e = m.x * e + m.y * (dHy - dHx);
    } else {
      var aux = psi[here];
      let px = pmlAt(1u, i) * aux.z + pmlAt(0u, i) * dHy;
      let py = pmlAt(3u, j) * aux.w + pmlAt(2u, j) * dHx;
      aux.z = px;
      aux.w = py;
      psi[here] = aux;
      e = m.x * e + m.y * (dHy + px - (dHx + py));
    }
  }
  field[here] = e + src[here];
}

struct ProbeU {
  nProbes: u32,
  stepIndex: u32,
  stride: u32,
  pad: u32,
}
@group(0) @binding(8) var<uniform> probeU: ProbeU;
@group(0) @binding(9) var<storage, read> probeIndex: array<u32>;
@group(0) @binding(10) var<storage, read_write> hist: array<f32>;

@compute @workgroup_size(64)
fn probeMain(@builtin(global_invocation_id) gid: vec3u) {
  let p = gid.x;
  if (p >= probeU.nProbes) { return; }
  hist[probeU.stepIndex * probeU.stride + p] = field[probeIndex[p]];
}
`;

function f32(src) {
  const out = new Float32Array(src.length);
  out.set(src);
  return out;
}

function storage(device, array) {
  const bytes = array.byteLength;
  const size = Math.max(16, bytes + ((16 - (bytes % 16)) % 16));
  const buffer = device.createBuffer({
    size,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    mappedAtCreation: true,
  });
  new Uint8Array(buffer.getMappedRange()).set(
    new Uint8Array(array.buffer, array.byteOffset, array.byteLength),
  );
  buffer.unmap();
  return buffer;
}

function packPsi(grid) {
  const n = grid.nx * grid.ny;
  const out = new Float32Array(n * 4);
  for (let i = 0; i < n; i++) {
    out[i * 4] = grid.psiHxy[i];
    out[i * 4 + 1] = grid.psiHyx[i];
    out[i * 4 + 2] = grid.psiEzx[i];
    out[i * 4 + 3] = grid.psiEzy[i];
  }
  return out;
}

function packMat(grid) {
  const n = grid.nx * grid.ny;
  const out = new Float32Array(n * 4);
  for (let i = 0; i < n; i++) {
    out[i * 4] = grid.ca[i];
    out[i * 4 + 1] = grid.cb[i];
    out[i * 4 + 2] = grid.pec[i] ? 1 : 0;
  }
  return out;
}

async function describeAdapter(adapter) {
  let info = adapter.info ?? {};
  if (typeof adapter.requestAdapterInfo === "function") {
    try {
      info = await adapter.requestAdapterInfo();
    } catch {
      info = adapter.info ?? info;
    }
  }
  return {
    vendor: info.vendor ?? "",
    architecture: info.architecture ?? "",
    device: info.device ?? "",
    description: info.description ?? "",
    isFallbackAdapter: adapter.info?.isFallbackAdapter === true || info?.isFallbackAdapter === true,
  };
}

function packPml(grid) {
  const span = Math.max(grid.nx, grid.ny);
  const out = new Float32Array(8 * span);
  const put = (slot, src) => out.set(src, slot * span);
  put(0, grid.aEx);
  put(1, grid.bEx);
  put(2, grid.aEy);
  put(3, grid.bEy);
  put(4, grid.aHx);
  put(5, grid.bHx);
  put(6, grid.aHy);
  put(7, grid.bHy);
  return { data: out, span };
}

export async function createWebGPUStepper(grid) {
  if (typeof navigator === "undefined" || !navigator.gpu) return null;
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) return null;
  const adapterInfo = await describeAdapter(adapter);
  const device = await adapter.requestDevice();
  const gpuErrors = [];
  device.addEventListener("uncapturederror", (event) => {
    gpuErrors.push(String(event.error?.message || event.error));
  });

  const { nx, ny } = grid;
  const n = nx * ny;
  const pmlPack = packPml(grid);
  const params = new ArrayBuffer(64);
  const u32 = new Uint32Array(params);
  const i32 = new Int32Array(params);
  const f = new Float32Array(params);
  u32[0] = nx;
  u32[1] = ny;
  u32[2] = grid.periodicY ? 1 : 0;
  u32[3] = pmlPack.span;
  i32[4] = grid.iInt0;
  i32[5] = grid.iInt1;
  i32[6] = grid.jInt0;
  i32[7] = grid.jInt1;
  i32[8] = grid.hx0;
  i32[9] = grid.hx1;
  i32[10] = grid.hy0;
  i32[11] = grid.hy1;
  f[12] = grid.invDx;
  f[13] = grid.invDy;
  f[14] = grid.ch;

  const buffers = [
    device.createBuffer({ size: 64, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST }),
    storage(device, f32(grid.field)),
    storage(device, f32(grid.Hx)),
    storage(device, f32(grid.Hy)),
    storage(device, packPsi(grid)),
    storage(device, packMat(grid)),
    storage(device, new Float32Array(n)),
    storage(device, pmlPack.data),
  ];
  device.queue.writeBuffer(buffers[0], 0, params);

  const module = device.createShaderModule({ code: WGSL });
  const compiled = await module.getCompilationInfo();
  const messages = compiled.messages.filter((m) => m.type === "error");
  if (messages.length) throw new Error(messages.map((m) => m.message).join("\n"));

  const entries = buffers.map((buffer, binding) => ({ binding, resource: { buffer } }));
  const pipeline = (entryPoint) =>
    device.createComputePipeline({ layout: "auto", compute: { module, entryPoint } });
  const pipes = {
    hx: pipeline("hxMain"),
    hy: pipeline("hyMain"),
    ez: pipeline("ezMain"),
  };
  const used = {
    hx: [0, 1, 2, 4, 7],
    hy: [0, 1, 3, 4, 7],
    ez: [0, 1, 2, 3, 4, 5, 6, 7],
  };
  const groups = {};
  for (const name of Object.keys(pipes)) {
    groups[name] = device.createBindGroup({
      layout: pipes[name].getBindGroupLayout(0),
      entries: used[name].map((binding) => entries[binding]),
    });
  }

  const sources = grid.sources.map((s) => ({ ...s, acc: s.acc ?? 0 }));
  const probes = grid.probes.map((p) => ({ i: p.i, j: p.j, samples: p.samples.slice() }));
  const probeIdx = new Uint32Array(Math.max(1, probes.length));
  for (let i = 0; i < probes.length; i++) probeIdx[i] = probes[i].j * nx + probes[i].i;
  const probeIndexBuf = storage(device, probeIdx);
  const probeUniform = device.createBuffer({
    size: 16,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  const probePipe = pipeline("probeMain");
  const delta = new Float32Array(n);
  const readback = new Float32Array(n);
  let time = grid.time;
  let pending = Promise.resolve();
  const gx = Math.ceil(nx / 8);
  const gy = Math.ceil(ny / 8);

  async function readFieldBuffer() {
    const bytes = n * 4;
    const staging = device.createBuffer({
      size: Math.max(16, bytes + ((16 - (bytes % 16)) % 16)),
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });
    const encoder = device.createCommandEncoder();
    encoder.copyBufferToBuffer(buffers[1], 0, staging, 0, bytes);
    device.queue.submit([encoder.finish()]);
    await staging.mapAsync(GPUMapMode.READ);
    readback.set(new Float32Array(staging.getMappedRange(), 0, n));
    staging.unmap();
    staging.destroy();
    return readback.slice();
  }

  async function readProbeHist(histBuf, count) {
    const nProbes = probes.length;
    const floats = count * nProbes;
    const bytes = Math.max(16, (floats * 4 + 15) & ~15);
    const staging = device.createBuffer({
      size: bytes,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });
    const encoder = device.createCommandEncoder();
    encoder.copyBufferToBuffer(histBuf, 0, staging, 0, bytes);
    device.queue.submit([encoder.finish()]);
    await staging.mapAsync(GPUMapMode.READ);
    const data = new Float32Array(staging.getMappedRange()).slice(0, floats);
    staging.unmap();
    staging.destroy();
    for (let s = 0; s < count; s++) {
      for (let q = 0; q < nProbes; q++) probes[q].samples.push(data[s * nProbes + q]);
    }
  }

  async function doSteps(count) {
    const nSteps = count | 0;
    if (nSteps <= 0) return;
    const nProbes = probes.length;
    let histBuf = null;
    let probeGroup = null;
    if (nProbes) {
      const bytes = Math.max(16, ((nSteps * nProbes * 4 + 15) & ~15));
      histBuf = device.createBuffer({
        size: bytes,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
      });
      probeGroup = device.createBindGroup({
        layout: probePipe.getBindGroupLayout(0),
        entries: [
          { binding: 1, resource: { buffer: buffers[1] } },
          { binding: 8, resource: { buffer: probeUniform } },
          { binding: 9, resource: { buffer: probeIndexBuf } },
          { binding: 10, resource: { buffer: histBuf } },
        ],
      });
    }
    for (let s = 0; s < nSteps; s++) {
      time += grid.dt;
      delta.fill(0);
      for (let q = 0; q < sources.length; q++) {
        const src = sources[q];
        const cell = src.j * nx + src.i;
        if (grid.pec[cell]) continue;
        delta[cell] += accumulateSource(src, time);
      }
      device.queue.writeBuffer(buffers[6], 0, delta);
      if (nProbes) {
        device.queue.writeBuffer(probeUniform, 0, new Uint32Array([nProbes, s, nProbes, 0]));
      }
      const encoder = device.createCommandEncoder();
      for (const name of ["hx", "hy", "ez"]) {
        const pass = encoder.beginComputePass();
        pass.setPipeline(pipes[name]);
        pass.setBindGroup(0, groups[name]);
        pass.dispatchWorkgroups(gx, gy);
        pass.end();
      }
      if (nProbes) {
        const pass = encoder.beginComputePass();
        pass.setPipeline(probePipe);
        pass.setBindGroup(0, probeGroup);
        pass.dispatchWorkgroups(Math.ceil(nProbes / 64));
        pass.end();
      }
      device.queue.submit([encoder.finish()]);
    }
    if (nProbes) {
      await readProbeHist(histBuf, nSteps);
      histBuf.destroy();
    }
    if (gpuErrors.length) throw new Error(gpuErrors.join("\n"));
  }

  function step(count = 1) {
    const run = pending.then(() => doSteps(count | 0));
    pending = run.then(
      () => {},
      () => {},
    );
    return run;
  }

  return {
    backend: "webgpu",
    adapter: adapterInfo,
    step,
    readField: () => pending.then(() => readFieldBuffer()),
    readProbes: () => probes.map((p) => ({ i: p.i, j: p.j, samples: p.samples })),
    destroy() {
      probeIndexBuf.destroy();
      probeUniform.destroy();
      for (const buffer of buffers) buffer.destroy();
      device.destroy();
    },
  };
}
