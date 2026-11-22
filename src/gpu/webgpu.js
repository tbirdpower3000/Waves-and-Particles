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
  field[here] = e;
}

struct Clock {
  time: f32,
  dt: f32,
  step: u32,
  batchStart: u32,
  nSrc: u32,
  nProbes: u32,
  ring: u32,
  pad: u32,
}
struct SrcMeta {
  cell: u32,
  pec: u32,
  pad0: u32,
  pad1: u32,
}
@group(0) @binding(6) var<storage, read> acc: array<f32>;
@group(0) @binding(8) var<storage, read_write> clock: Clock;
@group(0) @binding(11) var<storage, read> srcMeta: array<SrcMeta>;
@group(0) @binding(9) var<storage, read> probeIndex: array<u32>;
@group(0) @binding(10) var<storage, read_write> hist: array<f32>;

@compute @workgroup_size(1)
fn tickMain() {
  clock.time = clock.time + clock.dt;
  clock.step = clock.step + 1u;
}

@compute @workgroup_size(1)
fn sourceMain() {
  let s = clock.step - 1u - clock.batchStart;
  for (var q = 0u; q < clock.nSrc; q = q + 1u) {
    let src = srcMeta[q];
    if (src.pec != 0u) { continue; }
    field[src.cell] = field[src.cell] + acc[s * clock.nSrc + q];
  }
}

@compute @workgroup_size(64)
fn probeMain(@builtin(global_invocation_id) gid: vec3u) {
  let p = gid.x;
  if (p >= clock.nProbes) { return; }
  let s = clock.step - 1u;
  hist[(s % clock.ring) * clock.nProbes + p] = field[probeIndex[p]];
}
`;

const DRAW = /* wgsl */ `
struct Draw {
  nx: u32,
  ny: u32,
  scale: f32,
  pad: f32,
}
@group(0) @binding(0) var<uniform> draw: Draw;
@group(0) @binding(1) var<storage, read> field: array<f32>;

struct VsOut {
  @builtin(position) pos: vec4f,
}

@vertex
fn vs(@builtin(vertex_index) i: u32) -> VsOut {
  var corner = array<vec2f, 3>(vec2f(-1.0, -1.0), vec2f(3.0, -1.0), vec2f(-1.0, 3.0));
  var o: VsOut;
  o.pos = vec4f(corner[i], 0.0, 1.0);
  return o;
}

@fragment
fn fs(@builtin(position) pos: vec4f) -> @location(0) vec4f {
  let x = u32(pos.x);
  let y = u32(pos.y);
  if (x >= draw.nx || y >= draw.ny) { return vec4f(0.0, 0.0, 0.0, 1.0); }
  let v = field[y * draw.nx + x];
  let t = clamp(0.5 + 0.5 * v / draw.scale, 0.0, 1.0);
  return vec4f(t, t, t, 1.0);
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
  const vendor = info.vendor ?? "";
  const architecture = info.architecture ?? "";
  const deviceName = info.device ?? "";
  const description = info.description ?? "";
  const blob = `${vendor} ${architecture} ${deviceName} ${description}`;
  const software =
    adapter.info?.isFallbackAdapter === true ||
    info?.isFallbackAdapter === true ||
    /swiftshader|llvmpipe|softpipe|lavapipe|software/i.test(blob);
  const named = blob.trim().length > 0;
  return {
    vendor,
    architecture,
    device: deviceName,
    description,
    isFallbackAdapter: software,
    software,
    hardware: named && !software,
    unmasked: true,
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
    storage(device, new Float32Array(4)),
    storage(device, pmlPack.data),
  ];
  device.queue.writeBuffer(buffers[0], 0, params);

  const module = device.createShaderModule({ code: WGSL });
  const drawModule = device.createShaderModule({ code: DRAW });
  const compiled = await module.getCompilationInfo();
  const drawCompiled = await drawModule.getCompilationInfo();
  const messages = [...compiled.messages, ...drawCompiled.messages].filter((m) => m.type === "error");
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
    ez: [0, 1, 2, 3, 4, 5, 7],
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
  const meta = new Uint32Array(Math.max(1, sources.length) * 4);
  for (let q = 0; q < sources.length; q++) {
    const cell = sources[q].j * nx + sources[q].i;
    meta[q * 4] = cell;
    meta[q * 4 + 1] = grid.pec[cell] ? 1 : 0;
  }
  const srcMetaBuf = storage(device, meta);
  const clockBuf = device.createBuffer({
    size: 32,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
  });
  const tickPipe = pipeline("tickMain");
  const sourcePipe = pipeline("sourceMain");
  const probePipe = pipeline("probeMain");
  const tickGroup = device.createBindGroup({
    layout: tickPipe.getBindGroupLayout(0),
    entries: [{ binding: 8, resource: { buffer: clockBuf } }],
  });
  const RING = 4096;
  const nSrc = sources.length;
  const nProbes = probes.length;
  const histBuf = storage(device, new Float32Array(RING * Math.max(nProbes, 1)));
  const stageBytes = Math.max(16, ((RING * Math.max(nProbes, 1) * 4 + 15) & ~15));
  const slots = [0, 1].map(() => ({
    buffer: device.createBuffer({
      size: stageBytes,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    }),
    busy: false,
  }));
  const matMirror = packMat(grid);
  const readback = new Float32Array(n);
  const pendingAcc = [];
  const jobs = [];
  let time = grid.time;
  let head = 0;
  let fullFieldReads = 0;
  let materialBytesUploaded = 0;
  let paint = null;
  let settled = Promise.resolve();
  let failure = null;
  let pumping = false;
  const gx = Math.ceil(nx / 8);
  const gy = Math.ceil(ny / 8);

  function track(promise) {
    const guarded = promise.then(
      (value) => value,
      (err) => {
        failure = failure || err;
        throw err;
      },
    );
    settled = settled.then(() => guarded).then(
      () => {},
      () => {},
    );
    return guarded;
  }

  function uploadRuns(ids) {
    if (!ids.length) return;
    ids.sort((a, b) => a - b);
    let start = ids[0];
    let prev = ids[0];
    const flush = (from, to) => {
      const slice = matMirror.subarray(from * 4, (to + 1) * 4);
      device.queue.writeBuffer(buffers[5], from * 16, slice);
      materialBytesUploaded += slice.byteLength;
    };
    for (let k = 1; k <= ids.length; k++) {
      if (k === ids.length || ids[k] !== prev + 1) {
        flush(start, prev);
        start = ids[k];
      }
      prev = ids[k];
    }
  }

  function syncMaterials(next = grid) {
    const changed = [];
    const bits = new Float32Array(1);
    const asF32 = (value) => {
      bits[0] = value;
      return bits[0];
    };
    for (let id = 0; id < n; id++) {
      const pec = next.pec[id] ? 1 : 0;
      const o = id * 4;
      const ca = asF32(next.ca[id]);
      const cb = asF32(next.cb[id]);
      if (matMirror[o] !== ca || matMirror[o + 1] !== cb || matMirror[o + 2] !== pec) {
        matMirror[o] = ca;
        matMirror[o + 1] = cb;
        matMirror[o + 2] = pec;
        changed.push(id);
      }
    }
    uploadRuns(changed);
    let metaDirty = false;
    for (let q = 0; q < nSrc; q++) {
      const cell = sources[q].j * nx + sources[q].i;
      const pec = next.pec[cell] ? 1 : 0;
      if (meta[q * 4 + 1] !== pec) {
        meta[q * 4 + 1] = pec;
        metaDirty = true;
      }
    }
    if (metaDirty) {
      device.queue.writeBuffer(srcMetaBuf, 0, meta);
      materialBytesUploaded += meta.byteLength;
    }
    return changed.length;
  }

  async function readFieldBuffer() {
    fullFieldReads += 1;
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

  function appendProbes(data, count) {
    for (let s = 0; s < count; s++) {
      for (let q = 0; q < nProbes; q++) probes[q].samples.push(data[s * nProbes + q]);
    }
  }

  function submitChunk(part, slot, draw) {
    const clock = new ArrayBuffer(32);
    const clockF = new Float32Array(clock);
    const clockU = new Uint32Array(clock);
    clockF[0] = part.t0;
    clockF[1] = grid.dt;
    clockU[2] = part.batchStart;
    clockU[3] = part.batchStart;
    clockU[4] = nSrc;
    clockU[5] = nProbes;
    clockU[6] = RING;
    device.queue.writeBuffer(clockBuf, 0, clock);
    const accBuf = storage(device, part.acc);
    const sourceGroup = device.createBindGroup({
      layout: sourcePipe.getBindGroupLayout(0),
      entries: [
        { binding: 1, resource: { buffer: buffers[1] } },
        { binding: 6, resource: { buffer: accBuf } },
        { binding: 8, resource: { buffer: clockBuf } },
        { binding: 11, resource: { buffer: srcMetaBuf } },
      ],
    });
    let probeGroup = null;
    if (nProbes) {
      probeGroup = device.createBindGroup({
        layout: probePipe.getBindGroupLayout(0),
        entries: [
          { binding: 1, resource: { buffer: buffers[1] } },
          { binding: 8, resource: { buffer: clockBuf } },
          { binding: 9, resource: { buffer: probeIndexBuf } },
          { binding: 10, resource: { buffer: histBuf } },
        ],
      });
    }
    const encoder = device.createCommandEncoder();
    for (let s = 0; s < part.chunk; s++) {
      for (const name of ["hx", "hy", "ez"]) {
        const pass = encoder.beginComputePass();
        pass.setPipeline(pipes[name]);
        pass.setBindGroup(0, groups[name]);
        pass.dispatchWorkgroups(gx, gy);
        pass.end();
      }
      const tick = encoder.beginComputePass();
      tick.setPipeline(tickPipe);
      tick.setBindGroup(0, tickGroup);
      tick.dispatchWorkgroups(1);
      tick.end();
      if (nSrc) {
        const srcPass = encoder.beginComputePass();
        srcPass.setPipeline(sourcePipe);
        srcPass.setBindGroup(0, sourceGroup);
        srcPass.dispatchWorkgroups(1);
        srcPass.end();
      }
      if (nProbes) {
        const probePass = encoder.beginComputePass();
        probePass.setPipeline(probePipe);
        probePass.setBindGroup(0, probeGroup);
        probePass.dispatchWorkgroups(Math.ceil(nProbes / 64));
        probePass.end();
      }
    }
    if (nProbes) {
      const bytes = part.chunk * nProbes * 4;
      encoder.copyBufferToBuffer(histBuf, part.ringOffset * nProbes * 4, slot.buffer, 0, bytes);
    }
    device.queue.submit([encoder.finish()]);
    if (draw && paint) {
      const drawEncoder = device.createCommandEncoder();
      paint(drawEncoder);
      device.queue.submit([drawEncoder.finish()]);
    }
    return accBuf;
  }

  function pump() {
    if (pumping) return;
    pumping = true;
    try {
      while (jobs.length) {
        const job = jobs[0];
        if (!job.prepared) {
          syncMaterials();
          const prepared = [];
          let left = job.count;
          while (left > 0) {
            const ringOffset = head % RING;
            const chunk = Math.min(left, RING - ringOffset);
            const acc = new Float32Array(Math.max(1, chunk * Math.max(nSrc, 1)));
            const t0 = time;
            for (let s = 0; s < chunk; s++) {
              time += grid.dt;
              for (let q = 0; q < nSrc; q++) {
                const src = sources[q];
                const cell = src.j * nx + src.i;
                acc[s * Math.max(nSrc, 1) + q] = grid.pec[cell] ? 0 : accumulateSource(src, time);
              }
            }
            prepared.push({ acc, t0, chunk, batchStart: head, ringOffset });
            head += chunk;
            left -= chunk;
          }
          job.prepared = prepared;
          job.chain = Promise.resolve();
        }
        if (!job.prepared.length) {
          jobs.shift();
          job.resolve();
          continue;
        }
        let slot = null;
        if (nProbes) {
          slot = slots.find((item) => !item.busy);
          if (!slot) break;
          slot.busy = true;
        }
        const part = job.prepared.shift();
        const last = job.prepared.length === 0;
        const accBuf = submitChunk(part, slot, last && job.draw);
        if (nProbes) {
          const mapped = slot.buffer.mapAsync(GPUMapMode.READ).then(() => {
            const floats = part.chunk * nProbes;
            const data = new Float32Array(slot.buffer.getMappedRange()).slice(0, floats);
            slot.buffer.unmap();
            accBuf.destroy();
            slot.busy = false;
            if (gpuErrors.length) throw new Error(gpuErrors.join("\n"));
            appendProbes(data, part.chunk);
            pump();
          });
          job.chain = job.chain.then(() => mapped);
          if (last) {
            jobs.shift();
            job.resolve(job.chain);
          }
        } else {
          pendingAcc.push(accBuf);
          if (last) {
            jobs.shift();
            job.resolve();
          }
        }
      }
    } finally {
      pumping = false;
    }
  }

  function kick(count = 1, opts = {}) {
    const nSteps = count | 0;
    if (nSteps <= 0) return Promise.resolve();
    let resolve;
    let reject;
    const promise = new Promise((res, rej) => {
      resolve = res;
      reject = rej;
    });
    jobs.push({
      count: nSteps,
      draw: !!opts.draw,
      prepared: null,
      chain: null,
      resolve,
      reject,
    });
    pump();
    return track(promise.then((value) => value));
  }

  function step(count = 1) {
    return kick(count | 0);
  }

  async function idle() {
    await settled;
    if (failure) throw failure;
    if (!nProbes) {
      const staging = device.createBuffer({
        size: 16,
        usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
      });
      const encoder = device.createCommandEncoder();
      encoder.copyBufferToBuffer(clockBuf, 0, staging, 0, 16);
      device.queue.submit([encoder.finish()]);
      await staging.mapAsync(GPUMapMode.READ);
      staging.unmap();
      staging.destroy();
      while (pendingAcc.length) pendingAcc.pop().destroy();
    }
    if (gpuErrors.length) throw new Error(gpuErrors.join("\n"));
  }

  function createView(canvas, opts = {}) {
    const scale = opts.scale ?? 1;
    const stepsPerFrame = opts.stepsPerFrame ?? 1;
    const context = canvas.getContext("webgpu");
    if (!context) throw new Error("canvas has no webgpu context");
    const format = navigator.gpu.getPreferredCanvasFormat();
    canvas.width = nx;
    canvas.height = ny;
    context.configure({ device, format, alphaMode: "opaque" });
    const drawParams = new ArrayBuffer(16);
    new Uint32Array(drawParams).set([nx, ny]);
    new Float32Array(drawParams)[2] = scale;
    const drawUniform = device.createBuffer({
      size: 16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(drawUniform, 0, drawParams);
    const drawPipe = device.createRenderPipeline({
      layout: "auto",
      vertex: { module: drawModule, entryPoint: "vs" },
      fragment: { module: drawModule, entryPoint: "fs", targets: [{ format }] },
      primitive: { topology: "triangle-list" },
    });
    const color = device.createTexture({
      size: [nx, ny],
      format,
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
    });
    const drawGroup = device.createBindGroup({
      layout: drawPipe.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: drawUniform } },
        { binding: 1, resource: { buffer: buffers[1] } },
      ],
    });
    const bytesPerRow = Math.ceil((nx * 4) / 256) * 256;
    const stage = device.createBuffer({
      size: bytesPerRow * ny,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });
    paint = (encoder) => {
      const pass = encoder.beginRenderPass({
        colorAttachments: [
          {
            view: color.createView(),
            loadOp: "clear",
            storeOp: "store",
            clearValue: { r: 0, g: 0, b: 0, a: 1 },
          },
        ],
      });
      pass.setPipeline(drawPipe);
      pass.setBindGroup(0, drawGroup);
      pass.draw(3);
      pass.end();
    };

    function frame(count) {
      const nFrame = count == null ? stepsPerFrame : count | 0;
      kick(nFrame, { draw: true });
    }

    async function readDrawn() {
      await idle();
      const encoder = device.createCommandEncoder();
      encoder.copyTextureToBuffer(
        { texture: color },
        { buffer: stage, bytesPerRow },
        { width: nx, height: ny },
      );
      device.queue.submit([encoder.finish()]);
      await stage.mapAsync(GPUMapMode.READ);
      const raw = new Uint8Array(stage.getMappedRange());
      const out = new Uint8ClampedArray(nx * ny * 4);
      for (let j = 0; j < ny; j++) out.set(raw.subarray(j * bytesPerRow, j * bytesPerRow + nx * 4), j * nx * 4);
      stage.unmap();
      return out;
    }

    return {
      backend: "webgpu",
      adapter: adapterInfo,
      canvas,
      stepsPerFrame,
      scale,
      get fullFieldReads() {
        return fullFieldReads;
      },
      frame,
      idle,
      readDrawn,
      readProbes: () => probes.map((p) => ({ i: p.i, j: p.j, samples: p.samples })),
      destroy() {
        drawUniform.destroy();
        stage.destroy();
        color.destroy();
      },
    };
  }

  return {
    backend: "webgpu",
    adapter: adapterInfo,
    get fullFieldReads() {
      return fullFieldReads;
    },
    get materialBytesUploaded() {
      return materialBytesUploaded;
    },
    probeStream: nProbes
      ? {
          buffer: histBuf,
          head: clockBuf,
          headByteOffset: 8,
          stride: nProbes,
          capacity: RING,
          probeCount: nProbes,
          format: "f32",
        }
      : null,
    step,
    kick,
    idle,
    syncMaterials,
    createView,
    readField: () => idle().then(() => readFieldBuffer()),
    readProbes: () => probes.map((p) => ({ i: p.i, j: p.j, samples: p.samples })),
    destroy() {
      probeIndexBuf.destroy();
      srcMetaBuf.destroy();
      clockBuf.destroy();
      histBuf.destroy();
      for (const slot of slots) slot.buffer.destroy();
      while (pendingAcc.length) pendingAcc.pop().destroy();
      for (const buffer of buffers) buffer.destroy();
      device.destroy();
    },
  };
}
