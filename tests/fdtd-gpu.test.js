import assert from "node:assert/strict";
import crypto from "node:crypto";
import { spawn } from "node:child_process";
import fs from "node:fs";
import http from "node:http";
import net from "node:net";
import path from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";
import { fieldToImage } from "../src/gpu/color.js";
import { createGpuStepper, createGpuView } from "../src/gpu/index.js";
import { classifyWebGLRenderer, createWebGL2Stepper } from "../src/gpu/webgl2.js";
import { createWebGPUStepper } from "../src/gpu/webgpu.js";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");

function findChrome() {
  const candidates = [
    "/usr/bin/google-chrome",
    "/usr/bin/google-chrome-stable",
    "/usr/local/bin/google-chrome",
  ];
  return candidates.find((p) => fs.existsSync(p)) ?? null;
}

const HARNESS = `<!doctype html>
<meta charset="utf-8">
<script>
window.__BOOT = "page";
window.addEventListener("error", (event) => {
  if (window.__RESULT) return;
  window.__RESULT = { absent: false, error: String(event.message || event.error) };
});
window.addEventListener("unhandledrejection", (event) => {
  if (window.__RESULT) return;
  const reason = event.reason;
  window.__RESULT = { absent: false, error: String(reason && reason.stack ? reason.stack : reason) };
});
</script>
<script type="module">
import { C0, createGrid, setMaterial, addSource, addProbe, step } from "/src/fdtd.js";
import { fieldToImage } from "/src/gpu/color.js";
import { createGpuStepper, createGpuView } from "/src/gpu/index.js";

function metrics(ref, other) {
  let num = 0, den = 0, peak = 0, maxAbs = 0;
  for (let i = 0; i < ref.length; i++) {
    const d = other[i] - ref[i];
    num += d * d;
    den += ref[i] * ref[i];
    peak = Math.max(peak, Math.abs(ref[i]));
    maxAbs = Math.max(maxAbs, Math.abs(d));
  }
  return {
    relL2: Math.sqrt(num / Math.max(den, 1e-300)),
    maxOverPeak: maxAbs / Math.max(peak, 1e-300),
  };
}

function make() {
  const g = createGrid({ nx: 36, ny: 28, dx: 1, dy: 1, mode: "tmz", pml: 6 });
  setMaterial(g, { x0: 16, y0: 6, x1: 28, y1: 20, epsR: 4 });
  setMaterial(g, { x0: 10, y0: 4, x1: 12, y1: 22, pec: 1 });
  const f0 = C0 / 16;
  addSource(g, { i: 8, j: 14, kind: "gaussian", f0, t0: 1.5 / f0, spread: 0.6 / f0, amp: 1 });
  addProbe(g, { i: 14, j: 14 });
  return g;
}

try {
  const cpu = make();
  const backend = new URLSearchParams(location.search).get("backend") || "auto";
  const gpu = await createGpuStepper(make(), { backend });
  const marks = [100, 500, 2000];
  const rows = [];
  let done = 0;
  for (const n of marks) {
    await gpu.step(n - done);
    step(cpu, n - done);
    done = n;
    const field = await gpu.readField();
    const m = metrics(cpu.field, field);
    let gpuPeak = 0, cpuPeak = 0;
    for (let i = 0; i < field.length; i++) {
      gpuPeak = Math.max(gpuPeak, Math.abs(field[i]));
      cpuPeak = Math.max(cpuPeak, Math.abs(cpu.field[i]));
    }
    rows.push({ n, relL2: m.relL2, maxOverPeak: m.maxOverPeak, gpuPeak, cpuPeak });
  }
  const probes = gpu.readProbes();
  const cpuProbe = cpu.probes[0].samples;
  const gpuProbe = probes[0] ? probes[0].samples : [];
  let pnum = 0, pden = 0;
  const pn = Math.min(cpuProbe.length, gpuProbe.length);
  for (let i = 0; i < pn; i++) {
    const d = gpuProbe[i] - cpuProbe[i];
    pnum += d * d;
    pden += cpuProbe[i] * cpuProbe[i];
  }
  const perf = [];
  gpu.destroy();
  for (const spec of [{ n: 256, steps: 40 }, { n: 512, steps: 20 }, { n: 1024, steps: 8 }]) {
    const bench = createGrid({ nx: spec.n, ny: spec.n, dx: 1, dy: 1, pml: 8 });
    const c = (spec.n / 2) | 0;
    const f0b = C0 / 20;
    addSource(bench, { i: c, j: c, kind: "sine", f0: f0b, amp: 1 });
    const stepper = await createGpuStepper(bench, { backend });
    await stepper.step(2);
    await stepper.idle();
    const t0 = performance.now();
    stepper.kick(spec.steps);
    await stepper.idle();
    const seconds = (performance.now() - t0) / 1000;
    stepper.destroy();
    perf.push({ n: spec.n, steps: spec.steps, stepsPerSec: spec.steps / seconds });
  }
  const adapter = gpu.adapter || null;
  const text = adapter ? [adapter.vendor, adapter.architecture, adapter.device, adapter.description].join(" ") : "";
  const software = !!(adapter && (adapter.software || adapter.isFallbackAdapter || /swiftshader|llvmpipe|softpipe|lavapipe|software/i.test(text)));
  const hardware = !!(adapter && adapter.hardware === true && !software);
  if (gpu.fallbackReason) console.log("GPU auto-fallback: " + gpu.fallbackReason);
  window.__RESULT = {
    absent: false,
    software,
    hardware,
    unknown: !software && !hardware,
    backend: gpu.backend,
    adapter,
    fallbackReason: gpu.fallbackReason || "",
    rows,
    probeSamples: gpuProbe.length,
    probeRel: Math.sqrt(pnum / Math.max(pden, 1e-300)),
    perf,
  };
} catch (err) {
  const message = String(err && err.stack ? err.stack : err);
  window.__RESULT = {
    absent: /UNKNOWN: no GPU backend/.test(message),
    error: message,
  };
}
</script>
`;

function serve() {
  const server = http.createServer((req, res) => {
    const url = new URL(req.url ?? "/", "http://127.0.0.1");
    if (url.pathname === "/harness.html" || url.pathname === "/view.html") {
      res.writeHead(200, { "content-type": "text/html; charset=utf-8" });
      res.end(url.pathname === "/view.html" ? VIEW : HARNESS);
      return;
    }
    const rel = decodeURIComponent(url.pathname);
    const file = path.normalize(path.join(root, rel));
    if (!file.startsWith(root)) {
      res.writeHead(403);
      res.end();
      return;
    }
    fs.readFile(file, (err, data) => {
      if (err) {
        res.writeHead(404);
        res.end();
        return;
      }
      const type = file.endsWith(".js") ? "text/javascript" : "application/octet-stream";
      res.writeHead(200, { "content-type": type });
      res.end(data);
    });
  });
  return new Promise((resolve) => {
    server.listen(0, "127.0.0.1", () => resolve(server));
  });
}

function waitFor(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

// Minimal client-masked WebSocket. Node 20 has no global WebSocket; Node 22+ does,
// but this path is the same on 20, 22, and 24 and needs no npm package.
function connectCdp(url) {
  const target = new URL(url);
  const socket = net.connect(Number(target.port), target.hostname);
  const listeners = { open: [], message: [], error: [], close: [] };
  const pending = [];
  let opened = false;
  let buf = Buffer.alloc(0);
  let handshook = false;
  const api = {
    addEventListener(type, fn) {
      listeners[type].push(fn);
    },
    send(data) {
      const frame = encodeClientFrame(String(data));
      if (!opened) pending.push(frame);
      else socket.write(frame);
    },
    close() {
      socket.end();
    },
  };
  const emit = (type, ev) => {
    for (const fn of listeners[type]) fn(ev);
  };
  socket.on("error", (err) => emit("error", err));
  socket.on("close", () => emit("close", {}));
  socket.on("connect", () => {
    const key = crypto.randomBytes(16).toString("base64");
    const pathAndQuery = `${target.pathname}${target.search}`;
    socket.write(
      `GET ${pathAndQuery} HTTP/1.1\r\n` +
        `Host: ${target.host}\r\n` +
        "Upgrade: websocket\r\n" +
        "Connection: Upgrade\r\n" +
        `Sec-WebSocket-Key: ${key}\r\n` +
        "Sec-WebSocket-Version: 13\r\n\r\n",
    );
  });
  socket.on("data", (chunk) => {
    buf = Buffer.concat([buf, chunk]);
    if (!handshook) {
      const end = buf.indexOf("\r\n\r\n");
      if (end < 0) return;
      const header = buf.subarray(0, end).toString("latin1");
      buf = buf.subarray(end + 4);
      if (!header.startsWith("HTTP/1.1 101")) {
        emit("error", new Error(header.split("\r\n")[0] || "websocket upgrade failed"));
        socket.destroy();
        return;
      }
      handshook = true;
      opened = true;
      for (const frame of pending) socket.write(frame);
      pending.length = 0;
      emit("open", {});
    }
    parseFrames();
  });

  function parseFrames() {
    while (buf.length >= 2) {
      const b0 = buf[0];
      const b1 = buf[1];
      const opcode = b0 & 0x0f;
      let length = b1 & 0x7f;
      let offset = 2;
      if (length === 126) {
        if (buf.length < 4) return;
        length = buf.readUInt16BE(2);
        offset = 4;
      } else if (length === 127) {
        if (buf.length < 10) return;
        const big = buf.readBigUInt64BE(2);
        length = Number(big);
        offset = 10;
      }
      const masked = (b1 & 0x80) !== 0;
      const maskLen = masked ? 4 : 0;
      if (buf.length < offset + maskLen + length) return;
      let payload = buf.subarray(offset + maskLen, offset + maskLen + length);
      if (masked) {
        const mask = buf.subarray(offset, offset + 4);
        payload = Buffer.from(payload);
        for (let i = 0; i < payload.length; i++) payload[i] ^= mask[i & 3];
      }
      buf = buf.subarray(offset + maskLen + length);
      if (opcode === 0x9) {
        socket.write(encodeClientFrame(payload, 0x8a));
        continue;
      }
      if (opcode === 0x8) {
        socket.end();
        continue;
      }
      if (opcode === 0x1) emit("message", { data: payload.toString("utf8") });
    }
  }
  return api;
}

function encodeClientFrame(data, opcode = 0x81) {
  const payload = Buffer.isBuffer(data) ? data : Buffer.from(data);
  const mask = crypto.randomBytes(4);
  let header;
  if (payload.length < 126) {
    header = Buffer.from([opcode, 0x80 | payload.length]);
  } else if (payload.length < 65536) {
    header = Buffer.alloc(4);
    header[0] = opcode;
    header[1] = 0x80 | 126;
    header.writeUInt16BE(payload.length, 2);
  } else {
    header = Buffer.alloc(10);
    header[0] = opcode;
    header[1] = 0x80 | 127;
    header.writeBigUInt64BE(BigInt(payload.length), 2);
  }
  const masked = Buffer.alloc(payload.length);
  for (let i = 0; i < payload.length; i++) masked[i] = payload[i] ^ mask[i & 3];
  return Buffer.concat([header, mask, masked]);
}

async function chromeResult(chrome, pageUrl) {
  const port = 9400 + Math.floor(Math.random() * 400);
  const profile = `/tmp/wap-chrome-${port}`;
  fs.rmSync(profile, { recursive: true, force: true });
  const proc = spawn(
    chrome,
    [
      "--headless=new",
      "--no-sandbox",
      "--disable-dev-shm-usage",
      "--disable-gpu-sandbox",
      "--enable-unsafe-swiftshader",
      "--use-gl=angle",
      "--use-angle=swiftshader",
      "--enable-unsafe-webgpu",
      `--remote-debugging-port=${port}`,
      `--user-data-dir=${profile}`,
      "about:blank",
    ],
    { stdio: "ignore" },
  );
  try {
    let version = null;
    for (let i = 0; i < 40 && !version; i++) {
      await waitFor(250);
      try {
        const res = await fetch(`http://127.0.0.1:${port}/json/version`);
        if (res.ok) version = await res.json();
      } catch {
        version = null;
      }
    }
    if (!version) return { absent: true, error: "chrome DevTools did not start" };
    const opened = await fetch(`http://127.0.0.1:${port}/json/new?${encodeURIComponent(pageUrl)}`, {
      method: "PUT",
    });
    if (!opened.ok) return { absent: true, error: "could not open a Chrome page" };
    const page = await opened.json();
    const ws = connectCdp(page.webSocketDebuggerUrl);
    await new Promise((resolve, reject) => {
      ws.addEventListener("open", resolve);
      ws.addEventListener("error", () => reject(new Error("devtools socket failed")));
    });
    let id = 0;
    const pending = new Map();
    const logs = [];
    ws.addEventListener("message", (ev) => {
      const msg = JSON.parse(ev.data);
      if (msg.id && pending.has(msg.id)) {
        pending.get(msg.id)(msg.result);
        pending.delete(msg.id);
      }
      if (msg.method === "Runtime.consoleAPICalled") {
        logs.push(msg.params.args.map((arg) => arg.value ?? arg.description ?? "").join(" "));
      }
      if (msg.method === "Runtime.exceptionThrown") {
        const detail = msg.params.exceptionDetails;
        logs.push(detail.exception?.description || detail.text);
      }
    });
    const send = (method, params) =>
      new Promise((resolve) => {
        const msgId = ++id;
        pending.set(msgId, resolve);
        ws.send(JSON.stringify({ id: msgId, method, params }));
      });
    await send("Runtime.enable");
    let result = null;
    for (let i = 0; i < 160 && !result; i++) {
      await waitFor(250);
      const evaled = await send("Runtime.evaluate", {
        expression: "window.__RESULT ? JSON.stringify(window.__RESULT) : ''",
        returnByValue: true,
      });
      const value = evaled?.result?.value;
      if (value) result = JSON.parse(value);
    }
    ws.close();
    if (result) return result;
    return { absent: false, error: logs.join("\n") || "harness timed out" };
  } finally {
    try {
      proc.kill("SIGKILL");
    } catch {
      /* already exited */
    }
    await new Promise((resolve) => setTimeout(resolve, 200));
    try {
      fs.rmSync(profile, { recursive: true, force: true });
    } catch {
      /* profile files can outlive the process */
    }
  }
}

test("gpu modules import on Node", () => {
  assert.equal(typeof createGpuStepper, "function");
  assert.equal(typeof createGpuView, "function");
  assert.equal(typeof createWebGPUStepper, "function");
  assert.equal(typeof createWebGL2Stepper, "function");
});

test("CPU view draws the field without a GPU readback", async () => {
  const { C0, createGrid, addSource, addProbe, step } = await import("../src/fdtd.js");
  const make = () => {
    const g = createGrid({ nx: 24, ny: 18, dx: 1, dy: 1, pml: 4 });
    addSource(g, { i: 8, j: 9, kind: "sine", f0: C0 / 12, amp: 1 });
    addProbe(g, { i: 12, j: 9 });
    return g;
  };
  const preview = make();
  step(preview, 16);
  let peak = 0;
  for (let i = 0; i < preview.field.length; i++) peak = Math.max(peak, Math.abs(preview.field[i]));
  const grid = make();
  const stored = { data: null };
  const canvas = {
    width: 0,
    height: 0,
    getContext(kind) {
      if (kind !== "2d") return null;
      return {
        putImageData(image) {
          stored.data = image.data;
        },
      };
    },
  };
  const view = await createGpuView(canvas, grid, { scale: peak });
  assert.equal(view.backend, "cpu");
  assert.equal(view.probeStream, null);
  view.frame(4);
  view.frame(12);
  const drawn = await view.readDrawn();
  const cpu = make();
  step(cpu, 16);
  const expected = fieldToImage(cpu.field, cpu.nx, cpu.ny, peak);
  assert.equal(view.fullFieldReads, 0);
  assert.equal(drawn.length, expected.length);
  for (let i = 0; i < drawn.length; i++) assert.equal(drawn[i], expected[i]);
});

test("A7 GPU vs CPU rel L2 < 1e-3 after 500 steps", async (t) => {
  const chrome = findChrome();
  if (!chrome) {
    console.log("UNKNOWN: no GPU backend");
    t.skip("UNKNOWN: no GPU backend");
    return;
  }
  const server = await serve();
  const { port } = server.address();
  let result;
  try {
    const forced = process.env.WAP_GPU_BACKEND;
    const query = forced ? `?backend=${encodeURIComponent(forced)}` : "";
    result = await chromeResult(chrome, `http://127.0.0.1:${port}/harness.html${query}`);
  } finally {
    await new Promise((resolve) => server.close(resolve));
  }
  if (!result) assert.fail("GPU harness returned no result");
  if (result.error && !result.rows) {
    if (result.absent) {
      console.log("UNKNOWN: no GPU backend");
      console.log(result.error);
      t.skip("UNKNOWN: no GPU backend");
      return;
    }
    assert.fail(result.error);
  }
  const adapter = result.adapter || {};
  if (result.fallbackReason) console.log(`GPU auto-fallback: ${result.fallbackReason}`);
  const kindLabel = result.hardware ? "hardware" : result.software ? "software" : "unknown";
  console.log(`A7 adapter ${JSON.stringify(adapter)} ${kindLabel}`);
  for (const row of result.rows) {
    const kind = kindLabel;
    console.log(
      `A7 ${result.backend} ${kind} n=${row.n} relL2=${row.relL2.toExponential(3)} max|d|/peak=${row.maxOverPeak.toExponential(3)} gpuPeak=${row.gpuPeak?.toExponential(3)} cpuPeak=${row.cpuPeak?.toExponential(3)}`,
    );
  }
  if (result.perf) {
    for (const row of result.perf) {
      console.log(
        `A9 GPU ${result.backend} ${kindLabel} ${row.n}^2 steps/s=${row.stepsPerSec.toFixed(2)} (${row.steps} steps)`,
      );
    }
  }
  const at500 = result.rows.find((row) => row.n === 500);
  assert.ok(at500 && Number.isFinite(at500.relL2), "missing 500-step sample");
  assert.ok(at500.relL2 < 1e-3, `rel L2 ${at500.relL2}`);
  assert.ok(result.probeSamples === 2000, `probe samples ${result.probeSamples}`);
  assert.ok(result.probeRel < 1e-3, `probe rel L2 ${result.probeRel}`);
  if (!result.hardware) {
    const why = result.software ? "software adapter" : "masked or unrecognized renderer";
    console.log(`A7 UNKNOWN: ${why}; not a hardware pass`);
    t.skip(`UNKNOWN: ${why}`);
    return;
  }
  console.log(
    `A7 PASS ${result.backend} hardware relL2(500)=${at500.relL2.toExponential(3)} < 1e-3; probes recorded=${result.probeSamples}`,
  );
});

test("A7 forced WebGL2 on SwiftShader is software UNKNOWN", async (t) => {
  const chrome = findChrome();
  if (!chrome) {
    console.log("UNKNOWN: no GPU backend");
    t.skip("UNKNOWN: no GPU backend");
    return;
  }
  const server = await serve();
  const { port } = server.address();
  let result;
  try {
    result = await chromeResult(chrome, `http://127.0.0.1:${port}/harness.html?backend=webgl2`);
  } finally {
    await new Promise((resolve) => server.close(resolve));
  }
  if (!result) assert.fail("GPU harness returned no result");
  if (result.error && !result.rows) {
    if (result.absent) {
      console.log("UNKNOWN: no GPU backend");
      console.log(result.error);
      t.skip("UNKNOWN: no GPU backend");
      return;
    }
    assert.fail(result.error);
  }
  const adapter = result.adapter || {};
  console.log(`A7 webgl2 adapter ${JSON.stringify(adapter)} hardware=${result.hardware === true} software=${result.software === true}`);
  assert.equal(result.backend, "webgl2");
  if (result.fallbackReason) console.log(`GPU auto-fallback: ${result.fallbackReason}`);
  const at500 = result.rows.find((row) => row.n === 500);
  assert.ok(at500 && at500.relL2 < 1e-3, `rel L2 ${at500 && at500.relL2}`);
  assert.ok(result.probeSamples === 2000, `probe samples ${result.probeSamples}`);
  assert.ok(result.probeRel < 1e-3, `probe rel L2 ${result.probeRel}`);
  if (result.hardware !== true) {
    const why = result.software ? "software=true (ANGLE SwiftShader)" : "masked or unrecognized renderer";
    console.log(`A7 WebGL2 UNKNOWN: ${why}; not a hardware pass`);
    t.skip(`UNKNOWN: WebGL2 ${why}`);
    return;
  }
  console.log(`A7 PASS webgl2 hardware relL2(500)=${at500.relL2.toExponential(3)}`);
});

test("masked WebGL renderer is UNKNOWN, not hardware", () => {
  const VENDOR = 0x1f00;
  const RENDERER = 0x1f01;
  const gl = {
    VENDOR,
    RENDERER,
    getExtension() {
      return null;
    },
    getParameter(name) {
      if (name === VENDOR) return "WebKit";
      if (name === RENDERER) return "WebKit WebGL";
      return "";
    },
  };
  const info = classifyWebGLRenderer(gl);
  assert.equal(info.unmasked, false);
  assert.equal(info.hardware, false);
  assert.equal(info.software, false);
  assert.equal(info.description, "WebKit WebGL");
  console.log("A7 WebGL2 UNKNOWN: masked renderer WebKit WebGL; not a hardware pass");
});

const VIEW = `<!doctype html>
<meta charset="utf-8">
<script>
window.addEventListener("error", (event) => {
  if (window.__RESULT) return;
  window.__RESULT = { absent: false, error: String(window.__GPU || "") + " " + String(event.message || event.error) };
});
window.addEventListener("unhandledrejection", (event) => {
  if (window.__RESULT) return;
  const reason = event.reason;
  window.__RESULT = { absent: false, error: String(window.__GPU || "") + " " + String(reason && reason.message ? reason.message : reason) };
});
</script>
<script type="module">
import { C0, createGrid, setMaterial, addSource, addProbe, step } from "/src/fdtd.js";
import { fieldToImage } from "/src/gpu/color.js";
import { createGpuStepper, createGpuView } from "/src/gpu/index.js";

function metrics(ref, other) {
  let num = 0, den = 0;
  for (let i = 0; i < ref.length; i++) {
    const d = other[i] - ref[i];
    num += d * d;
    den += ref[i] * ref[i];
  }
  return Math.sqrt(num / Math.max(den, 1e-300));
}

function make() {
  const g = createGrid({ nx: 40, ny: 32, dx: 1, dy: 1, pml: 6 });
  addSource(g, { i: 12, j: 16, kind: "sine", f0: C0 / 12, amp: 1 });
  addProbe(g, { i: 20, j: 16 });
  return g;
}

try {
  const backend = new URLSearchParams(location.search).get("backend") || "auto";
  const preview = make();
  step(preview, 32);
  let peak = 0;
  for (let i = 0; i < preview.field.length; i++) peak = Math.max(peak, Math.abs(preview.field[i]));
  const live = make();
  const cpu = make();
  const canvas = document.createElement("canvas");
  document.body.appendChild(canvas);
  const view = await createGpuView(canvas, live, { backend: backend, stepsPerFrame: 4, scale: peak });
  window.__STEP = "view";
  for (let i = 0; i < 8; i++) view.frame();
  window.__STEP = "framed";
  const queued = view.framesInFlight;
  await view.idle();
  window.__STEP = "idle";
  step(cpu, 32);
  const snap = document.createElement("canvas");
  snap.width = canvas.width;
  snap.height = canvas.height;
  const snapCtx = snap.getContext("2d", { willReadFrequently: true });
  if (canvas.getContext("webgpu")) {
    snapCtx.drawImage(canvas, 0, 0, canvas.width, canvas.height);
  } else {
    snapCtx.drawImage(canvas, 0, 0, canvas.width, canvas.height);
  }
  const drawn = snapCtx.getImageData(0, 0, canvas.width, canvas.height).data;
  const expected = fieldToImage(cpu.field, cpu.nx, cpu.ny, peak);
  let maxDelta = 0;
  for (let i = 0; i < drawn.length; i++) maxDelta = Math.max(maxDelta, Math.abs(drawn[i] - expected[i]));
  const probes = view.readProbes();
  const stream = view.probeStream;
  const edited = make();
  const reference = make();
  const stepper = await createGpuStepper(edited, { backend: backend });
  const before = stepper.materialBytesUploaded;
  setMaterial(edited, { x0: 8, y0: 8, x1: 10, y1: 9, pec: 1 });
  setMaterial(reference, { x0: 8, y0: 8, x1: 10, y1: 9, pec: 1 });
  const changed = stepper.syncMaterials(edited);
  const uploaded = stepper.materialBytesUploaded - before;
  await stepper.step(24);
  step(reference, 24);
  const relL2 = metrics(reference.field, await stepper.readField());
  stepper.destroy();
  const adapter = view.adapter || {};
  const text = [adapter.vendor, adapter.architecture, adapter.device, adapter.description].join(" ");
  const software = !!(adapter && (adapter.software || adapter.isFallbackAdapter || /swiftshader|llvmpipe|softpipe|lavapipe|software/i.test(text)));
  const hardware = !!(adapter && adapter.hardware === true && !software);
  window.__RESULT = {
    absent: false,
    software: software,
    hardware: hardware,
    backend: view.backend,
    adapter: adapter,
    maxDelta: maxDelta,
    lit: drawn.some((v, i) => i % 4 !== 3 && v !== 0),
    hasDevice: !!view.device,
    queued: queued,
    fullFieldReads: view.fullFieldReads,
    probeSamples: probes[0] ? probes[0].samples.length : 0,
    stream: stream ? {
      stride: stream.stride,
      capacity: stream.capacity,
      probeCount: stream.probeCount,
      format: stream.format,
      headByteOffset: stream.headByteOffset,
      hasBuffer: !!stream.buffer,
    } : null,
    changed: changed,
    uploaded: uploaded,
    relL2: relL2,
    cells: edited.nx * edited.ny,
  };
} catch (err) {
  const message = String(err && err.stack ? err.stack : err);
  window.__RESULT = { absent: /UNKNOWN: no GPU backend/.test(message), error: (window.__STEP || "start") + " swap=" + (window.__SWAP || "") + " " + message };
}
</script>
`;

test("drawn field matches the CPU image", async (t) => {
  const chrome = findChrome();
  if (!chrome) {
    console.log("UNKNOWN: no GPU backend");
    t.skip("UNKNOWN: no GPU backend");
    return;
  }
  const server = await serve();
  const { port } = server.address();
  let result;
  try {
    result = await chromeResult(chrome, `http://127.0.0.1:${port}/view.html`);
  } finally {
    await new Promise((resolve) => server.close(resolve));
  }
  if (!result) assert.fail("GPU view harness returned no result");
  if (result.error && result.maxDelta == null) {
    if (result.absent) {
      console.log("UNKNOWN: no GPU backend");
      console.log(result.error);
      t.skip("UNKNOWN: no GPU backend");
      return;
    }
    assert.fail(result.error);
  }
  const adapter = result.adapter || {};
  console.log(
    `view ${result.backend} adapter ${JSON.stringify(adapter)} maxDelta=${result.maxDelta} relL2=${result.relL2} uploaded=${result.uploaded} changed=${result.changed} reads=${result.fullFieldReads}`,
  );
  assert.equal(result.fullFieldReads, 0);
  assert.equal(result.lit, true, "on-screen canvas has no lit pixels");
  if (result.backend === "webgpu") assert.ok(result.queued <= 2, `frames in flight ${result.queued}`);
  assert.equal(result.probeSamples, 32);
  assert.ok(result.changed > 0 && result.changed < result.cells);
  assert.equal(result.changed, 6);
  assert.ok(result.uploaded > 0 && result.uploaded <= result.changed * 16, `uploaded ${result.uploaded}`);
  if (result.backend === "webgpu") {
    assert.equal(result.hasDevice, true);
    assert.equal(result.stream.capacity, 4096);
    assert.equal(result.stream.stride, 1);
    assert.equal(result.stream.format, "f32");
    assert.equal(result.stream.headByteOffset, 8);
    assert.equal(result.stream.hasBuffer, true);
  }
  assert.ok(result.maxDelta <= 1, `drawn field max channel delta ${result.maxDelta}`);
  assert.ok(result.relL2 < 1e-3, `patched rel L2 ${result.relL2}`);
  if (result.hardware !== true) {
    const why = result.software ? "software adapter" : "masked or unrecognized renderer";
    console.log(`drawn field UNKNOWN: ${why}; numbers are not a hardware pass`);
    t.skip(`UNKNOWN: ${why}`);
    return;
  }
  console.log(`drawn field PASS ${result.backend} hardware maxDelta=${result.maxDelta}`);
});

test("on-screen WebGL2 canvas matches the CPU view", async (t) => {
  const chrome = findChrome();
  if (!chrome) {
    console.log("UNKNOWN: no GPU backend");
    t.skip("UNKNOWN: no GPU backend");
    return;
  }
  const server = await serve();
  const { port } = server.address();
  let result;
  try {
    result = await chromeResult(chrome, `http://127.0.0.1:${port}/view.html?backend=webgl2`);
  } finally {
    await new Promise((resolve) => server.close(resolve));
  }
  if (!result) assert.fail("WebGL2 view harness returned no result");
  if (result.error && result.maxDelta == null) {
    if (result.absent) {
      console.log("UNKNOWN: no GPU backend");
      t.skip("UNKNOWN: no GPU backend");
      return;
    }
    assert.fail(result.error);
  }
  console.log(
    `webgl2 canvas maxDelta=${result.maxDelta} lit=${result.lit} relL2=${result.relL2} reads=${result.fullFieldReads}`,
  );
  assert.equal(result.backend, "webgl2");
  assert.equal(result.fullFieldReads, 0);
  assert.equal(result.lit, true, "on-screen WebGL2 canvas has no lit pixels");
  assert.equal(result.probeSamples, 32);
  assert.ok(result.maxDelta <= 1, `on-screen max channel delta ${result.maxDelta}`);
  if (result.hardware !== true) {
    console.log("WebGL2 canvas UNKNOWN: software or unrecognized renderer; not a hardware pass");
    t.skip("UNKNOWN: WebGL2 renderer");
    return;
  }
  console.log("WebGL2 canvas PASS hardware");
});

const RING = `<!doctype html>
<meta charset="utf-8">
<script type="module">
import { C0, createGrid, addSource, addProbe, step } from "/src/fdtd.js";
import { createGpuView } from "/src/gpu/index.js";
try {
  const steps = 4096 + 128;
  const make = () => {
    const g = createGrid({ nx: 18, ny: 14, dx: 1, dy: 1, pml: 4 });
    addSource(g, { i: 6, j: 7, kind: "sine", f0: C0 / 10, amp: 1 });
    addProbe(g, { i: 10, j: 7 });
    return g;
  };
  const cpu = make();
  const canvas = document.createElement("canvas");
  document.body.appendChild(canvas);
  const view = await createGpuView(canvas, make(), { backend: "webgl2", stepsPerFrame: 128 });
  const frames = steps / 128;
  for (let i = 0; i < frames; i++) view.frame();
  await view.idle();
  step(cpu, steps);
  const gpu = view.readProbes()[0].samples;
  const ref = cpu.probes[0].samples;
  let num = 0, den = 0;
  const n = Math.min(gpu.length, ref.length);
  for (let i = 0; i < n; i++) {
    const d = gpu[i] - ref[i];
    num += d * d;
    den += ref[i] * ref[i];
  }
  window.__RESULT = {
    absent: false,
    samples: gpu.length,
    expected: ref.length,
    relL2: Math.sqrt(num / Math.max(den, 1e-300)),
  };
} catch (err) {
  const message = String(err && err.stack ? err.stack : err);
  window.__RESULT = { absent: /UNKNOWN: no GPU backend/.test(message), error: message };
}
</script>
`;

test("WebGL2 probe ring survives more than 4096 steps", async (t) => {
  const chrome = findChrome();
  if (!chrome) {
    console.log("UNKNOWN: no GPU backend");
    t.skip("UNKNOWN: no GPU backend");
    return;
  }
  const server = http.createServer((req, res) => {
    const url = new URL(req.url ?? "/", "http://127.0.0.1");
    if (url.pathname === "/ring.html") {
      res.writeHead(200, { "content-type": "text/html; charset=utf-8" });
      res.end(RING);
      return;
    }
    const rel = decodeURIComponent(url.pathname);
    const file = path.normalize(path.join(root, rel));
    if (!file.startsWith(root)) {
      res.writeHead(403);
      res.end();
      return;
    }
    fs.readFile(file, (err, data) => {
      if (err) {
        res.writeHead(404);
        res.end();
        return;
      }
      res.writeHead(200, { "content-type": "text/javascript" });
      res.end(data);
    });
  });
  await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
  const { port } = server.address();
  let result;
  try {
    result = await chromeResult(chrome, `http://127.0.0.1:${port}/ring.html`);
  } finally {
    await new Promise((resolve) => server.close(resolve));
  }
  if (!result) assert.fail("probe ring harness returned no result");
  if (result.error) {
    if (result.absent) {
      console.log("UNKNOWN: no GPU backend");
      t.skip("UNKNOWN: no GPU backend");
      return;
    }
    assert.fail(result.error);
  }
  console.log(`WebGL2 probe ring samples=${result.samples} relL2=${result.relL2}`);
  assert.equal(result.samples, result.expected);
  assert.ok(result.relL2 < 1e-5, `WebGL2 probe rel L2 ${result.relL2}`);
  console.log("WebGL2 probe ring PASS (software renderer, numeric gate)");
});
