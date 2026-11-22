import assert from "node:assert/strict";
import { spawn } from "node:child_process";
import fs from "node:fs";
import http from "node:http";
import path from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";

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
  window.__RESULT = { backend: null, error: String(event.message || event.error) };
});
window.addEventListener("unhandledrejection", (event) => {
  const reason = event.reason;
  window.__RESULT = { backend: null, error: String(reason && reason.stack ? reason.stack : reason) };
});
</script>
<script type="module">
import { C0, createGrid, setMaterial, addSource, addProbe, step } from "/src/fdtd.js";
import { createGpuStepper } from "/src/gpu/index.js";

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
  gpu.destroy();
  window.__RESULT = {
    backend: gpu.backend,
    rows,
    probeSamples: probes[0] ? probes[0].samples.length : 0,
  };
} catch (err) {
  const message = String(err && err.message ? err.message : err);
  window.__RESULT = {
    backend: message.includes("UNKNOWN") ? null : "error",
    error: message,
  };
}
</script>
`;

function serve() {
  const server = http.createServer((req, res) => {
    const url = new URL(req.url ?? "/", "http://127.0.0.1");
    if (url.pathname === "/harness.html") {
      res.writeHead(200, { "content-type": "text/html; charset=utf-8" });
      res.end(HARNESS);
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
    if (!version) return { backend: null, error: "chrome DevTools did not start" };
    const opened = await fetch(`http://127.0.0.1:${port}/json/new?${encodeURIComponent(pageUrl)}`, {
      method: "PUT",
    });
    if (!opened.ok) return { backend: null, error: "could not open a Chrome page" };
    const page = await opened.json();
    const ws = new WebSocket(page.webSocketDebuggerUrl);
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
    return { backend: null, error: logs.join("\n") || "harness timed out" };
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
  if (!result || result.backend == null) {
    console.log("UNKNOWN: no GPU backend");
    if (result?.error) console.log(result.error);
    t.skip("UNKNOWN: no GPU backend");
    return;
  }
  if (result.backend === "error") {
    assert.fail(result.error || "GPU backend failed");
  }
  for (const row of result.rows) {
    console.log(
      `A7 ${result.backend} n=${row.n} relL2=${row.relL2.toExponential(3)} max|d|/peak=${row.maxOverPeak.toExponential(3)} gpuPeak=${row.gpuPeak?.toExponential(3)} cpuPeak=${row.cpuPeak?.toExponential(3)}`,
    );
  }
  const at500 = result.rows.find((row) => row.n === 500);
  assert.ok(at500, "missing 500-step sample");
  assert.ok(at500.relL2 < 1e-3, `rel L2 ${at500.relL2}`);
  console.log(
    `A7 PASS ${result.backend} relL2(500)=${at500.relL2.toExponential(3)} < 1e-3; probes recorded=${result.probeSamples}`,
  );
});
