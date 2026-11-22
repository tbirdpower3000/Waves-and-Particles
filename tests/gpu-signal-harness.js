import crypto from 'node:crypto';
import { spawn } from 'node:child_process';
import fs from 'node:fs';
import http from 'node:http';
import net from 'node:net';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const THEORY = 2.3882907809328075e-3;

function findChrome() {
  const candidates = [
    '/usr/bin/google-chrome',
    '/usr/bin/google-chrome-stable',
    '/usr/bin/chromium',
    '/usr/bin/chromium-browser',
    '/usr/local/bin/google-chrome',
  ];
  return candidates.find((candidate) => fs.existsSync(candidate)) ?? null;
}

const HARNESS = `<!doctype html>
<meta charset="utf-8">
<script>
window.__BOOT = 'page';
window.addEventListener('error', (event) => {
  window.__RESULT = { unknown: false, error: String(event.message || event.error) };
});
window.addEventListener('unhandledrejection', (event) => {
  const reason = event.reason;
  window.__RESULT = { unknown: false, error: String(reason && reason.stack ? reason.stack : reason) };
});
</script>
<script type="module">
import { createGpuSignal } from '/src/gpu-signal/index.js';
import { makeRng } from '/src/rng.js';
import { awgn, ber, bitsRandom, convolveChannel, demodulate, fftMag, modulate } from '/src/signal.js';

const THEORY = ${THEORY};
const N = 1000000;

function relL2(cpu, gpu) {
  if (cpu.length !== gpu.length) throw new Error('length ' + cpu.length + ' vs ' + gpu.length);
  let num = 0;
  let den = 0;
  for (let i = 0; i < cpu.length; i++) {
    const d = gpu[i] - cpu[i];
    num += d * d;
    den += cpu[i] * cpu[i];
  }
  return Math.sqrt(num / Math.max(den, 1e-300));
}

function adapterReport(adapter) {
  const info = adapter.info || {};
  const description = String(info.description || '');
  const architecture = String(info.architecture || '');
  const vendor = String(info.vendor || '');
  const deviceName = String(info.device || '');
  const blob = [vendor, architecture, deviceName, description].join(' ');
  const fallback = info.isFallbackAdapter === true;
  const software = fallback || /swiftshader|llvmpipe|softpipe|software/i.test(blob);
  const name = architecture && description.toLowerCase().indexOf(architecture.toLowerCase()) === -1
    ? (description + ' ' + architecture).trim()
    : (description || architecture || vendor || 'unknown');
  let label = name;
  if (/swiftshader/i.test(blob)) label = 'software renderer (SwiftShader)';
  else if (/llvmpipe/i.test(blob)) label = 'software renderer (llvmpipe)';
  else if (/softpipe/i.test(blob)) label = 'software renderer (softpipe)';
  else if (software) label = 'software renderer (software)';
  return {
    description: name,
    label,
    vendor,
    architecture,
    device: deviceName,
    fallback,
    software,
  };
}

try {
  if (!navigator.gpu) {
    window.__RESULT = { unknown: true, reason: 'navigator.gpu is missing' };
  } else {
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) {
      window.__RESULT = { unknown: true, reason: 'requestAdapter returned null' };
    } else {
      const report = adapterReport(adapter);
      const device = await adapter.requestDevice();
      const gpu = await createGpuSignal(device);

      const tx = bitsRandom(4096, makeRng(3));
      let roundtripMismatches = 0;
      for (const spec of [{ scheme: 'bpsk', sps: 4 }, { scheme: 'qpsk', sps: 2 }]) {
        const cpuBits = demodulate(modulate(tx, spec), spec);
        const gpuWave = await gpu.modulate(tx, spec);
        const gpuBits = await gpu.demodulate(gpuWave, spec);
        const n = Math.min(cpuBits.length, gpuBits.length);
        if (cpuBits.length !== gpuBits.length) roundtripMismatches += Math.abs(cpuBits.length - gpuBits.length);
        for (let i = 0; i < n; i++) if (cpuBits[i] !== gpuBits[i]) roundtripMismatches++;
      }

      const qpskTx = tx.length % 2 === 0 ? tx : tx.subarray(0, tx.length - 1);
      const qpskCpu = demodulate(modulate(qpskTx, { scheme: 'qpsk', sps: 2 }), { scheme: 'qpsk', sps: 2 });
      const qpskGpu = await gpu.demodulate(await gpu.modulate(qpskTx, { scheme: 'qpsk', sps: 2 }), { scheme: 'qpsk', sps: 2 });
      if (qpskCpu.length !== qpskGpu.length) roundtripMismatches += 1;

      const toneN = 64;
      const k0 = 5;
      const toneI = new Float64Array(toneN);
      const toneQ = new Float64Array(toneN);
      for (let n = 0; n < toneN; n++) {
        const th = 2 * Math.PI * k0 * n / toneN;
        toneI[n] = Math.cos(th);
        toneQ[n] = Math.sin(th);
      }
      const cpuMag = fftMag({ i: toneI, q: toneQ });
      const gpuMag = await gpu.fftMag({ i: toneI, q: toneQ });
      const fftRel = relL2(cpuMag, gpuMag);

      const firN = 180;
      const firM = 17;
      const firI = new Float64Array(firN);
      const firQ = new Float64Array(firN);
      const taps = new Float64Array(firM);
      for (let n = 0; n < firN; n++) {
        firI[n] = Math.sin(n * 0.37);
        firQ[n] = Math.cos(n * 0.21);
      }
      for (let k = 0; k < firM; k++) taps[k] = (k === 3 ? 0 : Math.cos(k * 0.5) * 0.25);
      const cpuFir = convolveChannel({ i: firI, q: firQ }, taps);
      const gpuFir = await gpu.convolveChannel({ i: firI, q: firQ }, taps);
      let firNum = 0;
      let firDen = 0;
      for (let n = 0; n < cpuFir.i.length; n++) {
        const di = gpuFir.i[n] - cpuFir.i[n];
        const dq = gpuFir.q[n] - cpuFir.q[n];
        firNum += di * di + dq * dq;
        firDen += cpuFir.i[n] * cpuFir.i[n] + cpuFir.q[n] * cpuFir.q[n];
      }
      const firRel = Math.sqrt(firNum / Math.max(firDen, 1e-300));

      const cpuRng = makeRng(20261121);
      const cpuT0 = performance.now();
      const cpuTx = bitsRandom(N, cpuRng);
      const cpuMod = modulate(cpuTx, { scheme: 'bpsk', sps: 1 });
      const cpuNoisy = awgn(cpuMod, { ebn0dB: 6, bitsPerSymbol: 1, sps: 1, rng: cpuRng, pRef: 1 });
      const cpuRx = demodulate(cpuNoisy, { scheme: 'bpsk', sps: 1, offset: 0, nSymbols: N });
      const cpuMs = performance.now() - cpuT0;
      const cpuErrors = Math.round(ber(cpuTx, cpuRx) * N);

      const gpuT0 = performance.now();
      const bpsk = await gpu.countErrors({
        nBits: N,
        scheme: 'bpsk',
        sps: 1,
        ebn0dB: 6,
        seed: 20261121,
        h: [1],
        offset: 0,
        nSymbols: N,
        pRef: 1,
        bitsPerSymbol: 1,
      });
      const qpsk = await gpu.countErrors({
        nBits: N,
        scheme: 'qpsk',
        sps: 1,
        ebn0dB: 6,
        seed: 20261121,
        h: [1],
        offset: 0,
        nSymbols: N / 2,
        pRef: 2,
        bitsPerSymbol: 2,
      });
      const gpuMs = performance.now() - gpuT0;

      async function threw(fn, pattern) {
        try {
          await fn();
          return false;
        } catch (err) {
          return pattern.test(String(err && err.message ? err.message : err));
        }
      }
      const limitBits = await threw(() => gpu.bitsRandom(5_000_000, 1), /above the limit/);
      const limitMod = await threw(
        () => gpu.modulate(new Uint8Array(5_000_000), { scheme: 'bpsk', sps: 1 }),
        /above the limit/,
      );
      const limitSweep = await threw(
        () => gpu.countErrors({ nBits: 8_000_000, scheme: 'bpsk', ebn0dB: 0, seed: 1, h: [1] }),
        /above the limit/,
      );
      const badOffset = await threw(
        () => gpu.countErrors({ nBits: 128, scheme: 'bpsk', ebn0dB: 6, seed: 1, h: [1], offset: 1000 }),
        /offset/,
      );
      const badSymbols = await threw(
        () => gpu.countErrors({ nBits: 64, scheme: 'bpsk', ebn0dB: 6, seed: 1, h: [1], nSymbols: 1000 }),
        /nSymbols/,
      );
      const badK = await threw(
        () => gpu.countErrors({ nBits: 64, scheme: 'bpsk', ebn0dB: 6, seed: 1, bitsPerSymbol: 2 }),
        /conflicts/,
      );
      const half = await gpu.countErrors({
        nBits: 200_000, scheme: 'bpsk', sps: 1, ebn0dB: 6, seed: 7, h: [0.5],
      });
      const loud = await gpu.countErrors({
        nBits: 200_000, scheme: 'bpsk', sps: 1, ebn0dB: 6, seed: 7, h: [2],
      });
      const zerosI = new Float64Array(64);
      const zerosQ = new Float64Array(64);
      const noiseA = await gpu.awgn(
        { i: zerosI, q: zerosQ },
        { ebn0dB: 6, bitsPerSymbol: 1, sps: 1, seed: 5, pRef: 1 },
      );
      const noiseB = await gpu.awgn(
        { i: zerosI, q: zerosQ },
        { ebn0dB: 6, bitsPerSymbol: 1, sps: 1, seed: 5 ^ 1, pRef: 1 },
      );
      let swapped = 0;
      for (let i = 0; i < noiseA.i.length; i++) {
        if (Math.abs(noiseA.i[i] - noiseB.q[i]) < 1e-5 && Math.abs(noiseA.q[i] - noiseB.i[i]) < 1e-5) swapped++;
      }
      const randN = 256;
      const randI = new Float64Array(randN);
      const randQ = new Float64Array(randN);
      let state = 1;
      for (let i = 0; i < randN; i++) {
        state = (Math.imul(state, 1103515245) + 12345) >>> 0;
        randI[i] = state / 2 ** 32 * 2 - 1;
        state = (Math.imul(state, 1103515245) + 12345) >>> 0;
        randQ[i] = state / 2 ** 32 * 2 - 1;
      }
      const randRel = relL2(fftMag({ i: randI, q: randQ }), await gpu.fftMag({ i: randI, q: randQ }));
      const tailSymbols = await threw(
        () => gpu.countErrors({
          nBits: 32, scheme: 'bpsk', ebn0dB: 6, seed: 1, h: [1, 0.25, 0.1, 0], nSymbols: 33,
        }),
        /tail/,
      );
      const nullH = await threw(
        () => gpu.convolveChannel({ i: new Float64Array([1, 0]), q: new Float64Array([0, 0]) }, null),
        /h must/,
      );
      const acN = 8192;
      const ac = await gpu.awgn(
        { i: new Float64Array(acN), q: new Float64Array(acN) },
        { ebn0dB: 6, bitsPerSymbol: 1, sps: 1, seed: 99, pRef: 1 },
      );
      let acMean = 0;
      for (let i = 0; i < acN; i++) acMean += ac.i[i];
      acMean /= acN;
      let acVar = 0;
      let acCov = 0;
      for (let i = 0; i < acN; i++) {
        const d = ac.i[i] - acMean;
        acVar += d * d;
        if (i + 1 < acN) acCov += d * (ac.i[i + 1] - acMean);
      }
      const acZ = (acCov / acVar) * Math.sqrt(acN - 1);
      const sweep = [];
      for (const spec of [
        { db: 0, nBits: 100000 },
        { db: 4, nBits: 100000 },
        { db: 8, nBits: 400000 },
      ]) {
        const row = await gpu.countErrors({
          nBits: spec.nBits, scheme: 'bpsk', sps: 1, ebn0dB: spec.db, seed: 11, h: [1], pRef: 1,
        });
        sweep.push({ db: spec.db, errors: row.errors, nBits: row.nBits });
      }
      async function timeFft(n) {
        const i = new Float64Array(n);
        const q = new Float64Array(n);
        let s = 7;
        for (let k = 0; k < n; k++) {
          s = (Math.imul(s, 1103515245) + 12345) >>> 0;
          i[k] = s / 4294967296 * 2 - 1;
          s = (Math.imul(s, 1103515245) + 12345) >>> 0;
          q[k] = s / 4294967296 * 2 - 1;
        }
        const c0 = performance.now();
        const cpu = fftMag({ i, q });
        const cpuMs = performance.now() - c0;
        const g0 = performance.now();
        const gpuMag = await gpu.fftMag({ i, q });
        const gpuMs = performance.now() - g0;
        return { n, cpuMs, gpuMs, rel: relL2(cpu, gpuMag) };
      }
      const fft16 = await timeFft(65536);
      const fft21 = await timeFft(2097152);
      const liveBefore = gpu.bufferCount();
      gpu.destroy();
      const liveAfter = gpu.bufferCount();
      const afterDestroy = await gpu.countErrors({
        nBits: 1024, scheme: 'bpsk', ebn0dB: 6, seed: 3, h: [1], pRef: 1,
      });
      device.destroy();

      window.__RESULT = {
        unknown: false,
        adapter: report,
        roundtripMismatches,
        fftRel,
        firRel,
        cpuErrors,
        cpuMs,
        bpskErrors: bpsk.errors,
        qpskErrors: qpsk.errors,
        nBits: N,
        gpuMs,
        theory: THEORY,
        limitBits,
        limitMod,
        limitSweep,
        badOffset,
        badSymbols,
        badK,
        halfErrors: half.errors,
        halfBits: half.nBits,
        loudErrors: loud.errors,
        loudBits: loud.nBits,
        swapped,
        randRel,
        tailSymbols,
        nullH,
        acZ,
        sweep,
        fft16,
        fft21,
        liveBefore,
        liveAfter,
        afterDestroy: afterDestroy.errors,
      };
    }
  }
} catch (err) {
  window.__RESULT = { unknown: false, error: String(err && err.stack ? err.stack : err) };
}
</script>
`;

function serve() {
  const server = http.createServer((req, res) => {
    const url = new URL(req.url ?? '/', 'http://127.0.0.1');
    if (url.pathname === '/harness.html') {
      res.writeHead(200, { 'content-type': 'text/html; charset=utf-8' });
      res.end(HARNESS);
      return;
    }
    const rel = decodeURIComponent(url.pathname);
    const file = path.normalize(path.join(root, rel));
    const rootPrefix = root.endsWith(path.sep) ? root : root + path.sep;
    if (file !== root && !file.startsWith(rootPrefix)) {
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
      const type = file.endsWith('.js') ? 'text/javascript' : 'application/octet-stream';
      res.writeHead(200, { 'content-type': type });
      res.end(data);
    });
  });
  return new Promise((resolve) => {
    server.listen(0, '127.0.0.1', () => resolve(server));
  });
}

function waitFor(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function maskFrame(text) {
  const payload = Buffer.from(text);
  const mask = crypto.randomBytes(4);
  const header = payload.length < 126
    ? Buffer.from([0x81, 0x80 | payload.length])
    : Buffer.alloc(4);
  if (payload.length >= 126) {
    header[0] = 0x81;
    header[1] = 0x80 | 126;
    header.writeUInt16BE(payload.length, 2);
  }
  const masked = Buffer.alloc(payload.length);
  for (let i = 0; i < payload.length; i++) masked[i] = payload[i] ^ mask[i & 3];
  return Buffer.concat([header, mask, masked]);
}

// Node 20 has no global WebSocket. DevTools only needs a small client.
function openWebSocket(url) {
  const parsed = new URL(url);
  return new Promise((resolve, reject) => {
    const socket = net.connect(Number(parsed.port), parsed.hostname);
    const key = crypto.randomBytes(16).toString('base64');
    let buf = Buffer.alloc(0);
    let accepted = false;
    const listeners = { message: [], error: [] };
    const queued = [];
    const ws = {
      addEventListener(type, fn) {
        listeners[type].push(fn);
        if (type === 'message') {
          for (const data of queued) fn({ data });
          queued.length = 0;
        }
      },
      send(data) { socket.write(maskFrame(data)); },
      close() { socket.end(); },
    };
    const fail = (err) => {
      listeners.error.forEach((fn) => fn(err));
      reject(err);
    };
    socket.on('error', fail);
    socket.on('data', (chunk) => {
      buf = Buffer.concat([buf, chunk]);
      if (!accepted) {
        const end = buf.indexOf('\r\n\r\n');
        if (end < 0) return;
        const header = buf.slice(0, end).toString('latin1');
        buf = buf.slice(end + 4);
        if (!header.startsWith('HTTP/1.1 101')) {
          fail(new Error('devtools socket failed'));
          socket.destroy();
          return;
        }
        accepted = true;
        resolve(ws);
      }
      while (buf.length >= 2) {
        const opcode = buf[0] & 0x0f;
        let len = buf[1] & 0x7f;
        let offset = 2;
        if (len === 126) {
          if (buf.length < 4) return;
          len = buf.readUInt16BE(2);
          offset = 4;
        } else if (len === 127) {
          if (buf.length < 10) return;
          len = Number(buf.readBigUInt64BE(2));
          offset = 10;
        }
        if (buf.length < offset + len) return;
        const payload = buf.subarray(offset, offset + len);
        buf = buf.subarray(offset + len);
        if (opcode === 1) {
          const data = payload.toString('utf8');
          if (listeners.message.length === 0) queued.push(data);
          else listeners.message.forEach((fn) => fn({ data }));
        } else if (opcode === 8) {
          socket.end();
        } else if (opcode === 9) {
          socket.write(Buffer.from([0x8a, 0x00]));
        }
      }
    });
    socket.on('connect', () => {
      const target = `${parsed.pathname}${parsed.search}`;
      socket.write(
        `GET ${target} HTTP/1.1\r\n` +
        `Host: ${parsed.host}\r\n` +
        'Upgrade: websocket\r\n' +
        'Connection: Upgrade\r\n' +
        `Sec-WebSocket-Key: ${key}\r\n` +
        'Sec-WebSocket-Version: 13\r\n\r\n',
      );
    });
  });
}

async function chromeResult(chrome, pageUrl) {
  const port = 9800 + Math.floor(Math.random() * 400);
  const profile = `/tmp/wap-signal-chrome-${port}`;
  fs.rmSync(profile, { recursive: true, force: true });
  const proc = spawn(chrome, [
    '--headless=new',
    '--no-sandbox',
    '--disable-dev-shm-usage',
    '--disable-gpu-sandbox',
    '--enable-unsafe-swiftshader',
    '--use-gl=angle',
    '--use-angle=swiftshader',
    '--enable-unsafe-webgpu',
    `--remote-allow-origins=http://127.0.0.1:${port}`,
    `--remote-debugging-port=${port}`,
    `--user-data-dir=${profile}`,
    'about:blank',
  ], { stdio: 'ignore' });
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
    if (!version) return { unknown: true, reason: 'chrome DevTools did not start' };
    const opened = await fetch(`http://127.0.0.1:${port}/json/new?${encodeURIComponent(pageUrl)}`, { method: 'PUT' });
    if (!opened.ok) return { unknown: true, reason: `could not open a Chrome page (${opened.status})` };
    const page = await opened.json();
    const ws = await openWebSocket(page.webSocketDebuggerUrl);
    let id = 0;
    const pending = new Map();
    const logs = [];
    ws.addEventListener('message', (ev) => {
      const msg = JSON.parse(ev.data);
      if (msg.id && pending.has(msg.id)) {
        pending.get(msg.id)(msg.result);
        pending.delete(msg.id);
      }
      if (msg.method === 'Runtime.consoleAPICalled') {
        logs.push(msg.params.args.map((arg) => arg.value ?? arg.description ?? '').join(' '));
      }
      if (msg.method === 'Runtime.exceptionThrown') {
        const detail = msg.params.exceptionDetails;
        logs.push(detail.exception?.description || detail.text);
      }
    });
    const send = (method, params) => new Promise((resolve) => {
      const msgId = ++id;
      pending.set(msgId, resolve);
      ws.send(JSON.stringify({ id: msgId, method, params }));
    });
    await send('Runtime.enable');
    let result = null;
    for (let i = 0; i < 240 && !result; i++) {
      await waitFor(250);
      const evaled = await send('Runtime.evaluate', {
        expression: 'window.__RESULT ? JSON.stringify(window.__RESULT) : ""',
        returnByValue: true,
      });
      const value = evaled?.result?.value;
      if (value) result = JSON.parse(value);
    }
    ws.close();
    if (result) return result;
    return { unknown: false, error: logs.join('\n') || 'harness timed out' };
  } finally {
    try {
      proc.kill('SIGKILL');
    } catch {
      /* already exited */
    }
    await waitFor(200);
    try {
      fs.rmSync(profile, { recursive: true, force: true });
    } catch {
      /* profile files can outlive the process */
    }
  }
}

export async function runGpuSignalCheck() {
  const chrome = findChrome();
  if (!chrome) return { unknown: true, reason: 'chrome is not installed' };
  const server = await serve();
  const { port } = server.address();
  try {
    return await chromeResult(chrome, `http://127.0.0.1:${port}/harness.html`);
  } finally {
    await new Promise((resolve) => server.close(resolve));
  }
}
