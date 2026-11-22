import { drawField } from './draw.js';
import { panelDetail } from './load-contract.js';
import { gpuBufferGate } from './params.js';
import {
  PROBE_COLORS,
  addProbe,
  cellFromOffset,
  createScene,
  materialFromRect,
  rebuildGrid,
  rectFromCells,
} from './scene.js';
import { COMING_SOON_TEXT, setReadyState } from './status.js';

function must(id) {
  const el = document.getElementById(id);
  if (!el) throw new Error(`missing #${id}`);
  return el;
}

export function formatDt(seconds) {
  if (!Number.isFinite(seconds)) return '—';
  return `${(seconds * 1e9).toFixed(3)} ns`;
}

export function mountField(doc, bus, params = {}) {
  const panel = must('panel-field');
  const canvas = must('field-canvas');
  const stateEl = must('field-state');
  const readout = must('field-readout');
  const epsInput = must('eps-r');
  const scene = createScene(params);
  epsInput.value = String(scene.epsR);

  let contract = null;
  let grid = null;
  let stepper = null;
  let lastField = null;
  let running = false;
  let recording = false;
  let steps = 0;
  let rate = 0;
  let drag = null;
  let hover = null;
  let frame = 0;
  let generation = 0;
  let gpuOn = false;
  let message = 'draw a metal wall or place a transmitter';
  let tool = 'metal';
  let solverReady = false;
  let fps = 0;
  let softwareGpu = false;
  let gpuEpoch = 0;

  const toolButtons = [...doc.querySelectorAll('[data-tool]')];
  const modeButtons = [...doc.querySelectorAll('[data-mode]')];

  function selectTool(next) {
    tool = next;
    for (const button of toolButtons) {
      button.setAttribute('aria-pressed', String(button.dataset.tool === next));
    }
  }

  function selectMode(next) {
    scene.mode = next === 'scalar' ? 'scalar' : 'tmz';
    for (const button of modeButtons) {
      button.setAttribute('aria-pressed', String(button.dataset.mode === scene.mode));
    }
    panel.dataset.mode = scene.mode;
    rebuild('time');
  }

  function updateTransport() {
    const play = must('play');
    const pause = must('pause');
    play.disabled = !solverReady || running;
    pause.disabled = !running;
    play.setAttribute('aria-pressed', String(running));
    must('use-gpu').disabled = !solverReady || !contract?.gpu?.ready;
    must('use-gpu').setAttribute('aria-pressed', String(gpuOn));
    showBackend();
  }

  function activeBackend() {
    if (!solverReady || !gpuOn) return 'cpu';
    if (softwareGpu) return 'software (SwiftShader)';
    return stepper?.backend || 'gpu';
  }

  function showBackend() {
    const el = doc.getElementById('backend-label');
    if (!el) return;
    const name = activeBackend();
    el.textContent = name;
    el.dataset.backend = name;
  }

  function updateReadout() {
    const threads = globalThis.navigator?.hardwareConcurrency ?? '?';
    const parts = [
      [message, 'The most recent event, such as running, paused, grid rebuilt or an input error.'],
      [`${scene.nx}x${scene.ny} cells`, 'Number of cells across × down. Each cell is 1 m, so the default grid is 512 m square.'],
      [scene.mode, 'Which wave model is active. See the TMz and Scalar buttons.'],
      [`dt ${formatDt(grid?.dt)}`, "Simulated time per step, set just under the 2D stability limit. It's about 2.3 nanoseconds here."],
      [`step ${steps}`, 'How many time steps have been simulated since the last reset.'],
      [`${rate.toFixed(0)} solver step/s`, 'Time steps computed per second of real time.'],
      [`${fps.toFixed(0)} frames/s`, 'Picture updates per second on your screen.'],
      [`backend ${activeBackend()}`, 'Same as the badge in the panel header: cpu, webgpu or webgl2.'],
      [`threads ${threads}`, "Logical CPU cores your browser reports. It's for information only."],
    ];
    if (hover) {
      parts.push([`cell ${hover.i},${hover.j}`, 'Column and row of the grid cell your pointer is over.']);
    }
    readout.replaceChildren();
    parts.forEach(([text, tip], index) => {
      if (index) readout.append(' · ');
      const span = document.createElement('span');
      span.textContent = text;
      span.title = tip;
      readout.append(span);
    });
  }

  function draw() {
    frame += 1;
    canvas.dataset.frame = String(frame);
    canvas.dataset.materials = String(scene.materials.length);
    canvas.dataset.probes = String(scene.probes.length);
    canvas.dataset.source = scene.source ? '1' : '0';
    canvas.dataset.steps = String(steps);
    panel.dataset.mode = scene.mode;
    const warning = drawField(canvas, {
      field: lastField,
      nx: scene.nx,
      ny: scene.ny,
      pml: scene.pml,
      scene,
      drag: drag ? { tool: drag.tool, rect: rectFromCells(drag.a, drag.b) } : null,
      solverReady,
      holdLabel: solverReady ? '' : 'Simulator unavailable',
      probeColors: PROBE_COLORS,
    });
    if (warning) message = warning;
    updateReadout();
  }

  function stopLoop() {
    running = false;
    recording = false;
    generation += 1;
  }

  function cpuStepper(fdtd, nextGrid) {
    return {
      backend: 'cpu',
      step(n = 1) { fdtd.step(nextGrid, n); },
      async readField() { return fdtd.snapshot(nextGrid); },
      async readProbes() { return nextGrid.probes; },
      destroy() {},
    };
  }

  async function readProbes() {
    try {
      const remote = await stepper?.readProbes?.();
      if (remote) return remote;
    } catch {
      /* fall through to the grid */
    }
    return grid?.probes ?? [];
  }

  function emitSamples(probes) {
    const list = Array.isArray(probes) ? probes : Array.from(probes ?? []);
    bus.emit('samples', {
      probes: list.map((probe, index) => ({
        index,
        i: probe?.i ?? scene.probes[index]?.i,
        j: probe?.j ?? scene.probes[index]?.j,
        samples: probe?.samples ?? [],
      })),
    });
  }

  async function rebuild(reason) {
    const keepGpu = gpuOn;
    stopLoop();
    if (stepper) {
      try { stepper.destroy(); } catch { /* ignore a dead stepper */ }
    }
    stepper = null;
    grid = null;
    lastField = null;
    steps = 0;
    rate = 0;
    fps = 0;
    if (!solverReady) {
      gpuOn = false;
      bus.emit('scene', { reset: true });
      draw();
      updateTransport();
      return;
    }
    try {
      gpuBufferGate(scene.nx, scene.ny);
      const kind = reason === 'record' ? 'gaussian' : (scene.source?.kind || 'sine');
      grid = rebuildGrid(scene, contract.fdtd.ns, scene.source ? kind : null);
      stepper = cpuStepper(contract.fdtd.ns, grid);
      lastField = contract.fdtd.ns.snapshot(grid);
      message = reason === 'record' ? 'recording pulse response' : 'grid rebuilt';
    } catch (err) {
      gpuOn = false;
      message = err instanceof Error ? err.message : String(err);
      setReadyState(stateEl, 'error', message);
      bus.emit('scene', { reset: true });
      draw();
      updateTransport();
      return;
    }
    bus.emit('scene', { reset: true });
    draw();
    if (keepGpu && contract?.gpu?.ready) {
      await enableGpu(true);
    } else {
      gpuOn = false;
      softwareGpu = false;
      setReadyState(stateEl, 'ready', 'cpu');
      updateTransport();
    }
  }

  async function enableGpu(on) {
    const epoch = ++gpuEpoch;
    if (!on) {
      if (solverReady && grid) stepper = cpuStepper(contract.fdtd.ns, grid);
      gpuOn = false;
      softwareGpu = false;
      message = 'backend cpu';
      if (solverReady) setReadyState(stateEl, 'ready', 'cpu');
      updateTransport();
      updateReadout();
      return;
    }
    try {
      const cells = gpuBufferGate(scene.nx, scene.ny);
      if (!contract?.gpu?.ready || !grid || cells < 1) return;
      const handle = await contract.gpu.ns.createGpuStepper(grid, { backend: 'auto' });
      if (epoch !== gpuEpoch) {
        try { handle.destroy?.(); } catch { /* a newer toggle won */ }
        return;
      }
      if (stepper) {
        try { stepper.destroy(); } catch { /* ignore */ }
      }
      stepper = {
        backend: handle.backend || 'gpu',
        step(n = 1) { return handle.step(n); },
        readField() { return handle.readField(); },
        readProbes() { return handle.readProbes?.() ?? grid.probes; },
        destroy() { try { handle.destroy?.(); } catch { /* ignore */ } },
      };
      gpuOn = true;
      softwareGpu = await softwareRenderer();
      const label = activeBackend();
      message = `backend ${label}`;
      setReadyState(stateEl, 'ready', label);
    } catch (err) {
      if (epoch !== gpuEpoch) return;
      gpuOn = false;
      softwareGpu = false;
      if (grid && contract?.fdtd?.ns) stepper = cpuStepper(contract.fdtd.ns, grid);
      message = err instanceof Error ? err.message : String(err);
      setReadyState(stateEl, 'ready', 'cpu');
    }
    updateTransport();
    updateReadout();
  }

  async function softwareRenderer() {
    try {
      const gpu = globalThis.navigator?.gpu;
      if (gpu?.requestAdapter) {
        const adapter = await gpu.requestAdapter();
        const info = adapter?.info;
        const text = `${info?.vendor || ''} ${info?.architecture || ''} ${info?.description || ''} ${info?.device || ''}`.toLowerCase();
        if (info?.isFallbackAdapter || text.includes('swiftshader') || text.includes('llvmpipe')) return true;
      }
    } catch { /* the CPU label stays if the adapter cannot be named */ }
    try {
      const canvas = doc.createElement('canvas');
      const gl = canvas.getContext('webgl2') || canvas.getContext('webgl');
      const ext = gl?.getExtension?.('WEBGL_debug_renderer_info');
      const renderer = ext ? String(gl.getParameter(ext.UNMASKED_RENDERER_WEBGL) || '') : '';
      const text = renderer.toLowerCase();
      return text.includes('swiftshader') || text.includes('llvmpipe') || text.includes('softpipe');
    } catch {
      return false;
    }
  }

  async function play() {
    if (!solverReady) {
      message = COMING_SOON_TEXT;
      updateReadout();
      return;
    }
    if (running) return;
    if (!stepper) rebuild('time');
    if (!stepper) return;
    running = true;
    const token = generation;
    updateTransport();
    let windowStart = performance.now();
    let windowSteps = 0;
    let solverMs = 0;
    let frames = 0;
    let fpsStart = windowStart;
    while (running && token === generation) {
      const active = stepper;
      try {
        const started = performance.now();
        await active.step(1);
        solverMs += performance.now() - started;
        if (token !== generation) return;
        steps += 1;
        windowSteps += 1;
        lastField = await active.readField();
        if (token !== generation) return;
        emitSamples(await readProbes());
      } catch (err) {
        running = false;
        message = err instanceof Error ? err.message : String(err);
        updateTransport();
        draw();
        return;
      }
      const now = performance.now();
      frames += 1;
      if (now - windowStart >= 500) {
        rate = solverMs > 0 ? (windowSteps / solverMs) * 1000 : 0;
        windowSteps = 0;
        solverMs = 0;
        windowStart = now;
      }
      if (now - fpsStart >= 500) {
        fps = (frames / (now - fpsStart)) * 1000;
        frames = 0;
        fpsStart = now;
      }
      message = 'running';
      draw();
      await new Promise((resolve) => requestAnimationFrame(resolve));
    }
    if (token === generation) running = false;
    updateTransport();
  }

  function pause() {
    running = false;
    recording = false;
    generation += 1;
    message = 'paused';
    updateTransport();
    updateReadout();
  }

  async function recordImpulse() {
    if (!solverReady) {
      message = COMING_SOON_TEXT;
      bus.emit('record-status', { ok: false, message });
      updateReadout();
      return;
    }
    if (!scene.source) {
      message = 'place a transmitter first';
      bus.emit('record-status', { ok: false, message });
      updateReadout();
      return;
    }
    if (scene.probes.length === 0) {
      message = 'place a receiver probe first';
      bus.emit('record-status', { ok: false, message });
      updateReadout();
      return;
    }
    rebuild('record');
    if (!stepper) return;
    recording = true;
    const token = generation;
    const budget = Math.min(4000, scene.nx * 2);
    const batch = 8;
    try {
      for (let done = 0; done < budget && recording && token === generation; done += batch) {
        const n = Math.min(batch, budget - done);
        await stepper.step(n);
        if (token !== generation) return;
        steps += n;
        lastField = await stepper.readField();
        emitSamples(await readProbes());
        message = `recording ${steps}/${budget}`;
        draw();
        await new Promise((resolve) => setTimeout(resolve, 0));
      }
    } catch (err) {
      message = err instanceof Error ? err.message : String(err);
      recording = false;
      bus.emit('record-status', { ok: false, message });
      updateReadout();
      return;
    }
    if (token !== generation) return;
    const probes = await readProbes();
    const list = (Array.isArray(probes) ? probes : Array.from(probes ?? [])).map((probe, index) => ({
      index,
      i: probe?.i ?? scene.probes[index]?.i,
      j: probe?.j ?? scene.probes[index]?.j,
      samples: Float64Array.from(probe?.samples ?? []),
    }));
    recording = false;
    message = `recorded ${steps} steps`;
    bus.emit('impulse', { probes: list });
    updateReadout();
  }

  function eventCell(event) {
    const rect = canvas.getBoundingClientRect();
    return cellFromOffset(event.clientX - rect.left, event.clientY - rect.top, rect.width, rect.height, scene.nx, scene.ny);
  }

  function markCustom() {
    const preset = doc.getElementById('preset');
    if (preset && preset.value !== 'custom') preset.value = 'custom';
  }

  function placeDefaultSource() {
    if (!scene.source) {
      scene.source = {
        i: Math.max(scene.pml + 4, Math.floor(scene.nx * 0.28)),
        j: Math.floor(scene.ny / 2),
        kind: 'sine',
      };
    }
    if (scene.probes.length === 0) {
      const i = Math.min(scene.nx - scene.pml - 4, scene.source.i + 48);
      addProbe(scene, i, scene.source.j);
    }
  }

  function applyPreset(name) {
    const y0 = Math.min(scene.ny - 2, scene.pml + 2);
    const y1 = Math.max(y0, scene.ny - scene.pml - 3);
    scene.materials = [];
    if (name === 'pec') {
      const x = Math.min(scene.nx - scene.pml - 4, Math.floor(scene.nx * 0.7));
      scene.materials.push(materialFromRect({ x0: x, y0, x1: x + 1, y1 }, 'metal', 1));
      message = 'PEC wall';
    } else if (name === 'slab') {
      const x0 = Math.floor(scene.nx * 0.55);
      const x1 = Math.min(scene.nx - scene.pml - 3, Math.floor(scene.nx * 0.72));
      scene.materials.push(materialFromRect({ x0, y0, x1, y1 }, 'dielectric', 4));
      scene.epsR = 4;
      epsInput.value = '4';
      message = 'eps_r=4 slab';
    } else {
      message = 'free space';
    }
    placeDefaultSource();
    const preset = doc.getElementById('preset');
    if (preset) preset.value = name === 'pec' || name === 'slab' ? name : 'free';
    rebuild('time');
  }

  function commitGeometry(nextMessage) {
    message = nextMessage;
    markCustom();
    rebuild('geometry');
  }

  canvas.addEventListener('pointerdown', (event) => {
    if (event.button !== 0) return;
    event.preventDefault();
    canvas.setPointerCapture(event.pointerId);
    const cell = eventCell(event);
    if (!cell) return;
    if (tool === 'tx') {
      scene.source = { i: cell.i, j: cell.j, kind: 'sine' };
      commitGeometry(`transmitter at ${cell.i},${cell.j}`);
      return;
    }
    if (tool === 'rx') {
      const result = addProbe(scene, cell.i, cell.j);
      if (!result.ok) {
        message = result.reason;
        draw();
        return;
      }
      commitGeometry(`probe ${scene.probes.length} at ${cell.i},${cell.j}`);
      return;
    }
    drag = { tool, a: cell, b: cell };
    draw();
  });

  canvas.addEventListener('pointermove', (event) => {
    hover = eventCell(event);
    if (drag) {
      const cell = eventCell(event);
      if (cell) drag.b = cell;
    }
    draw();
  });

  canvas.addEventListener('pointerup', () => {
    if (!drag) return;
    const rect = rectFromCells(drag.a, drag.b);
    const toolName = drag.tool;
    drag = null;
    scene.epsR = Number(epsInput.value);
    scene.materials.push(materialFromRect(rect, toolName, scene.epsR));
    commitGeometry(toolName === 'metal' ? 'metal wall' : `dielectric eps_r ${scene.epsR}`);
  });

  canvas.addEventListener('pointercancel', () => {
    drag = null;
    draw();
  });

  for (const button of toolButtons) {
    button.addEventListener('click', () => selectTool(button.dataset.tool));
  }
  for (const button of modeButtons) {
    button.addEventListener('click', () => selectMode(button.dataset.mode));
  }
  epsInput.addEventListener('change', () => {
    const value = Number(epsInput.value);
    if (!Number.isFinite(value) || value < 1 || value > 80) {
      epsInput.value = String(scene.epsR);
      message = 'eps_r stays between 1 and 80';
      updateReadout();
      return;
    }
    scene.epsR = value;
  });
  must('play').addEventListener('click', () => { play(); });
  must('pause').addEventListener('click', pause);
  must('reset').addEventListener('click', () => {
    message = 'reset';
    rebuild('time');
  });
  must('clear-geometry').addEventListener('click', () => {
    scene.materials = [];
    scene.source = null;
    scene.probes = [];
    message = 'geometry cleared';
    rebuild('geometry');
  });
  must('use-gpu').addEventListener('click', () => { enableGpu(!gpuOn); });
  must('preset').addEventListener('change', () => {
    const name = must('preset').value;
    if (name === 'custom') return;
    applyPreset(name);
  });
  must('freq').addEventListener('change', () => {
    const n = Number(must('freq').value);
    if (!Number.isInteger(n) || n < 8 || n > 64) {
      must('freq').value = String(scene.cellsPerLambda);
      message = 'cells per wavelength must be 8–64';
      updateReadout();
      return;
    }
    scene.cellsPerLambda = n;
    const mhz = 299792458 / n / 1e6;
    message = `${n} cells per wavelength (≈${mhz.toFixed(1)} MHz)`;
    rebuild('time');
  });

  doc.addEventListener('keydown', (event) => {
    const tag = event.target?.tagName;
    if (tag === 'INPUT' || tag === 'SELECT' || tag === 'TEXTAREA') return;
    if (event.code === 'Space') {
      event.preventDefault();
      if (running) pause();
      else play();
    }
  });

  bus.on('record-request', () => { recordImpulse(); });

  const observer = new ResizeObserver(() => draw());
  observer.observe(canvas.parentElement);
  selectTool('metal');
  panel.dataset.mode = scene.mode;
  setReadyState(stateEl, 'checking', 'looking for src/fdtd.js');
  updateTransport();
  draw();
  requestAnimationFrame(() => draw());

  return {
    draw,
    recordImpulse,
    setContract(next) {
      contract = next;
      solverReady = Boolean(next?.fdtd?.ready && next.fdtd.ns);
      if (!solverReady) {
        const detail = panelDetail(next, ['fdtd']) || 'src/fdtd.js is not in main yet';
        setReadyState(stateEl, 'not-ready', detail, COMING_SOON_TEXT);
        message = COMING_SOON_TEXT;
      } else {
        setReadyState(stateEl, 'ready', 'cpu');
        if (!scene.source && scene.materials.length === 0) applyPreset('free');
        else {
          message = 'backend cpu';
          rebuild('time');
        }
      }
      updateTransport();
      draw();
    },
    fail(text) {
      solverReady = false;
      setReadyState(stateEl, 'error', text);
      message = text;
      updateTransport();
      draw();
    },
  };
}
