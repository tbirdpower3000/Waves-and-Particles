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

function fmt(value) {
  if (!Number.isFinite(value)) return '—';
  if (value !== 0 && (Math.abs(value) < 1e-2 || Math.abs(value) >= 1e4)) return value.toExponential(3);
  return String(Math.round(value * 1000) / 1000);
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
  }

  function updateReadout() {
    const gpuText = !contract?.gpu?.ready
      ? 'GPU module not ready'
      : gpuOn ? `GPU ${stepper?.backend || 'on'}` : 'GPU idle';
    const threads = globalThis.navigator?.hardwareConcurrency ?? '?';
    const cell = hover ? `cell ${hover.i},${hover.j}` : '';
    readout.textContent = [
      message,
      `${scene.nx}x${scene.ny}`,
      scene.mode,
      `dt ${fmt(grid?.dt)}`,
      `step ${steps}`,
      `${rate.toFixed(0)} step/s`,
      stepper?.backend || 'cpu',
      gpuText,
      `threads ${threads}`,
      cell,
    ].filter(Boolean).join(' · ');
  }

  function draw() {
    frame += 1;
    canvas.dataset.frame = String(frame);
    canvas.dataset.materials = String(scene.materials.length);
    canvas.dataset.probes = String(scene.probes.length);
    canvas.dataset.source = scene.source ? '1' : '0';
    panel.dataset.mode = scene.mode;
    const warning = drawField(canvas, {
      field: lastField,
      nx: scene.nx,
      ny: scene.ny,
      pml: scene.pml,
      scene,
      drag: drag ? { tool: drag.tool, rect: rectFromCells(drag.a, drag.b) } : null,
      solverReady,
      holdLabel: solverReady ? '' : COMING_SOON_TEXT,
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

  function rebuild(reason) {
    stopLoop();
    if (stepper) {
      try { stepper.destroy(); } catch { /* ignore a dead stepper */ }
    }
    stepper = null;
    gpuOn = false;
    grid = null;
    lastField = null;
    steps = 0;
    rate = 0;
    if (!solverReady) {
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
      message = reason === 'record' ? 'recording impulse' : 'grid rebuilt';
    } catch (err) {
      message = err instanceof Error ? err.message : String(err);
      setReadyState(stateEl, 'error', message);
    }
    bus.emit('scene', { reset: true });
    draw();
    updateTransport();
  }

  async function enableGpu(on) {
    if (!on) {
      if (solverReady && grid) stepper = cpuStepper(contract.fdtd.ns, grid);
      gpuOn = false;
      message = 'CPU reference';
      updateTransport();
      updateReadout();
      return;
    }
    try {
      const cells = gpuBufferGate(scene.nx, scene.ny);
      if (!contract?.gpu?.ready || !grid || cells < 1) return;
      const handle = await contract.gpu.ns.createGpuStepper(grid, { backend: 'auto' });
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
      message = `GPU ${stepper.backend}`;
    } catch (err) {
      gpuOn = false;
      if (grid && contract?.fdtd?.ns) stepper = cpuStepper(contract.fdtd.ns, grid);
      message = err instanceof Error ? err.message : String(err);
    }
    updateTransport();
    updateReadout();
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
    while (running && token === generation) {
      const active = stepper;
      try {
        await active.step(1);
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
      if (now - windowStart >= 500) {
        rate = (windowSteps / (now - windowStart)) * 1000;
        windowSteps = 0;
        windowStart = now;
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

  function commitGeometry(nextMessage) {
    message = nextMessage;
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
        const gpuDetail = next.gpu?.ready
          ? 'CPU reference, GPU available'
          : 'CPU reference. GPU module not ready';
        setReadyState(stateEl, 'ready', gpuDetail);
        message = 'CPU reference';
        rebuild('time');
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
