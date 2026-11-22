import { drawSeries } from './draw.js';
import { panelDetail } from './load-contract.js';
import { PROBE_COLORS } from './scene.js';
import { COMING_SOON_TEXT, setReadyState } from './status.js';

function must(id) {
  const el = document.getElementById(id);
  if (!el) throw new Error(`missing #${id}`);
  return el;
}

export function mountChannel(doc, bus) {
  const stateEl = must('channel-state');
  const note = must('channel-note');
  const button = must('record-ir');
  const canvases = [0, 1, 2, 3].map((index) => must(`channel-plot-${index}`));
  let ready = false;
  let live = [];
  let frozen = [];

  function traces() {
    return frozen.length ? frozen : live;
  }

  function draw() {
    const rows = traces();
    canvases.forEach((canvas, index) => {
      const row = rows[index];
      const label = must(`channel-label-${index}`);
      const count = row?.samples?.length ?? 0;
      const kind = frozen.length ? 'frozen' : 'live';
      label.textContent = row ? `Probe ${index + 1} trace · ${kind} ${count}` : `Probe ${index + 1} trace`;
      label.style.color = PROBE_COLORS[index];
      drawSeries(canvas, row?.samples ?? [], {
        color: PROBE_COLORS[index],
        empty: ready ? 'no samples' : COMING_SOON_TEXT,
        warn: !ready,
      });
    });
  }

  button.addEventListener('click', () => bus.emit('record-request'));
  bus.on('samples', (payload) => {
    live = payload?.probes ?? [];
    draw();
  });
  bus.on('impulse', (payload) => {
    frozen = payload?.probes ?? [];
    note.textContent = frozen.length
      ? `frozen pulse response, ${frozen.length} probe${frozen.length === 1 ? '' : 's'}`
      : '';
    draw();
  });
  bus.on('scene', () => {
    live = [];
    frozen = [];
    note.textContent = '';
    bus.emit('impulse', { probes: [] });
    draw();
  });
  bus.on('record-status', (payload) => {
    note.textContent = payload?.message || '';
  });

  const observer = new ResizeObserver(() => draw());
  for (const canvas of canvases) observer.observe(canvas.parentElement);
  setReadyState(stateEl, 'checking', 'waiting for the field solver');
  button.disabled = true;
  draw();

  return {
    draw,
    setContract(contract) {
      ready = Boolean(contract?.fdtd?.ready);
      button.disabled = !ready;
      if (!ready) {
        setReadyState(stateEl, 'not-ready', panelDetail(contract, ['fdtd']) || 'src/fdtd.js is not in main yet', COMING_SOON_TEXT);
      } else {
        setReadyState(stateEl, 'ready', 'probe trace');
      }
      draw();
    },
    fail(text) {
      ready = false;
      button.disabled = true;
      setReadyState(stateEl, 'error', text);
      draw();
    },
  };
}
