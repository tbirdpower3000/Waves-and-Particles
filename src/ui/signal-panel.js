import { drawBer, drawConstellation, drawSpectrum } from './draw.js';
import {
  formatBer,
  runLinkSafely,
  spectrumDb,
  symbolPoints,
  trajectory,
} from './link.js';
import { panelDetail } from './load-contract.js';
import { reportLinkError, setReadyState } from './status.js';

const BIT_COUNT = 1024;
const SPS = 8;
const CHIRP = { f0: 200, f1: 3000, T: 0.04, fs: 8000 };

function must(id) {
  const el = document.getElementById(id);
  if (!el) throw new Error(`missing #${id}`);
  return el;
}

export function mountSignal(doc, bus, params = {}) {
  const stateEl = must('signal-state');
  const note = must('signal-note');
  const schemeEl = must('scheme');
  const seedEl = must('seed');
  const ebn0El = must('ebn0');
  const ebn0Out = must('ebn0-out');
  const channelEl = must('channel-source');
  const runButton = must('run-link');
  const sweepButton = must('sweep-ber');
  const berReadout = must('ber-readout');
  const constellation = must('constellation');
  const berPlot = must('ber-plot');
  const spectrum = must('spectrum');

  seedEl.value = String(params.seed ?? 20261121);
  ebn0El.value = String(params.ebn0 ?? 6);
  ebn0Out.textContent = Number(ebn0El.value).toFixed(1);

  let contract = null;
  let recordings = [];
  let theory = [];
  let simulated = [];
  let points = [];
  let trajectoryOn = false;
  let spectrumValues = new Float64Array(0);
  let runGen = 0;

  function ready() {
    return Boolean(contract?.signal?.ready && contract?.rng?.ready);
  }

  function setDisabled(disabled) {
    for (const el of [schemeEl, seedEl, ebn0El, channelEl, runButton, sweepButton]) {
      el.disabled = disabled;
    }
  }

  function currentSeed() {
    const n = Number(seedEl.value);
    return Number.isFinite(n) ? Math.trunc(n) : 1;
  }

  function currentDb() {
    const n = Number(ebn0El.value);
    return Number.isFinite(n) ? n : 6;
  }

  function selectedH() {
    if (channelEl.value === 'identity') return new Float64Array([1]);
    const row = recordings[Number(channelEl.value)];
    if (!row?.samples?.length) return new Float64Array([1]);
    return row.samples;
  }

  function fillChannels() {
    const previous = channelEl.value;
    channelEl.replaceChildren();
    const identity = doc.createElement('option');
    identity.value = 'identity';
    identity.textContent = 'Identity';
    channelEl.append(identity);
    recordings.forEach((row, index) => {
      if (!row?.samples?.length) return;
      const option = doc.createElement('option');
      option.value = String(index);
      option.textContent = `Probe ${index + 1} (${row.samples.length})`;
      channelEl.append(option);
    });
    const values = [...channelEl.options].map((option) => option.value);
    channelEl.value = values.includes(previous) ? previous : 'identity';
  }

  function draw() {
    const missing = !ready();
    drawConstellation(constellation, points, {
      trajectory: trajectoryOn,
      empty: missing ? 'module not ready' : 'no samples',
      warn: missing,
    });
    const scheme = schemeEl.value;
    if (scheme === 'chirp') {
      drawBer(berPlot, { note: 'BER theory is defined for BPSK and QPSK', warn: false });
    } else {
      drawBer(berPlot, { theory, simulated, markerDb: currentDb() });
    }
    drawSpectrum(spectrum, spectrumValues, {
      empty: missing ? 'module not ready' : 'no spectrum',
      warn: missing,
    });
  }

  function drawTheory() {
    theory = [];
    if (!ready() || schemeEl.value === 'chirp') {
      draw();
      return;
    }
    try {
      const fn = contract.signal.ns.berTheory;
      for (let db = -2; db <= 12; db += 0.5) {
        theory.push({ ebn0dB: db, ber: fn(schemeEl.value, db) });
      }
    } catch (err) {
      reportLinkError(note, err instanceof Error ? err.message : String(err));
    }
    draw();
  }

  function applyResult(outcome, db, keepSimulated) {
    if (!outcome.ok) {
      reportLinkError(note, outcome.message);
      berReadout.textContent = 'BER error';
      return;
    }
    const { result } = outcome;
    if (outcome.chirp) {
      trajectoryOn = true;
      points = trajectory(result.noisy.i, result.noisy.q, result.offset, result.span);
      spectrumValues = spectrumDb(contract.signal.ns.fftMag(windowed(result.noisy, result.offset, result.span)));
      simulated = [];
      berReadout.textContent = 'BER —';
      note.textContent = `chirp offset ${result.offset}, pRef ${result.pRef.toExponential(2)}. No demodulator for chirp.`;
      return;
    }
    trajectoryOn = false;
    points = symbolPoints(result.noisy.i, result.noisy.q, SPS, result.offset, result.nSymbols);
    spectrumValues = spectrumDb(contract.signal.ns.fftMag(windowed(result.noisy, result.offset, result.span)));
    if (!keepSimulated) simulated = [{ ebn0dB: db, ber: result.errorRate }];
    berReadout.textContent = `BER ${formatBer(result.errorRate)}`;
    note.textContent = `BER ${formatBer(result.errorRate)} at ${db.toFixed(1)} dB, ${result.txBits.length} bits, offset ${result.offset}, pRef ${result.pRef.toExponential(2)}. 1024-bit preview; the 1e6-bit check is tests/signal.test.js.`;
  }

  function windowed(sig, offset, span) {
    const maxLen = 16384;
    const start = Math.max(0, Math.min(offset, sig.i.length));
    const len = Math.max(0, Math.min(span, maxLen, sig.i.length - start));
    return { i: sig.i.subarray(start, start + len), q: sig.q.subarray(start, start + len) };
  }

  async function runOnce(db, rngOffset) {
    const scheme = schemeEl.value;
    const rng = contract.rng.ns.makeRng(currentSeed() + rngOffset);
    const h = selectedH();
    if (scheme === 'chirp') {
      return runLinkSafely(contract.signal.ns, {
        scheme,
        h,
        ebn0dB: db,
        rng,
        ...CHIRP,
      });
    }
    const bits = contract.signal.ns.bitsRandom(BIT_COUNT, rng);
    return runLinkSafely(contract.signal.ns, {
      bits,
      scheme,
      sps: SPS,
      h,
      ebn0dB: db,
      rng,
    });
  }

  async function run({ sweep }) {
    if (!ready()) {
      reportLinkError(note, 'module not ready');
      return;
    }
    const gen = ++runGen;
    const scheme = schemeEl.value;
    if (sweep && scheme === 'chirp') {
      reportLinkError(note, 'BER theory is defined for BPSK and QPSK');
      return;
    }
    note.textContent = 'running';
    const levels = sweep ? [0, 2, 4, 6, 8, 10] : [currentDb()];
    const pointsSoFar = [];
    try {
      for (const db of levels) {
        if (gen !== runGen) return;
        const outcome = await runOnce(db, sweep ? Math.round(db * 2) : 0);
        if (gen !== runGen) return;
        if (!outcome.ok) {
          reportLinkError(note, outcome.message);
          berReadout.textContent = 'BER error';
          draw();
          return;
        }
        if (!outcome.chirp) pointsSoFar.push({ ebn0dB: db, ber: outcome.result.errorRate });
        simulated = pointsSoFar;
        applyResult(outcome, db, sweep);
        draw();
        await new Promise((resolve) => setTimeout(resolve, 0));
      }
    } catch (err) {
      reportLinkError(note, err instanceof Error ? err.message : String(err));
    }
  }

  schemeEl.addEventListener('change', () => {
    simulated = [];
    points = [];
    spectrumValues = new Float64Array(0);
    drawTheory();
  });
  ebn0El.addEventListener('input', () => {
    ebn0Out.textContent = Number(ebn0El.value).toFixed(1);
    draw();
  });
  runButton.addEventListener('click', () => { run({ sweep: false }); });
  sweepButton.addEventListener('click', () => { run({ sweep: true }); });
  bus.on('impulse', (payload) => {
    recordings = payload?.probes ?? [];
    fillChannels();
  });

  const observer = new ResizeObserver(() => draw());
  observer.observe(constellation.parentElement);
  observer.observe(berPlot.parentElement);
  observer.observe(spectrum.parentElement);
  setReadyState(stateEl, 'checking', 'looking for src/signal.js and src/rng.js');
  setDisabled(true);
  fillChannels();
  draw();

  return {
    draw,
    setContract(next) {
      contract = next;
      const ok = ready();
      setDisabled(!ok);
      if (!ok) {
        setReadyState(stateEl, 'not-ready', panelDetail(next, ['signal', 'rng']) || 'src/signal.js is not available yet');
        note.textContent = '';
      } else {
        setReadyState(stateEl, 'ready', 'cursor on the main tap, noise from that region');
        note.textContent = 'Identity channel until a probe impulse is frozen.';
        drawTheory();
      }
      draw();
    },
    fail(text) {
      contract = null;
      setDisabled(true);
      setReadyState(stateEl, 'error', text);
      reportLinkError(note, text);
      draw();
    },
  };
}
