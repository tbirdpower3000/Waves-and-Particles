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

const BIT_COUNT = 8192;
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
    identity.textContent = 'Perfect direct path';
    identity.title = 'No echoes or loss. The signal arrives unchanged before noise is added.';
    channelEl.append(identity);
    recordings.forEach((row, index) => {
      if (!row?.samples?.length) return;
      const option = doc.createElement('option');
      option.value = String(index);
      option.textContent = `Probe ${index + 1} channel`;
      option.title = `Uses probe ${index + 1}'s recorded pulse response, with ${row.samples.length} samples, as the echo pattern the signal passes through.`;
      channelEl.append(option);
    });
    const values = [...channelEl.options].map((option) => option.value);
    channelEl.value = values.includes(previous) ? previous : 'identity';
  }

  function draw() {
    const missing = !ready();
    const scheme = schemeEl.value;
    drawConstellation(constellation, points, {
      trajectory: trajectoryOn,
      idealRadius: scheme === 'qpsk' ? Math.SQRT2 : 1,
      circleLabel: scheme === 'qpsk' ? '|s| = √2' : '|s| = 1',
      empty: missing ? 'module not ready' : 'no samples',
      warn: missing,
    });
    if (scheme === 'chirp') {
      drawBer(berPlot, { note: 'No chirp error curve', warn: false });
    } else {
      const probe = channelEl.value !== 'identity';
      drawBer(berPlot, {
        theory,
        simulated,
        markerDb: currentDb(),
        legend: 'Theory: AWGN, no multipath',
        note: probe ? 'Probe channel: the line ignores echoes.' : '',
      });
    }
    drawSpectrum(spectrum, spectrumValues, {
      empty: missing ? 'module not ready' : 'no spectrum',
      warn: missing,
      unit: scheme === 'chirp' ? 'Hz' : 'cycles/sample',
      fs: scheme === 'chirp' ? CHIRP.fs : 1,
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
      berReadout.textContent = 'Bit error rate —';
      return;
    }
    const { result } = outcome;
    if (outcome.chirp) {
      trajectoryOn = true;
      points = trajectory(result.noisy.i, result.noisy.q, result.offset, result.span);
      spectrumValues = spectrumDb(contract.signal.ns.fftMag(windowed(result.noisy, result.offset, result.span)));
      simulated = [];
      berReadout.textContent = 'Bit error rate —';
      note.textContent = `chirp offset ${result.offset}, pRef ${result.pRef.toExponential(2)}. No demodulator for chirp.`;
      return;
    }
    trajectoryOn = false;
    points = symbolPoints(result.noisy.i, result.noisy.q, SPS, result.offset, result.nSymbols);
    spectrumValues = spectrumDb(contract.signal.ns.fftMag(windowed(result.noisy, result.offset, result.span)));
    if (!keepSimulated) simulated = [{ ebn0dB: db, ber: result.errorRate }];
    const shown = formatBer(result.errorRate, result.txBits.length);
    berReadout.textContent = `Bit error rate ${shown}`;
    note.textContent = `BER ${shown} at ${db.toFixed(1)} dB, ${result.txBits.length} bits, offset ${result.offset}, pRef ${result.pRef.toExponential(2)}. Preview only; the 1e6-bit check is tests/signal.test.js.`;
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
      reportLinkError(note, 'No chirp error curve');
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
          berReadout.textContent = 'Bit error rate —';
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

  function syncNoiseLabel() {
    const label = doc.getElementById('ebn0-label');
    const unit = doc.getElementById('ebn0-unit');
    const caption = doc.getElementById('ebn0-caption');
    const chirp = schemeEl.value === 'chirp';
    if (label) label.textContent = chirp ? 'SNR (per sample), dB' : 'Signal-to-noise per bit';
    if (unit) unit.textContent = chirp ? '' : 'dB';
    const tip = chirp
      ? 'Per-sample signal power divided by noise density. The chirp has no bits, so this is not Eb/N0.'
      : 'Received energy per bit divided by noise power density. Higher means cleaner. Each +3 dB doubles the ratio.';
    if (caption) caption.title = tip;
    ebn0El.title = tip;
  }

  channelEl.addEventListener('change', () => draw());
  schemeEl.addEventListener('change', () => {
    simulated = [];
    points = [];
    spectrumValues = new Float64Array(0);
    syncNoiseLabel();
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
  syncNoiseLabel();
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
        setReadyState(stateEl, 'ready', 'main tap, region power');
        note.textContent = 'Perfect direct path until a probe pulse response is frozen.';
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
