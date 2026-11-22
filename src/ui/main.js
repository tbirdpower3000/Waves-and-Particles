import { createBus } from './bus.js';
import { mountChannel } from './channel-panel.js';
import { mountField } from './field-panel.js';
import { loadContract } from './load-contract.js';
import { readLabParams } from './params.js';
import { mountSignal } from './signal-panel.js';
import { renderParamNotes, setReadyState } from './status.js';

function boot() {
  const params = readLabParams(window.location.search);
  renderParamNotes(document.getElementById('param-notes'), params.notes);
  const bus = createBus();
  let field;
  let channel;
  let signal;
  try {
    field = mountField(document, bus, params);
    channel = mountChannel(document, bus);
    signal = mountSignal(document, bus, params);
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    setReadyState(document.getElementById('field-state'), 'error', message);
    return;
  }
  field.draw();
  channel.draw();
  signal.draw();
  loadContract()
    .then((contract) => {
      field.setContract(contract);
      channel.setContract(contract);
      signal.setContract(contract);
    })
    .catch((err) => {
      const message = err instanceof Error ? err.message : String(err);
      field.fail(message);
      channel.fail(message);
      signal.fail(message);
    });
}

if (typeof document !== 'undefined') boot();
