export const NOT_READY_TEXT = 'module not ready';
export const COMING_SOON_TEXT = 'coming soon';

export function setReadyState(el, state, detail, label) {
  if (!el) return;
  el.dataset.state = state;
  const text = label || (state === 'ready' ? 'ready' : state === 'error' ? 'module error' : NOT_READY_TEXT);
  el.replaceChildren();
  const strong = document.createElement('strong');
  strong.textContent = text;
  el.append(strong);
  if (detail) {
    const span = document.createElement('span');
    span.textContent = ` — ${detail}`;
    el.append(span);
  }
  el.title = el.textContent;
}

export function renderParamNotes(el, notes) {
  if (!el) return;
  el.textContent = notes.length ? notes.join('; ') : '';
}

export function reportLinkError(el, message) {
  if (!el) return;
  el.textContent = message;
}
