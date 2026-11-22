export const GRID_MIN = 32;
export const GRID_MAX = 512;

export const LIMITS = {
  nx: { min: GRID_MIN, max: GRID_MAX, integer: true, fallback: 512 },
  ny: { min: GRID_MIN, max: GRID_MAX, integer: true, fallback: 512 },
  pml: { min: 0, max: 64, integer: true, fallback: 16 },
  seed: { min: 0, max: 0xffffffff, integer: true, fallback: 20261121 },
  ebn0: { min: -2, max: 12, integer: false, fallback: 6 },
  eps: { min: 1, max: 80, integer: false, fallback: 4 },
};

function clipRaw(raw) {
  const text = String(raw);
  return text.length > 40 ? `${text.slice(0, 40)}...` : text;
}

export function readBounded(raw, spec) {
  if (raw == null || raw === '') {
    return { ok: true, value: spec.fallback, fallback: true };
  }
  const n = typeof raw === 'number' ? raw : Number(String(raw).trim());
  if (!Number.isFinite(n)) return { ok: false, value: spec.fallback, fallback: true };
  if (spec.integer && !Number.isInteger(n)) return { ok: false, value: spec.fallback, fallback: true };
  if (n < spec.min || n > spec.max) return { ok: false, value: spec.fallback, fallback: true };
  return { ok: true, value: n, fallback: false };
}

// Grid dimensions size the field arrays and any WebGPU buffer. Reject a
// number before that allocation, including values that arrived from the URL.
export function assertGridBudget(nx, ny) {
  if (!Number.isInteger(nx) || !Number.isInteger(ny) || !Number.isFinite(nx) || !Number.isFinite(ny)) {
    throw new Error('grid size must be finite integers');
  }
  if (nx < GRID_MIN || ny < GRID_MIN || nx > GRID_MAX || ny > GRID_MAX) {
    throw new Error(`grid size ${nx}x${ny} is outside ${GRID_MIN}..${GRID_MAX}`);
  }
  const cells = nx * ny;
  if (!Number.isFinite(cells) || cells > GRID_MAX * GRID_MAX) {
    throw new Error('grid buffer exceeds the lab cap');
  }
  return cells;
}

export function gpuBufferGate(nx, ny) {
  return assertGridBudget(nx, ny);
}

export function readLabParams(search) {
  const query = search.startsWith('?') || search === '' ? search : `?${search}`;
  const params = new URLSearchParams(query);
  const notes = [];
  const out = {};
  for (const key of Object.keys(LIMITS)) {
    const raw = params.get(key);
    const parsed = readBounded(raw, LIMITS[key]);
    out[key] = parsed.value;
    if (!parsed.ok) notes.push(`ignored ${key} (${clipRaw(raw)})`);
  }
  const maxPml = Math.max(0, Math.floor((Math.min(out.nx, out.ny) - 1) / 2));
  if (out.pml > maxPml) {
    notes.push(`ignored pml (${out.pml})`);
    out.pml = Math.min(LIMITS.pml.fallback, maxPml);
  }
  out.notes = notes;
  return out;
}
