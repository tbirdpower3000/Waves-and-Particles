// Fetch a module before import(). A missing file then stays a 404 from fetch,
// which the page handles, instead of a failed module load.

export const MODULES = [
  {
    id: 'fdtd',
    label: 'src/fdtd.js',
    optional: false,
    // A missing file logs a console error in Chromium. Do not request it
    // until the field solver is in the tree. Pass request: true to probe.
    request: false,
    candidates: ['../fdtd.js'],
    exports: {
      C0: 'number',
      courantDt: 'function',
      createGrid: 'function',
      setMaterial: 'function',
      addSource: 'function',
      addProbe: 'function',
      step: 'function',
      snapshot: 'function',
    },
  },
  {
    id: 'gpu',
    label: 'src/gpu/',
    optional: true,
    request: false,
    candidates: ['../gpu/index.js', '../gpu/gpu.js', '../gpu/stepper.js', '../gpu/webgpu.js', '../gpu/backend.js'],
    exports: {
      createGpuStepper: 'function',
    },
  },
  {
    id: 'signal',
    label: 'src/signal.js',
    optional: false,
    candidates: ['../signal.js'],
    exports: {
      bitsRandom: 'function',
      modulate: 'function',
      chirp: 'function',
      convolveChannel: 'function',
      awgn: 'function',
      demodulate: 'function',
      ber: 'function',
      berTheory: 'function',
      fftMag: 'function',
    },
  },
  {
    id: 'rng',
    label: 'src/rng.js',
    optional: false,
    candidates: ['../rng.js'],
    exports: {
      makeRng: 'function',
    },
  },
];

export function missingExports(ns, exportsSpec) {
  const missing = [];
  for (const [name, type] of Object.entries(exportsSpec)) {
    if (typeof ns?.[name] !== type) missing.push(name);
  }
  return missing;
}

function isJavaScript(response, body) {
  const type = String(response?.headers?.get?.('content-type') || '').toLowerCase();
  if (type.includes('text/html') || type.includes('application/json')) return false;
  const trimmed = String(body ?? '').trim();
  if (!trimmed || trimmed.startsWith('<')) return false;
  return true;
}

function blank(spec, state, extra = {}) {
  return {
    id: spec.id,
    label: spec.label,
    optional: spec.optional,
    ready: false,
    state,
    url: extra.url ?? null,
    missing: extra.missing ?? [],
    error: extra.error ?? null,
    ns: extra.ns ?? null,
  };
}

async function tryCandidate(spec, rel, deps) {
  const url = new URL(rel, deps.base);
  const href = url.href;
  let response;
  try {
    response = await deps.fetchImpl(href, {
      cache: 'no-store',
      signal: AbortSignal.timeout(8000),
    });
  } catch (err) {
    return { state: 'missing', url: href, error: err instanceof Error ? err.message : String(err) };
  }
  if (!response?.ok) return { state: 'missing', url: href };
  let body = '';
  try {
    body = await response.text();
  } catch {
    body = '';
  }
  if (!isJavaScript(response, body)) return { state: 'missing', url: href };
  try {
    const ns = await deps.importImpl(href);
    const missing = missingExports(ns, spec.exports);
    if (missing.length) return { state: 'incomplete', url: href, missing, ns };
    return { state: 'ready', url: href, missing: [], ns };
  } catch (err) {
    return {
      state: 'error',
      url: href,
      error: err instanceof Error ? err.message : String(err),
      missing: [],
    };
  }
}

async function loadOne(spec, deps) {
  if (spec.request === false) return blank(spec, 'missing');
  let fallback = null;
  for (const rel of spec.candidates) {
    const result = await tryCandidate(spec, rel, deps);
    if (result.state === 'ready') {
      return {
        id: spec.id,
        label: spec.label,
        optional: spec.optional,
        ready: true,
        state: 'ready',
        url: result.url,
        missing: [],
        error: null,
        ns: result.ns,
      };
    }
    if (result.state === 'incomplete' || result.state === 'error') fallback = result;
  }
  if (fallback) return blank(spec, fallback.state, fallback);
  return blank(spec, 'missing');
}

export function panelDetail(contract, ids) {
  const parts = [];
  for (const id of ids) {
    const mod = contract?.[id];
    if (!mod || mod.ready) continue;
    if (mod.state === 'incomplete') {
      parts.push(`${mod.label} is missing exports: ${mod.missing.join(', ')}`);
    } else if (mod.state === 'error') {
      parts.push(`${mod.label} failed to load${mod.error ? ` (${mod.error})` : ''}`);
    } else {
      parts.push(`${mod.label} is not available yet`);
    }
  }
  return parts.join('; ');
}

export async function loadContract(options = {}) {
  const deps = {
    fetchImpl: options.fetchImpl ?? globalThis.fetch.bind(globalThis),
    importImpl: options.importImpl ?? ((href) => import(href)),
    base: options.base ?? import.meta.url,
  };
  const specs = options.modules ?? MODULES;
  const loaded = await Promise.all(specs.map((spec) => loadOne(spec, deps)));
  const contract = {};
  for (const mod of loaded) contract[mod.id] = mod;
  return contract;
}
