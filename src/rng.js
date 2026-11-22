/**
 * Seeded PRNG shared by Node and the browser. No runtime dependencies.
 *
 * Pipeline (integer core is bit-exact via `| 0`, `>>>` and `Math.imul`):
 *   splitmix32(seed) fills sfc32 state (a, b, c), d starts at 1,
 *   12 warm-up draws are discarded,
 *   uniform() is a 53-bit value in (0, 1): (hi27 * 2^26 + lo26 + 0.5) / 2^53,
 *   normal() is Box–Muller, cosine returned first and sine cached.
 *
 * Locked to the WAP-001 brief vectors. `Math.log` / `sin` / `cos` are not
 * bit-exact across engines; the gauss vector matches V8.
 * splitmix32(1) first 4 uints: 2527132011, 314344336, 2535364964, 2041432039.
 */

const WARMUP = 12;

// Murmur-style splitmix32. Returns uint32s in [0, 2^32).
function splitmix32(seed) {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x9e3779b9) >>> 0;
    let z = state;
    z = Math.imul(z ^ (z >>> 16), 0x85ebca6b);
    z = Math.imul(z ^ (z >>> 13), 0xc2b2ae35);
    return (z ^ (z >>> 16)) >>> 0;
  };
}

// sfc32 seeded from `seed`. The returned function yields uint32s after warm-up.
function sfc32(seed) {
  const mix = splitmix32(seed);
  let a = mix() | 0;
  let b = mix() | 0;
  let c = mix() | 0;
  let d = 1;
  const next = () => {
    const t = (a + b | 0) + d | 0;
    d = d + 1 | 0;
    a = b ^ (b >>> 9);
    b = c + (c << 3) | 0;
    c = (c << 21) | (c >>> 11);
    c = c + t | 0;
    return t >>> 0;
  };
  for (let i = 0; i < WARMUP; i++) next();
  return next;
}

/**
 * @param {number} seed Any number. Reduced with `>>> 0` to a uint32.
 * @returns {{ uniform: () => number, normal: () => number }}
 *   uniform in (0, 1), never 0 or 1. normal is standard normal (mean 0, var 1).
 */
export function makeRng(seed) {
  const nextUint = sfc32(seed);
  let spare = null;

  function uniform() {
    // 27 high bits, then 26 high bits. The +1/2 offset keeps the result in (0, 1).
    const hi = nextUint() >>> 5;
    const lo = nextUint() >>> 6;
    return (hi * 67108864 + lo + 0.5) / 9007199254740992;
  }

  function normal() {
    if (spare !== null) {
      const z = spare;
      spare = null;
      return z;
    }
    const u1 = uniform();
    const u2 = uniform();
    const mag = Math.sqrt(-2 * Math.log(u1));
    spare = mag * Math.sin(2 * Math.PI * u2);
    return mag * Math.cos(2 * Math.PI * u2);
  }

  return { uniform, normal };
}
