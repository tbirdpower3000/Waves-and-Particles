import assert from 'node:assert/strict';
import { test } from 'node:test';
import { makeRng, sfc32, splitmix32 } from '../src/rng.js';

// Brief vectors (Node/V8). uniform53(sfc32(42)) and gauss(42).
const UNIFORM_42 = [
  0.57672138908419068,
  0.21293001466432809,
  0.23419718284633800,
  0.83992477684647149,
];
const GAUSS_42 = [
  0.24217036478852957,
  1.0208552762307241,
  0.91230083296059428,
  -1.4390590341036926,
  -0.91551175264863893,
  -1.1189243734805443,
];

// sfc32(12345) first 8 uints after splitmix32 seeding and 12 warm-up draws.
const SFC_12345 = [
  1977310364, 4106765030, 1538400628, 2967644150,
  3285158194, 102214278, 3222376572, 3029861462,
];

function uniformFromWords(hiWord, loWord) {
  const hi = hiWord >>> 5;
  const lo = loWord >>> 6;
  return (hi * 67108864 + lo + 0.5) / 9007199254740992;
}

test('uniform53(sfc32(42)) matches the brief', () => {
  const rng = makeRng(42);
  for (let i = 0; i < UNIFORM_42.length; i++) {
    assert.equal(rng.uniform(), UNIFORM_42[i]);
  }
});

test('gauss(42) matches the brief', () => {
  const rng = makeRng(42);
  for (let i = 0; i < GAUSS_42.length; i++) {
    assert.equal(rng.normal(), GAUSS_42[i]);
  }
});

test('splitmix32(1) matches the brief', () => {
  const next = splitmix32(1);
  assert.deepEqual(
    [next(), next(), next(), next()],
    [2527132011, 314344336, 2535364964, 2041432039],
  );
});

test('sfc32(12345) raw words match the brief', () => {
  const next = sfc32(12345);
  assert.deepEqual(
    [next(), next(), next(), next(), next(), next(), next(), next()],
    SFC_12345,
  );
});

test('sfc32(12345) uint stream is the uniform source', () => {
  const rng = makeRng(12345);
  for (let i = 0; i < 4; i++) {
    const expected = uniformFromWords(SFC_12345[2 * i], SFC_12345[2 * i + 1]);
    assert.equal(rng.uniform(), expected);
  }
});

test('same seed replays uniform and normal, including the Box-Muller spare', () => {
  for (const seed of [0, 1, 42, 20261121]) {
    const a = makeRng(seed);
    const b = makeRng(seed);
    for (let i = 0; i < 32; i++) {
      assert.equal(a.uniform(), b.uniform());
      assert.equal(a.normal(), b.normal());
    }
  }
  assert.notEqual(makeRng(1).uniform(), makeRng(2).uniform());
});

test('uniform stays inside (0, 1)', () => {
  const rng = makeRng(99);
  for (let i = 0; i < 10000; i++) {
    const u = rng.uniform();
    assert.ok(u > 0 && u < 1);
  }
});

test('normal() has mean ~0 and variance ~1', () => {
  const rng = makeRng(7);
  const n = 100_000;
  let sum = 0;
  let sum2 = 0;
  for (let i = 0; i < n; i++) {
    const z = rng.normal();
    sum += z;
    sum2 += z * z;
  }
  const mean = sum / n;
  const variance = sum2 / n - mean * mean;
  assert.ok(Math.abs(mean) < 0.01, `mean ${mean}`);
  assert.ok(Math.abs(variance - 1) < 0.02, `var ${variance}`);
});
