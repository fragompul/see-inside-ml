import { createRng } from '@sim/core';
import { describe, expect, it } from 'vitest';
import { standardise, trainProbe } from './probe';

const blobs = (n: number, seed: number, gap: number) => {
  const rng = createRng(seed);
  const x = new Float64Array(n * 3);
  const y: number[] = [];
  for (let i = 0; i < n; i++) {
    const c = i % 3;
    y.push(c);
    x[3 * i] = (c === 0 ? -gap : c === 1 ? gap : 0) + rng.normal();
    x[3 * i + 1] = (c === 2 ? gap : 0) + rng.normal();
    x[3 * i + 2] = rng.normal();
  }
  return { x, y };
};

describe('linear probe', () => {
  it('reads a linearly separable property almost perfectly and an unrelated one at chance', () => {
    const train = blobs(300, 1, 4);
    const test = blobs(150, 2, 4);
    const good = trainProbe(train.x, train.y, test.x, test.y, 3, 3);
    expect(good.testAccuracy).toBeGreaterThan(0.95);
    const shuffler = createRng(11);
    const shuffled = train.y.map(() => Math.floor(shuffler.next() * 3));
    const control = trainProbe(train.x, shuffled, test.x, test.y, 3, 3);
    expect(control.testAccuracy).toBeLessThan(0.6);
  });

  it('standardising gives zero mean and unit variance on the training rows', () => {
    const { x } = blobs(90, 3, 2);
    const { train } = standardise(x, x, 3);
    for (let j = 0; j < 3; j++) {
      let m = 0;
      let v = 0;
      for (let i = 0; i < 90; i++) m += train[3 * i + j]! / 90;
      for (let i = 0; i < 90; i++) v += (train[3 * i + j]! - m) ** 2 / 90;
      expect(m).toBeCloseTo(0, 9);
      expect(v).toBeCloseTo(1, 4);
    }
  });

  it('with more features than examples a probe can memorise random labels but not generalise them', () => {
    const rng = createRng(5);
    const dim = 60;
    const n = 40;
    const make = (count: number) => ({
      x: Float64Array.from({ length: count * dim }, () => rng.normal()),
      y: Array.from({ length: count }, () => Math.floor(rng.next() * 3)),
    });
    const train = make(n);
    const test = make(120);
    const r = trainProbe(train.x, train.y, test.x, test.y, dim, 3, {
      iterations: 400,
      learningRate: 0.3,
      decay: 1e-4,
      seed: 1,
    });
    expect(r.trainAccuracy).toBeGreaterThan(0.9);
    expect(r.testAccuracy).toBeLessThan(0.55);
  });
});
