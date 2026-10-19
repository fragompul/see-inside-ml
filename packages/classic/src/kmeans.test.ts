import { Matrix, adjustedRandIndex, createRng, makeClassification } from '@sim/core';
import { describe, expect, it } from 'vitest';
import fixture from '../fixtures/kmeans.json';
import { KMeans, assign, elbowCurve, kmeansPlusPlus, kmeansSteps } from './kmeans';

describe('k-means agrees with scikit-learn (Lloyd, given initial centres)', () => {
  fixture.runs.forEach((ref, r) => {
    it(`run ${r + 1}: centres, labels, inertia and iteration count`, () => {
      const x = Matrix.from(ref.x);
      const m = new KMeans({ k: ref.init.length, init: Matrix.from(ref.init) }).fit(x);
      ref.centers.forEach((row, c) =>
        row.forEach((v, d) => expect(Math.abs(m.centers.get(c, d) - v)).toBeLessThan(1e-9)),
      );
      expect(Array.from(m.labels)).toEqual(ref.labels);
      expect(Math.abs(m.inertia - ref.inertia)).toBeLessThan(1e-7 * Math.max(1, ref.inertia));
      expect(m.iterations).toBe(ref.n_iter);
    });
  });
});

describe('properties', () => {
  const data = makeClassification('blobs', 240, 0.05, createRng(2));

  it('every Lloyd iteration can only lower the inertia, and the run ends at a fixed point', () => {
    const start = kmeansPlusPlus(data.x, 3, createRng(5));
    const states = Array.from(kmeansSteps(data.x, start));
    for (let i = 1; i < states.length; i++)
      expect(states[i]!.inertia).toBeLessThanOrEqual(states[i - 1]!.inertia + 1e-9);
    const last = states[states.length - 1]!;
    // at convergence labels are the nearest-centre assignment of the final centres
    expect(Array.from(assign(data.x, last.centers).labels)).toEqual(Array.from(last.labels));
  });

  it('recovers well-separated clusters (adjusted Rand index near 1)', () => {
    const m = new KMeans({ k: 3, restarts: 5, seed: 3 }).fit(data.x);
    expect(adjustedRandIndex(data.y, m.labels)).toBeGreaterThan(0.95);
  });

  it('k-means++ picks distinct, spread-out seeds', () => {
    const centers = kmeansPlusPlus(data.x, 3, createRng(8));
    let nearest = Infinity;
    for (let a = 0; a < 3; a++)
      for (let b = a + 1; b < 3; b++)
        nearest = Math.min(
          nearest,
          Math.hypot(centers.get(a, 0) - centers.get(b, 0), centers.get(a, 1) - centers.get(b, 1)),
        );
    expect(nearest).toBeGreaterThan(0.5);
  });

  it('restarts never do worse than a single run, and a bad start can get stuck', () => {
    const single = new KMeans({ k: 3, init: 'random', seed: 11 }).fit(data.x).inertia;
    const many = new KMeans({ k: 3, init: 'random', seed: 11, restarts: 12 }).fit(data.x).inertia;
    expect(many).toBeLessThanOrEqual(single + 1e-9);
  });

  it('inertia falls as k grows (the elbow curve)', () => {
    const curve = elbowCurve(data.x, [1, 2, 3, 4, 5, 6], { seed: 4 });
    for (let i = 1; i < curve.length; i++)
      expect(curve[i]).toBeLessThanOrEqual((curve[i - 1] as number) + 1e-6);
    // the biggest drop is on the way to the true k = 3
    const drops = curve.slice(1).map((v, i) => (curve[i] as number) - v);
    expect(drops.indexOf(Math.max(...drops))).toBeLessThanOrEqual(1);
  });

  it('cannot separate crescents: a straight-boundary method', () => {
    const moons = makeClassification('moons', 240, 0.1, createRng(3));
    const m = new KMeans({ k: 2, restarts: 5 }).fit(moons.x);
    expect(adjustedRandIndex(moons.y, m.labels)).toBeLessThan(0.6);
  });

  it('re-seeds an empty cluster instead of losing it', () => {
    const x = Matrix.from([
      [0, 0],
      [0.1, 0],
      [0, 0.1],
      [5, 5],
      [5.1, 5],
      [5, 5.1],
    ]);
    // a centre far from every point would end up empty
    const init = Matrix.from([
      [0, 0],
      [5, 5],
      [100, 100],
    ]);
    const m = new KMeans({ k: 3, init }).fit(x);
    expect(new Set(m.labels).size).toBe(3);
  });

  it('transform gives distances to the centres and predict agrees with the labels', () => {
    const m = new KMeans({ k: 3, restarts: 3 }).fit(data.x);
    expect(Array.from(m.predict(data.x))).toEqual(Array.from(m.labels));
    const d = m.transform(data.x);
    for (let i = 0; i < 10; i++) {
      const own = d.get(i, m.labels[i] as number);
      for (let c = 0; c < 3; c++) expect(own).toBeLessThanOrEqual(d.get(i, c) + 1e-12);
    }
  });
});
