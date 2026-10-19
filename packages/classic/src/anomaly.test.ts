import { Matrix, createRng } from '@sim/core';
import { describe, expect, it } from 'vitest';
import fx from '../fixtures/anomaly.json';
import {
  IsolationForest,
  MahalanobisDetector,
  averagePathLength,
  localOutlierFactor,
  rocAuc,
  thresholdFor,
} from './anomaly';
import { SOM, somSteps, quantizationError, topographicError, uMatrix } from './som';

const toMatrix = (rows: number[][]) =>
  new Matrix(rows.length, rows[0]!.length, Float64Array.from(rows.flat()));

function blobsWithOutliers(seed: number) {
  const rng = createRng(seed);
  const rows: number[][] = [];
  const label: number[] = [];
  for (let i = 0; i < 200; i++) {
    rows.push([rng.normal(), rng.normal()]);
    label.push(0);
  }
  for (let i = 0; i < 10; i++) {
    rows.push(
      [rng.uniform(-8, 8), rng.uniform(-8, 8)].map((v) =>
        Math.abs(v) < 4 ? v + Math.sign(v || 1) * 4 : v,
      ),
    );
    label.push(1);
  }
  return { x: toMatrix(rows), label };
}

describe('local outlier factor', () => {
  const x = toMatrix(fx.x);
  it.each(fx.lof)('matches scikit-learn for k=$k', ({ k, scores }) => {
    const mine = localOutlierFactor(x, k).scores;
    scores.forEach((s, i) => expect(mine[i]).toBeCloseTo(s, 6));
  });
});

describe('isolation forest', () => {
  it('uses the reference path-length normaliser', () => {
    for (const [n, v] of Object.entries(fx.path_length))
      expect(averagePathLength(Number(n))).toBeCloseTo(v, 9);
  });
  it('scores injected outliers above the crowd', () => {
    const { x, label } = blobsWithOutliers(3);
    const s = new IsolationForest({ trees: 100, seed: 2 }).fit(x).scores(x);
    expect(rocAuc(s, label)).toBeGreaterThan(0.95);
    for (const v of s) {
      expect(v).toBeGreaterThan(0);
      expect(v).toBeLessThan(1);
    }
  });
  it('is deterministic for a seed', () => {
    const { x } = blobsWithOutliers(4);
    const a = new IsolationForest({ seed: 5 }).fit(x).scores(x);
    const b = new IsolationForest({ seed: 5 }).fit(x).scores(x);
    expect(Array.from(a)).toEqual(Array.from(b));
  });
});

describe('Mahalanobis', () => {
  it('matches the closed form on a correlated cloud and finds outliers', () => {
    const { x, label } = blobsWithOutliers(6);
    const m = new MahalanobisDetector().fit(x);
    expect(rocAuc(m.scores(x), label)).toBeGreaterThan(0.95);
    const unit = new MahalanobisDetector().fit(
      toMatrix([
        [1, 0],
        [-1, 0],
        [0, 1],
        [0, -1],
      ]),
    );
    const s = unit.scores(
      toMatrix([
        [0, 0],
        [1, 0],
      ]),
    );
    expect(s[0]).toBeCloseTo(0, 9);
    expect(s[1]).toBeCloseTo(Math.sqrt(1.5), 6); // sample variance 2/3 per axis → 1/√(2/3)
  });
});

describe('evaluation helpers', () => {
  it('rocAuc handles perfect, reversed and tied scores', () => {
    expect(rocAuc([1, 2, 3, 4], [0, 0, 1, 1])).toBe(1);
    expect(rocAuc([4, 3, 2, 1], [0, 0, 1, 1])).toBe(0);
    expect(rocAuc([1, 1, 1, 1], [0, 0, 1, 1])).toBe(0.5);
  });
  it('thresholdFor keeps the requested fraction', () => {
    const s = Array.from({ length: 100 }, (_, i) => i);
    expect(thresholdFor(s, 0.1)).toBe(90);
  });
});

describe('self-organising map', () => {
  const rng = createRng(8);
  const rows: number[][] = [];
  for (let i = 0; i < 300; i++) rows.push([rng.uniform(0, 1), rng.uniform(0, 1)]);
  const x = toMatrix(rows);
  const som = new SOM({ rows: 6, cols: 6, epochs: 30, seed: 2 }).fit(x);
  it('ends with a small quantization error and a tangled start', () => {
    expect(som.history[som.history.length - 1]!).toBeLessThan(0.08);
    const start = [...somSteps(x, { rows: 6, cols: 6, epochs: 1, seed: 2 })][0]!;
    expect(topographicError(start.weights, x, 6)).toBeGreaterThan(
      topographicError(som.weights, x, 6),
    );
    expect(quantizationError(som.weights, x)).toBeCloseTo(som.history[som.history.length - 1]!, 9);
  });
  it('unfolds a uniform square into a topology-preserving sheet', () => {
    expect(topographicError(som.weights, x, 6)).toBeLessThan(0.25);
    expect(uMatrix(som.weights, 6, 6)).toHaveLength(36);
  });
  it('assigns every sample to a valid cell', () => {
    for (const c of som.assign(x)) {
      expect(c.row).toBeGreaterThanOrEqual(0);
      expect(c.col).toBeLessThan(6);
    }
  });
});
