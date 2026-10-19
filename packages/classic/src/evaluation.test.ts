import { Matrix, createRng } from '@sim/core';
import { describe, expect, it } from 'vitest';
import fx from '../fixtures/evaluation.json';
import {
  bestThreshold,
  brierScore,
  calibrationBins,
  costAt,
  expectedCalibrationError,
  metricsAt,
  oversample,
  plattScaler,
  prCurve,
  smote,
} from './evaluation';

describe('against scikit-learn', () => {
  it('average precision and the precision-recall curve match', () => {
    const pr = prCurve(fx.score, fx.y);
    expect(pr.averagePrecision).toBeCloseTo(fx.ap, 10);
    // every (recall, precision) point of ours is a point of scikit-learn's curve
    const ref = new Set(
      fx.pr.recall.map((r, i) => `${r.toFixed(10)}|${fx.pr.precision[i]!.toFixed(10)}`),
    );
    pr.recall.forEach((r, i) =>
      expect(ref.has(`${r.toFixed(10)}|${pr.precision[i]!.toFixed(10)}`)).toBe(true),
    );
  });

  it('metrics at a threshold match', () => {
    const m = metricsAt(fx.score, fx.y, 0.4);
    expect(m.precision).toBeCloseTo(fx.at04.precision, 12);
    expect(m.recall).toBeCloseTo(fx.at04.recall, 12);
    expect(m.f1).toBeCloseTo(fx.at04.f1, 12);
    expect(m.mcc).toBeCloseTo(fx.at04.mcc, 12);
  });

  it('Brier score and the reliability diagram match', () => {
    expect(brierScore(fx.score, fx.y)).toBeCloseTo(fx.brier, 12);
    const bins = calibrationBins(fx.score, fx.y, 10);
    expect(bins).toHaveLength(fx.calibration.observed.length);
    bins.forEach((b, i) => {
      expect(b.observed).toBeCloseTo(fx.calibration.observed[i]!, 10);
      expect(b.predicted).toBeCloseTo(fx.calibration.predicted[i]!, 10);
    });
  });
});

describe('metrics', () => {
  it('a perfect classifier scores 1 everywhere', () => {
    const m = metricsAt([0.9, 0.8, 0.1, 0.2], [1, 1, 0, 0], 0.5);
    expect([m.accuracy, m.precision, m.recall, m.f1, m.mcc]).toEqual([1, 1, 1, 1, 1]);
  });

  it('predicting everything negative on rare positives has high accuracy and zero recall', () => {
    const y = [...new Array(98).fill(0), 1, 1];
    const m = metricsAt(new Array(100).fill(0.1), y, 0.5);
    expect(m.accuracy).toBe(0.98);
    expect(m.recall).toBe(0);
    expect(m.mcc).toBe(0);
  });

  it('lowering the threshold never lowers recall and never raises specificity', () => {
    const rng = createRng(4);
    const y = Array.from({ length: 300 }, () => (rng.next() < 0.3 ? 1 : 0));
    const s = y.map((v) => Math.min(0.99, Math.max(0.01, 0.3 + 0.3 * v + rng.normal() * 0.2)));
    let prev = { recall: -1, specificity: 2 };
    for (const t of [0.9, 0.7, 0.5, 0.3, 0.1]) {
      const m = metricsAt(s, y, t);
      expect(m.recall).toBeGreaterThanOrEqual(prev.recall);
      expect(m.specificity).toBeLessThanOrEqual(prev.specificity);
      prev = m;
    }
  });
});

describe('calibration', () => {
  const rng = createRng(6);
  // a model that is over-confident: true probability p, reported probability pushed towards 0 or 1
  const p = Array.from({ length: 4000 }, () => rng.next());
  const y = p.map((v) => (rng.next() < v ? 1 : 0));
  const over = p.map((v) => 1 / (1 + Math.exp(-3 * Math.log(v / (1 - v)))));

  it('a calibrated model has a small error and an over-confident one a large error', () => {
    expect(expectedCalibrationError(p, y)).toBeLessThan(0.04);
    expect(expectedCalibrationError(over, y)).toBeGreaterThan(0.1);
  });

  it('Platt scaling repairs the over-confident scores (fitted on one half, checked on the other)', () => {
    const half = 2000;
    const f = plattScaler(over.slice(0, half), y.slice(0, half));
    const fixed = over.slice(half).map(f);
    expect(expectedCalibrationError(fixed, y.slice(half))).toBeLessThan(0.05);
    expect(brierScore(fixed, y.slice(half))).toBeLessThan(
      brierScore(over.slice(half), y.slice(half)),
    );
  });
});

describe('rebalancing', () => {
  const rng = createRng(2);
  const x = new Matrix(
    110,
    2,
    Float64Array.from({ length: 220 }, () => rng.normal()),
  );
  const y = Array.from({ length: 110 }, (_, i) => (i < 10 ? 1 : 0));

  it('oversampling duplicates minority rows until the classes are equal', () => {
    const r = oversample(x, y);
    expect(r.y.filter((v) => v === 1).length).toBe(100);
    expect(r.x.rows).toBe(200);
  });

  it('SMOTE adds new minority points that lie within the minority cloud', () => {
    const r = smote(x, y, 3);
    expect(r.y.filter((v) => v === 1).length).toBe(100);
    const lo0 = Math.min(...Array.from({ length: 10 }, (_, i) => x.get(i, 0)));
    const hi0 = Math.max(...Array.from({ length: 10 }, (_, i) => x.get(i, 0)));
    for (let i = 110; i < r.x.rows; i++) {
      expect(r.x.get(i, 0)).toBeGreaterThanOrEqual(lo0 - 1e-9);
      expect(r.x.get(i, 0)).toBeLessThanOrEqual(hi0 + 1e-9);
    }
  });
});

describe('costs', () => {
  it('when missing a positive costs far more than a false alarm the best threshold drops', () => {
    const rng = createRng(8);
    const y = Array.from({ length: 1000 }, () => (rng.next() < 0.2 ? 1 : 0));
    const s = y.map((v) => Math.min(0.99, Math.max(0.01, 0.3 + 0.35 * v + rng.normal() * 0.2)));
    const equal = bestThreshold(s, y, 1, 1);
    const costly = bestThreshold(s, y, 20, 1);
    expect(costly.threshold).toBeLessThan(equal.threshold);
    expect(costly.cost).toBeLessThanOrEqual(costAt(s, y, 0.5, 20, 1));
  });
});
