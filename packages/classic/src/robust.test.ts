import { Matrix } from '@sim/core';
import { describe, expect, it } from 'vitest';
import fixture from '../fixtures/robust.json';
import { LinearRegression } from './linear';
import { HuberRegression, QuantileRegression, TheilSen, pinballLoss } from './robust';

const x = Matrix.from(fixture.x);
const near = (a: number, b: number, tol: number) =>
  expect(Math.abs(a - b)).toBeLessThanOrEqual(tol * Math.max(1, Math.abs(b)));

describe('robust regression agrees with statsmodels and SciPy', () => {
  it('Huber M-estimation (RLM with HuberT)', () => {
    const m = new HuberRegression().fit(x, fixture.y);
    near(m.intercept, fixture.huber.params[0] as number, 1e-6);
    near(m.coef[0] as number, fixture.huber.params[1] as number, 1e-6);
    near(m.scale, fixture.huber.scale, 1e-6);
    fixture.huber.weights.forEach((w, i) => near(m.weights[i] as number, w, 1e-5));
  });

  for (const [q, params] of Object.entries(fixture.quantile)) {
    it(`quantile regression, q = ${q}`, () => {
      const m = new QuantileRegression({ quantile: Number(q) }).fit(x, fixture.y);
      near(m.intercept, params[0] as number, 2e-3);
      near(m.coef[0] as number, params[1] as number, 2e-3);
    });
  }

  it('Theil–Sen (median of pairwise slopes)', () => {
    const m = new TheilSen().fit(x, fixture.y);
    near(m.slope, fixture.theil_sen.slope, 1e-12);
    near(m.intercept, fixture.theil_sen.intercept, 1e-6);
  });
});

describe('what robustness buys', () => {
  const clean = Matrix.from(Array.from({ length: 40 }, (_, i) => [i / 10]));
  const yClean = Array.from({ length: 40 }, (_, i) => 1 + 2 * (i / 10) + (i % 2 ? 0.1 : -0.1));
  const yDirty = yClean.map((v, i) => (i === 38 ? v - 25 : v));

  it('one wild point drags least squares far more than Huber, quantile or Theil–Sen', () => {
    const slopeError = (slope: number) => Math.abs(slope - 2);
    const ols = new LinearRegression().fit(clean, yDirty).coef[0] as number;
    const huber = new HuberRegression().fit(clean, yDirty).coef[0] as number;
    const median = new QuantileRegression().fit(clean, yDirty).coef[0] as number;
    const ts = new TheilSen().fit(clean, yDirty).slope;
    expect(slopeError(ols)).toBeGreaterThan(0.3);
    for (const robust of [huber, median, ts]) expect(slopeError(robust)).toBeLessThan(0.1);
  });

  it('Huber gives the outlier a small weight', () => {
    const m = new HuberRegression().fit(clean, yDirty);
    expect(m.weights[38]).toBeLessThan(0.1);
    expect(m.weights[3]).toBe(1);
  });

  it('the fitted quantile line leaves about that share of points below it', () => {
    for (const q of [0.2, 0.5, 0.8]) {
      const m = new QuantileRegression({ quantile: q }).fit(x, fixture.y);
      const pred = m.predict(x);
      const below =
        fixture.y.filter((v, i) => v <= (pred[i] as number) + 1e-9).length / fixture.y.length;
      expect(Math.abs(below - q)).toBeLessThan(0.12);
    }
  });

  it('quantile regression minimises the pinball loss: no nearby line does better', () => {
    const q = 0.7;
    const m = new QuantileRegression({ quantile: q }).fit(x, fixture.y);
    const best = pinballLoss(fixture.y, m.predict(x), q);
    for (const [di, ds] of [
      [0.05, 0],
      [-0.05, 0],
      [0, 0.03],
      [0, -0.03],
    ] as const) {
      const pred = Float64Array.from(
        { length: x.rows },
        (_, i) => m.intercept + di + ((m.coef[0] as number) + ds) * x.get(i, 0),
      );
      expect(pinballLoss(fixture.y, pred, q)).toBeGreaterThanOrEqual(best - 1e-9);
    }
  });
});
