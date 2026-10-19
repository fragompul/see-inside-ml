import { describe, expect, it } from 'vitest';
import {
  DEFAULT_BV,
  GRID,
  complexityCurve,
  fitFeatures,
  legendreFeatures,
  learningCurve,
  simulate,
} from './biasVarianceCore';

describe('Legendre features and the fit', () => {
  it('the first Legendre polynomials are 1, t, (3t^2-1)/2', () => {
    const f = legendreFeatures([0, 0.5, 1], 3);
    // x=1 -> t=1: 1, 1, 1 ; x=0.5 -> t=0: 1, 0, -0.5
    expect([f.get(2, 0), f.get(2, 1), f.get(2, 2)]).toEqual([1, 1, 1]);
    expect(f.get(1, 1)).toBeCloseTo(0, 12);
    expect(f.get(1, 2)).toBeCloseTo(-0.5, 12);
  });

  it('recovers an exact quadratic from a few points', () => {
    const x = [0.1, 0.3, 0.5, 0.7, 0.9];
    const y = x.map((v) => 2 - 3 * v + 4 * v * v);
    const fit = fitFeatures(x, y, 3, 1e-12);
    fit
      .predict([0.2, 0.8])
      .forEach((p, i) =>
        expect(p).toBeCloseTo(2 - 3 * [0.2, 0.8][i]! + 4 * [0.2, 0.8][i]! ** 2, 6),
      );
  });

  it('with more features than points it interpolates the training data exactly', () => {
    const x = [0.05, 0.2, 0.4, 0.65, 0.9];
    const y = [1, -1, 0.5, 2, -0.5];
    const fit = fitFeatures(x, y, 12, 1e-12);
    fit.predict(x).forEach((p, i) => expect(p).toBeCloseTo(y[i]!, 5));
  });

  it('the minimum-norm interpolator has the smallest coefficients among interpolators', () => {
    const x = [0.1, 0.4, 0.7];
    const y = [1, 0, 1];
    const small = fitFeatures(x, y, 6, 1e-12).coefficients;
    const norm = Math.hypot(...small);
    // adding a component of the null space (a vector orthogonal to all three feature rows) can only increase the norm
    const phi = legendreFeatures(x, 6);
    const w2 = Array.from(small);
    // find a null-space direction by Gram-Schmidt against the three rows
    const rows = [0, 1, 2].map((r) => Array.from({ length: 6 }, (_, c) => phi.get(r, c)));
    const candidate = [0, 0, 0, 0, 0, 1];
    const basis: number[][] = [];
    for (const rrow of rows) {
      const v = [...rrow];
      for (const b of basis) {
        const d = v.reduce((a, vi, i) => a + vi * b[i]!, 0);
        v.forEach((_, i) => (v[i] = v[i]! - d * b[i]!));
      }
      const nn = Math.hypot(...v);
      basis.push(v.map((a) => a / nn));
    }
    for (const b of basis) {
      const d = candidate.reduce((a, ci, i) => a + ci * b[i]!, 0);
      candidate.forEach((_, i) => (candidate[i] = candidate[i]! - d * b[i]!));
    }
    const perturbed = w2.map((v, i) => v + candidate[i]!);
    expect(Math.hypot(...perturbed)).toBeGreaterThan(norm);
  });
});

describe('the bias-variance decomposition', () => {
  it('test error equals bias squared plus variance plus noise, exactly', () => {
    for (const features of [1, 3, 8]) {
      const r = simulate({ ...DEFAULT_BV, features });
      expect(r.testError).toBeCloseTo(r.bias2 + r.variance + r.noiseVariance, 10);
    }
  });

  it('a too-simple model has high bias and low variance, a too-flexible one the opposite', () => {
    const simple = simulate({ ...DEFAULT_BV, features: 1 });
    const flexible = simulate({ ...DEFAULT_BV, features: 9, n: 40 });
    expect(simple.bias2).toBeGreaterThan(flexible.bias2);
    expect(flexible.variance).toBeGreaterThan(simple.variance);
  });

  it('is deterministic for a seed and covers the unit interval', () => {
    expect(simulate(DEFAULT_BV).testError).toBe(simulate(DEFAULT_BV).testError);
    expect(GRID[0]).toBe(0);
    expect(GRID[100]).toBe(1);
  });
});

describe('curves', () => {
  it('training error never rises with model size while it fits, and test error is U-shaped', () => {
    const curve = complexityCurve({ ...DEFAULT_BV, n: 30 }, 12);
    for (let i = 1; i < 10; i++)
      expect(curve[i]!.train).toBeLessThanOrEqual(curve[i - 1]!.train + 0.01);
    const best = curve.reduce((b, c) => (c.test < b.test ? c : b));
    expect(best.features).toBeGreaterThan(2);
    expect(best.features).toBeLessThan(12);
  });

  it('more data lowers the test error of a fixed model', () => {
    const lc = learningCurve(DEFAULT_BV, [10, 30, 100, 300]);
    expect(lc[3]!.test).toBeLessThan(lc[0]!.test);
  });
});
