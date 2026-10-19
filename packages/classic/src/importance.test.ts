import { Matrix, createRng, r2 } from '@sim/core';
import { describe, expect, it } from 'vitest';
import { iceCurves, permutationImportance } from './importance';
import { LinearRegression } from './linear';

function makeData(n: number, seed: number) {
  const rng = createRng(seed);
  const x = new Matrix(
    n,
    4,
    Float64Array.from({ length: n * 4 }, () => rng.normal()),
  );
  // y = 3 x0 - 2 x1 + 0 x2 + 0 x3 + noise
  const y = Array.from(
    { length: n },
    (_, i) => 3 * x.get(i, 0) - 2 * x.get(i, 1) + rng.normal() * 0.3,
  );
  return { x, y };
}

describe('permutation importance', () => {
  const { x, y } = makeData(2000, 4);
  const model = new LinearRegression().fit(x, y);
  const predict = (m: Matrix) => model.predict(m);

  it('for a linear model the drop in R² of a feature is twice its squared coefficient times its variance, over the variance of y', () => {
    const res = permutationImportance(predict, x, y, r2, 20, 2);
    const varY =
      y.reduce((a, v) => a + v * v, 0) / y.length - (y.reduce((a, v) => a + v, 0) / y.length) ** 2;
    // the shuffled column is an independent copy: the error grows by coef^2 * E[(x - x')^2] = 2 coef^2 var(x)
    expect(res.importance[0]).toBeCloseTo((2 * 9) / varY, 1);
    expect(res.importance[1]).toBeCloseTo((2 * 4) / varY, 1);
  });

  it('features the model ignores have no importance', () => {
    const res = permutationImportance(predict, x, y, r2, 20, 2);
    expect(Math.abs(res.importance[2]!)).toBeLessThan(0.01);
    expect(Math.abs(res.importance[3]!)).toBeLessThan(0.01);
  });

  it('is deterministic for a seed and reports a baseline equal to the plain score', () => {
    const a = permutationImportance(predict, x, y, r2, 5, 9);
    const b = permutationImportance(predict, x, y, r2, 5, 9);
    expect(a.importance).toEqual(b.importance);
    expect(a.baseline).toBeCloseTo(r2(y, predict(x)), 12);
  });
});

describe('ICE and partial dependence', () => {
  const { x, y } = makeData(500, 5);
  const model = new LinearRegression().fit(x, y);
  const predict = (m: Matrix) => model.predict(m);

  it('for a linear model every ICE curve is a straight line with the coefficient as slope', () => {
    const { grid, curves } = iceCurves(predict, x, 0, 10, 50);
    for (const c of curves) {
      const slope = (c[9]! - c[0]!) / (grid[9]! - grid[0]!);
      expect(slope).toBeCloseTo(model.coef[0]!, 6);
    }
  });

  it('the partial dependence is the average of the curves and is flat for an ignored feature', () => {
    const r = iceCurves(predict, x, 2, 8, 100);
    const mid = r.curves.reduce((s, c) => s + c[3]!, 0) / r.curves.length;
    expect(r.average[3]).toBeCloseTo(mid, 12);
    expect(Math.max(...r.average) - Math.min(...r.average)).toBeLessThan(0.15);
  });

  it('an interaction shows up as ICE curves that disagree while their average hides it', () => {
    const rng = createRng(6);
    const n = 400;
    const xs = new Matrix(
      n,
      2,
      Float64Array.from({ length: n * 2 }, () => rng.uniform(-1, 1)),
    );
    // f = x0 * sign(x1): half of the rows slope up, half slope down, the average is flat
    const f = (m: Matrix) =>
      Float64Array.from({ length: m.rows }, (_, i) => m.get(i, 0) * Math.sign(m.get(i, 1)));
    const r = iceCurves(f, xs, 0, 9, 200);
    const slopes = r.curves.map((c) => c[8]! - c[0]!);
    expect(Math.max(...slopes)).toBeGreaterThan(1);
    expect(Math.min(...slopes)).toBeLessThan(-1);
    expect(Math.abs(r.average[8]! - r.average[0]!)).toBeLessThan(0.3);
  });
});
