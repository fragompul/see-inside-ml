import { Matrix, createRng, makeRegression, anscombe, mean } from '@sim/core';
import { describe, expect, it } from 'vitest';
import fixture from '../fixtures/linear.json';
import { LinearRegression, lineGradientDescent } from './linear';

const x = Matrix.from(fixture.x);
const close = (a: number, b: number, tol = 1e-8) =>
  expect(Math.abs(a - b)).toBeLessThanOrEqual(tol * Math.max(1, Math.abs(b)));

describe('least squares agrees with statsmodels', () => {
  const model = new LinearRegression().fit(x, fixture.y);
  const s = model.summary(['x1', 'x2', 'x3']);

  it('estimates, standard errors, t statistics, p-values and intervals', () => {
    fixture.params.forEach((b, j) => close(s.coefficients[j]!.estimate, b));
    fixture.bse.forEach((b, j) => close(s.coefficients[j]!.stdError, b));
    fixture.tvalues.forEach((b, j) => close(s.coefficients[j]!.t, b));
    fixture.pvalues.forEach((b, j) => close(s.coefficients[j]!.p, b, 1e-7));
    fixture.conf_int.forEach((ci, j) => {
      close(s.coefficients[j]!.ci[0], ci[0] as number, 1e-6);
      close(s.coefficients[j]!.ci[1], ci[1] as number, 1e-6);
    });
    expect(s.coefficients.map((c) => c.name)).toEqual(['intercept', 'x1', 'x2', 'x3']);
  });

  it('R², adjusted R², F test, information criteria and noise variance', () => {
    close(s.r2, fixture.rsquared);
    close(s.adjustedR2, fixture.rsquared_adj);
    close(s.fStatistic, fixture.fvalue, 1e-7);
    close(s.fPValue, fixture.f_pvalue, 1e-6);
    close(s.aic, fixture.aic, 1e-8);
    close(s.bic, fixture.bic, 1e-8);
    close(s.sigma2, fixture.sigma2);
    expect(s.dfResidual).toBe(56);
    expect(s.n).toBe(60);
  });

  it('leverage, Cook’s distance and variance inflation factors', () => {
    model.leverage().forEach((h, i) => close(h, fixture.hat[i] as number, 1e-9));
    model.cooksDistance().forEach((c, i) => close(c, fixture.cooks[i] as number, 1e-8));
    model.vif().forEach((v, j) => close(v, fixture.vif[j] as number, 1e-8));
    expect(model.leverage().reduce((a, b) => a + b, 0)).toBeCloseTo(4, 9); // trace of the hat matrix = parameters
  });

  it('predictions equal the design matrix times the coefficients; residuals are orthogonal to the features', () => {
    const pred = model.predict(x);
    const resid = model.residuals();
    for (let j = 0; j < 3; j++) {
      let dot = 0;
      for (let i = 0; i < 60; i++) dot += x.get(i, j) * (resid[i] as number);
      expect(Math.abs(dot)).toBeLessThan(1e-9);
    }
    expect(Math.abs(mean(resid))).toBeLessThan(1e-12);
    close(
      pred[0] as number,
      model.intercept + [0, 1, 2].reduce((a, j) => a + (model.coef[j] as number) * x.get(0, j), 0),
    );
  });
});

describe('ridge regression agrees with scikit-learn', () => {
  it.each(fixture.ridge)('alpha = $alpha', ({ alpha, coef, intercept }) => {
    const r = new LinearRegression({ l2: alpha }).fit(x, fixture.y);
    coef.forEach((c, j) => close(r.coef[j] as number, c, 1e-8));
    close(r.intercept, intercept, 1e-8);
  });

  it('shrinks the coefficients as the penalty grows, towards the mean of y', () => {
    const norms = [0, 1, 10, 100, 1e5].map((l2) => {
      const r = new LinearRegression({ l2 }).fit(x, fixture.y);
      return Math.hypot(...r.coef);
    });
    for (let i = 1; i < norms.length; i++) expect(norms[i]).toBeLessThan(norms[i - 1] as number);
    const heavy = new LinearRegression({ l2: 1e9 }).fit(x, fixture.y);
    expect(Math.abs(heavy.intercept - mean(fixture.y))).toBeLessThan(1e-3);
  });
});

describe('simple regression facts', () => {
  it('fits a noiseless line exactly, with and without an intercept', () => {
    const xs = Matrix.from([[1], [2], [3], [4]]);
    const m = new LinearRegression().fit(xs, [3, 5, 7, 9]);
    close(m.coef[0] as number, 2);
    close(m.intercept, 1);
    const through = new LinearRegression({ intercept: false }).fit(xs, [2, 4, 6, 8]);
    close(through.coef[0] as number, 2);
    expect(through.intercept).toBe(0);
  });

  it("Anscombe's four datasets give the same line", () => {
    for (const i of [1, 2, 3, 4] as const) {
      const t = anscombe(i);
      const m = new LinearRegression().fit(t.x, t.y);
      close(m.coef[0] as number, 0.5, 0.005);
      close(m.intercept, 3, 0.005);
    }
  });

  it('a single far-away point has huge leverage and a huge Cook’s distance only when it also breaks the trend', () => {
    const t = anscombe(4); // seven... ten points at x = 8, one at x = 19
    const m = new LinearRegression().fit(t.x, t.y);
    const h = m.leverage();
    expect(h[7]).toBeCloseTo(1, 9); // the lone point at x = 19 determines the slope entirely
    const noisy = makeRegression('outliers', 60, 0.3, createRng(2));
    const a = new LinearRegression().fit(noisy.x, noisy.y);
    expect(Math.max(...a.cooksDistance())).toBeGreaterThan(0.1);
  });

  it('collinear features inflate their variance, and the VIF says so', () => {
    const rng = createRng(1);
    const rows = Array.from({ length: 80 }, () => {
      const a = rng.normal();
      return [a, a + 0.05 * rng.normal(), rng.normal()];
    });
    const m = new LinearRegression().fit(
      Matrix.from(rows),
      rows.map((r) => r[0]! + r[2]!),
    );
    const v = m.vif();
    expect(v[0]).toBeGreaterThan(100);
    expect(v[1]).toBeGreaterThan(100);
    expect(v[2]).toBeLessThan(1.5);
  });

  it('requires fit() before the diagnostics and survives a singular design', () => {
    expect(() => new LinearRegression().residuals()).toThrow();
    const dup = Matrix.from([
      [1, 1],
      [2, 2],
      [3, 3],
      [4, 4],
    ]);
    expect(() => new LinearRegression().fit(dup, [1, 2, 3, 4])).toThrow(RangeError); // perfectly collinear: cannot be solved
    const ridge = new LinearRegression({ l2: 1 }).fit(dup, [1, 2, 3, 4]); // a penalty makes it solvable
    expect(Number.isFinite(ridge.coef[0] as number)).toBe(true);
  });
});

describe('gradient descent on a line', () => {
  it('converges to the least-squares solution and the loss never rises with a small step', () => {
    const t = makeRegression('line', 80, 0.4, createRng(3));
    const xs = Array.from({ length: 80 }, (_, i) => t.x.get(i, 0));
    const exact = new LinearRegression().fit(t.x, t.y);
    const steps = [...lineGradientDescent(xs, t.y, { lr: 0.05, steps: 600 })];
    expect(steps).toHaveLength(601);
    for (let i = 1; i < steps.length; i++)
      expect(steps[i]!.loss).toBeLessThanOrEqual(steps[i - 1]!.loss + 1e-12);
    const last = steps.at(-1)!;
    close(last.slope, exact.coef[0] as number, 1e-3);
    close(last.intercept, exact.intercept, 1e-3);
  });

  it('a step that is too large makes the loss explode', () => {
    const xs = [1, 2, 3, 4];
    const steps = [...lineGradientDescent(xs, [2, 4, 6, 8], { lr: 0.5, steps: 20 })];
    expect(steps.at(-1)!.loss).toBeGreaterThan(steps[0]!.loss * 100);
  });

  it('can start from a given line', () => {
    const first = lineGradientDescent([1, 2], [1, 2], {
      lr: 0.1,
      steps: 1,
      start: { slope: 3, intercept: -1 },
    }).next().value;
    expect(first).toMatchObject({ step: 0, slope: 3, intercept: -1 });
  });
});
