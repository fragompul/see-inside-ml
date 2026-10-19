import { Matrix } from '@sim/core';
import { describe, expect, it } from 'vitest';
import fixture from '../fixtures/regularized.json';
import { PolynomialRegression, polynomialFeatures } from './polynomial';
import {
  ElasticNet,
  Lasso,
  Ridge,
  alphaMax,
  regularizationPath,
  ridgePath,
  softThreshold,
} from './regularized';

const x = Matrix.from(fixture.x);
const near = (a: ArrayLike<number>, b: ArrayLike<number>, tol: number) => {
  expect(a.length).toBe(b.length);
  for (let i = 0; i < a.length; i++)
    expect(Math.abs((a[i] as number) - (b[i] as number))).toBeLessThanOrEqual(tol);
};

describe('lasso and elastic net agree with scikit-learn', () => {
  for (const ref of fixture.lasso) {
    it(`lasso, α = ${ref.alpha}`, () => {
      const m = new Lasso({ alpha: ref.alpha }).fit(x, fixture.y);
      near(m.coef, ref.coef, 1e-7);
      expect(Math.abs(m.intercept - ref.intercept)).toBeLessThan(1e-7);
    });
  }
  for (const ref of fixture.enet) {
    it(`elastic net, α = ${ref.alpha}, ρ = ${ref.l1_ratio}`, () => {
      const m = new ElasticNet({ alpha: ref.alpha, l1Ratio: ref.l1_ratio }).fit(x, fixture.y);
      near(m.coef, ref.coef, 1e-7);
      expect(Math.abs(m.intercept - ref.intercept)).toBeLessThan(1e-7);
    });
  }
  it('without an intercept', () => {
    const m = new Lasso({ alpha: fixture.no_intercept.alpha, intercept: false }).fit(x, fixture.y);
    near(m.coef, fixture.no_intercept.coef, 1e-7);
    expect(m.intercept).toBe(0);
  });
  it('the path matches lasso_path, warm starts included', () => {
    const p = regularizationPath(x, fixture.y, { alphas: fixture.path.alphas });
    p.coefs.forEach((c, k) => near(c, fixture.path.coefs[k] as number[], 1e-7));
  });
});

describe('properties of the penalties', () => {
  it('soft thresholding shrinks towards zero and clips', () => {
    expect(softThreshold(3, 1)).toBe(2);
    expect(softThreshold(-3, 1)).toBe(-2);
    expect(softThreshold(0.5, 1)).toBe(0);
  });

  it('above alphaMax every coefficient is exactly zero; just below, the first ones enter', () => {
    const top = alphaMax(x, fixture.y);
    expect(new Lasso({ alpha: top * 1.0001 }).fit(x, fixture.y).active).toBe(0);
    // the feature most correlated with y enters first (a correlated twin may follow immediately)
    const just = new Lasso({ alpha: top * 0.99 }).fit(x, fixture.y);
    expect(just.active).toBeGreaterThanOrEqual(1);
    expect(just.active).toBeLessThanOrEqual(2);
  });

  it('a weaker penalty keeps more features (the lasso selects variables)', () => {
    const path = regularizationPath(x, fixture.y, { count: 30 });
    const counts = path.coefs.map((c) => c.reduce((n, w) => n + (w !== 0 ? 1 : 0), 0));
    expect(counts[0]).toBe(0);
    for (let k = 1; k < counts.length; k++)
      expect(counts[k]).toBeGreaterThanOrEqual(counts[k - 1]! - 1);
    expect(counts[counts.length - 1]).toBeGreaterThanOrEqual(5);
  });

  it('elastic net with ρ = 0 is ridge with α scaled by n', () => {
    const n = x.rows;
    const enet = new ElasticNet({ alpha: 0.3, l1Ratio: 0 }).fit(x, fixture.y);
    const ridge = new Ridge({ alpha: 0.3 * n }).fit(x, fixture.y);
    near(enet.coef, ridge.coef, 1e-8);
  });

  it('ridge shrinks every coefficient monotonically as the penalty grows', () => {
    const p = ridgePath(x, fixture.y, [0.01, 0.1, 1, 10, 100, 1000]);
    const norms = p.coefs.map((c) => Math.hypot(...c));
    for (let k = 1; k < norms.length; k++) expect(norms[k]).toBeLessThan(norms[k - 1] as number);
  });

  it('the lasso solution satisfies its optimality (KKT) conditions', () => {
    const alpha = 0.3;
    const m = new Lasso({ alpha }).fit(x, fixture.y);
    const pred = m.predict(x);
    const n = x.rows;
    for (let j = 0; j < x.cols; j++) {
      let g = 0;
      for (let i = 0; i < n; i++)
        g += x.get(i, j) * ((fixture.y[i] as number) - (pred[i] as number));
      g /= n;
      if (m.coef[j] === 0) expect(Math.abs(g)).toBeLessThanOrEqual(alpha + 1e-8);
      else expect(Math.abs(g - alpha * Math.sign(m.coef[j] as number))).toBeLessThan(1e-7);
    }
  });
});

describe('polynomial regression agrees with the scikit-learn pipeline', () => {
  const xs = Matrix.from(fixture.poly.x.map((v) => [v]));
  for (const ref of fixture.poly.fits) {
    it(`degree ${ref.degree}, α = ${ref.alpha}`, () => {
      const m = new PolynomialRegression({
        degree: ref.degree,
        penalty: ref.alpha > 0 ? 'ridge' : 'none',
        alpha: ref.alpha,
      }).fit(xs, fixture.poly.y);
      const pred = m.predict(Matrix.from(ref.grid.map((v) => [v])));
      near(pred, ref.pred, ref.degree > 6 ? 1e-4 : 1e-6);
    });
  }
  for (const ref of fixture.poly.lasso) {
    it(`lasso on degree ${ref.degree}`, () => {
      const m = new PolynomialRegression({
        degree: ref.degree,
        penalty: 'lasso',
        alpha: ref.alpha,
      }).fit(xs, fixture.poly.y);
      near(m.predict(Matrix.from(ref.grid.map((v) => [v]))), ref.pred, 1e-6);
    });
  }
  it('expands a column into its powers', () => {
    const f = polynomialFeatures([2, -3], 3);
    expect(Array.from(f.data)).toEqual([2, 4, 8, -3, 9, -27]);
  });
});
