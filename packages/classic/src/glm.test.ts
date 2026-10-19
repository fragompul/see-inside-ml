import { Matrix } from '@sim/core';
import { describe, expect, it } from 'vitest';
import fixture from '../fixtures/glm.json';
import { Glm, type GlmFamily } from './glm';

const x = Matrix.from(fixture.x);
const near = (a: number, b: number, tol: number) =>
  expect(Math.abs(a - b)).toBeLessThanOrEqual(tol * Math.max(1, Math.abs(b)));

describe('IRLS agrees with statsmodels GLM', () => {
  for (const [name, ref] of Object.entries(fixture.families)) {
    it(`${name}: coefficients, standard errors, deviances and p-values`, () => {
      const model = new Glm({ family: name as GlmFamily }).fit(x, ref.y);
      const s = model.summary();
      expect(model.converged).toBe(true);
      ref.params.forEach((b, j) => near(s.estimates[j] as number, b, 1e-7));
      ref.bse.forEach((b, j) => near(s.stdErrors[j] as number, b, 1e-6));
      ref.pvalues.forEach((b, j) => near(s.p[j] as number, b, 1e-5));
      near(s.deviance, ref.deviance, 1e-8);
      near(s.nullDeviance, ref.null_deviance, 1e-8);
      near(s.pearson, ref.pearson_chi2, 1e-8);
      near(s.dispersion, ref.scale, 1e-8);
      if (ref.llf !== null) {
        near(s.logLikelihood as number, ref.llf, 1e-8);
        near(s.aic as number, ref.aic as number, 1e-8);
      }
    });
  }

  it('predictions are means: probabilities for the binomial, positive rates for the Poisson', () => {
    const binom = new Glm({ family: 'binomial' }).fit(x, fixture.families.binomial.y);
    for (const p of binom.predict(x)) expect(p > 0 && p < 1).toBe(true);
    const pois = new Glm({ family: 'poisson' }).fit(x, fixture.families.poisson.y);
    for (const p of pois.predict(x)) expect(p).toBeGreaterThan(0);
  });

  it('the score equations hold at the solution: Xᵀ(y − μ) = 0 for canonical links', () => {
    for (const name of ['gaussian', 'binomial', 'poisson'] as const) {
      const y = fixture.families[name].y;
      const model = new Glm({ family: name }).fit(x, y);
      const mu = model.predict(x);
      let sum = 0;
      let sum0 = 0;
      for (let i = 0; i < x.rows; i++) {
        sum += x.get(i, 0) * ((y[i] as number) - (mu[i] as number));
        sum0 += (y[i] as number) - (mu[i] as number);
      }
      expect(Math.abs(sum)).toBeLessThan(1e-6);
      expect(Math.abs(sum0)).toBeLessThan(1e-6);
    }
  });

  it('a ridge penalty shrinks the coefficients', () => {
    const y = fixture.families.binomial.y;
    const plain = new Glm({ family: 'binomial' }).fit(x, y);
    const ridge = new Glm({ family: 'binomial', l2: 20 }).fit(x, y);
    expect(Math.hypot(...ridge.coef)).toBeLessThan(Math.hypot(...plain.coef));
  });
});
