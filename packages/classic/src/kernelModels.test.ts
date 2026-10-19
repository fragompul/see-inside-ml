import { Matrix, createRng } from '@sim/core';
import { describe, expect, it } from 'vitest';
import fixture from '../fixtures/kernel_models.json';
import { BayesianRidge } from './bayesianLinear';
import { GaussianProcess, KernelRidge, type GpKernel } from './gp';

const x = Matrix.from(fixture.x);
const grid = Matrix.from(fixture.grid);
const close = (a: number, b: number, tol: number) => expect(Math.abs(a - b)).toBeLessThan(tol);

describe('Gaussian process regression agrees with scikit-learn', () => {
  for (const ref of fixture.gp) {
    it(`${ref.kernel}, σ² = ${ref.variance}, ℓ = ${ref.length}, noise = ${ref.noise}`, () => {
      const gp = new GaussianProcess({
        kernel: ref.kernel as GpKernel,
        variance: ref.variance,
        lengthScale: ref.length,
        noise: ref.noise,
        period: 2.5,
      }).fit(x, fixture.y);
      const { mean, std } = gp.posterior(grid);
      ref.mean.forEach((v, i) => close(mean[i] as number, v, 1e-7));
      ref.std.forEach((v, i) => close(std[i] as number, v, 1e-6));
      close(gp.logMarginalLikelihood(), ref.lml, 1e-8);
    });
  }
});

describe('kernel ridge regression agrees with scikit-learn', () => {
  for (const ref of fixture.krr) {
    it(`α = ${ref.alpha}, γ = ${ref.gamma}`, () => {
      // scikit-learn's RBF is exp(−γ d²) = our RBF with length scale 1/√(2γ)
      const m = new KernelRidge({
        kernel: 'rbf',
        lengthScale: 1 / Math.sqrt(2 * ref.gamma),
        alpha: ref.alpha,
      }).fit(x, fixture.y);
      const pred = m.predict(grid);
      ref.pred.forEach((v, i) => close(pred[i] as number, v, 1e-8));
      ref.dual.forEach((v, i) => close(m.coef[i] as number, v, 1e-8));
    });
  }
});

describe('Bayesian ridge agrees with scikit-learn', () => {
  const ref = fixture.bayes;
  const m = new BayesianRidge({ tolerance: 1e-12, maxIterations: 1000 }).fit(
    Matrix.from(ref.x),
    ref.y,
  );
  it('coefficients, intercept and the two precisions found by evidence maximisation', () => {
    ref.coef.forEach((v, j) => close(m.coef[j] as number, v, 1e-6));
    close(m.intercept, ref.intercept, 1e-6);
    close(m.alpha, ref.alpha, 1e-5 * ref.alpha);
    close(m.lambda, ref.lambda, 1e-5 * ref.lambda);
    ref.sigma.forEach((row, a) => row.forEach((v, b) => close(m.covariance.get(a, b), v, 1e-7)));
  });
  it('predictive mean and standard deviation', () => {
    const q = Matrix.from(ref.queries);
    const pred = m.predict(q);
    const std = m.predictStd(q);
    ref.pred.forEach((v, i) => close(pred[i] as number, v, 1e-6));
    ref.std.forEach((v, i) => close(std[i] as number, v, 1e-6));
  });
});

describe('properties of Gaussian processes', () => {
  it('uncertainty is small near the data and returns to the prior far from it', () => {
    const gp = new GaussianProcess({ variance: 1, lengthScale: 0.7, noise: 1e-4 }).fit(
      x,
      fixture.y,
    );
    const { std } = gp.posterior(Matrix.from([[x.get(10, 0)], [30]]));
    expect(std[0]).toBeLessThan(0.05);
    expect(std[1]).toBeCloseTo(1, 6);
  });

  it('with tiny noise the mean interpolates well-separated training points', () => {
    const xs = Matrix.from(Array.from({ length: 12 }, (_, i) => [-3 + i * 0.55]));
    const ys = Array.from({ length: 12 }, (_, i) => Math.sin(-3 + i * 0.55));
    const gp = new GaussianProcess({ variance: 1, lengthScale: 0.8, noise: 1e-8 }).fit(xs, ys);
    const pred = gp.predict(xs);
    ys.forEach((v, i) => close(pred[i] as number, v, 1e-4));
  });

  it('optimising the hyperparameters raises the marginal likelihood and finds a sensible scale', () => {
    const gp = new GaussianProcess({ variance: 1, lengthScale: 0.1, noise: 0.5 }).fit(x, fixture.y);
    const before = gp.logMarginalLikelihood();
    const after = gp.optimize();
    expect(after).toBeGreaterThan(before);
    expect(gp.hyper.lengthScale).toBeGreaterThan(0.5);
    expect(gp.hyper.lengthScale).toBeLessThan(5);
    expect(gp.hyper.noise).toBeLessThan(0.3);
    // a local maximum: nudging a hyperparameter does not help
    for (const key of ['lengthScale', 'variance', 'noise'] as const) {
      for (const f of [0.97, 1.03]) {
        const probe = new GaussianProcess({ ...gp.hyper, [key]: gp.hyper[key] * f }).fit(
          x,
          fixture.y,
        );
        expect(probe.logMarginalLikelihood()).toBeLessThanOrEqual(after + 1e-6);
      }
    }
  });

  it('posterior samples vary around the mean and are reproducible for a seed', () => {
    const gp = new GaussianProcess({ variance: 1, lengthScale: 1, noise: 0.05 }).fit(x, fixture.y);
    const a = gp.sample(grid, 3, createRng(7));
    const b = gp.sample(grid, 3, createRng(7));
    expect(Array.from(a[0]!)).toEqual(Array.from(b[0]!));
    expect(Array.from(a[0]!)).not.toEqual(Array.from(a[1]!));
    const mean = gp.predict(grid);
    // averaging many samples recovers the posterior mean
    const many = gp.sample(grid, 400, createRng(3));
    const avg = Array.from(
      mean,
      (_, i) => many.reduce((s, f) => s + (f[i] as number), 0) / many.length,
    );
    avg.forEach((v, i) => close(v, mean[i] as number, 0.15));
  });
});
