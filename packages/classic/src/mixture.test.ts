import { Matrix, adjustedRandIndex, createRng, makeClassification } from '@sim/core';
import { describe, expect, it } from 'vitest';
import fixture from '../fixtures/mixture.json';
import { GaussianMixture, bicCurve, type CovarianceType } from './mixture';

const x = Matrix.from(fixture.x);

describe('Gaussian mixtures agree with scikit-learn (explicit initial parameters)', () => {
  for (const ref of fixture.fits) {
    it(`${ref.type} covariance: parameters, iteration count, likelihood, BIC and posteriors`, () => {
      const m = new GaussianMixture({
        k: 3,
        covariance: ref.type as CovarianceType,
        tolerance: 1e-3,
        maxIterations: 200,
        init: {
          weights: fixture.weights0,
          means: Matrix.from(fixture.means0),
          covariances: ref.covs0.map((c) => Matrix.from(c)),
        },
      }).fit(x);
      expect(m.iterations).toBe(ref.n_iter);
      expect(m.converged).toBe(ref.converged);
      ref.weights.forEach((v, c) =>
        expect(Math.abs((m.weights[c] as number) - v)).toBeLessThan(1e-8),
      );
      ref.means.forEach((row, c) =>
        row.forEach((v, d) => expect(Math.abs(m.means.get(c, d) - v)).toBeLessThan(1e-7)),
      );
      ref.covariances.forEach((cov, c) =>
        cov.forEach((row, a) =>
          row.forEach((v, b) =>
            expect(Math.abs((m.covariances[c] as Matrix).get(a, b) - v)).toBeLessThan(1e-7),
          ),
        ),
      );
      expect(Math.abs(m.lowerBound - ref.lower_bound)).toBeLessThan(1e-9);
      expect(Math.abs(m.score(x) - ref.score)).toBeLessThan(1e-8);
      expect(Math.abs(m.bic(x) - ref.bic)).toBeLessThan(1e-6);
      expect(Math.abs(m.aic(x) - ref.aic)).toBeLessThan(1e-6);
      const p = m.predictProba(Matrix.from(fixture.x.slice(0, 8)));
      ref.proba.forEach((row, i) =>
        row.forEach((v, c) => expect(Math.abs(p.get(i, c) - v)).toBeLessThan(1e-7)),
      );
      expect(Array.from(m.labels)).toEqual(ref.labels);
    });
  }
});

describe('properties of EM', () => {
  const data = makeClassification('blobs', 300, 0.05, createRng(6));

  it('the likelihood never decreases from one iteration to the next', () => {
    const m = new GaussianMixture({ k: 3, seed: 2, tolerance: 1e-9, maxIterations: 60 });
    const lls = Array.from(m.states(data.x)).map((s) => s.logLikelihood);
    expect(lls.length).toBeGreaterThan(3);
    for (let i = 1; i < lls.length; i++)
      expect(lls[i]).toBeGreaterThanOrEqual((lls[i - 1] as number) - 1e-9);
  });

  it('responsibilities are probabilities that sum to one and weights sum to one', () => {
    const m = new GaussianMixture({ k: 3, seed: 3 }).fit(data.x);
    for (let i = 0; i < 20; i++) {
      let s = 0;
      for (let c = 0; c < 3; c++) s += m.responsibilities.get(i, c);
      expect(s).toBeCloseTo(1, 12);
    }
    expect(Array.from(m.weights).reduce((a, b) => a + b, 0)).toBeCloseTo(1, 12);
  });

  it('recovers well-separated clusters', () => {
    const m = new GaussianMixture({ k: 3, seed: 3 }).fit(data.x);
    expect(adjustedRandIndex(data.y, m.labels)).toBeGreaterThan(0.95);
  });

  it('BIC is lowest near the true number of components', () => {
    const curve = bicCurve(data.x, 6, { seed: 4 });
    const best = curve.indexOf(Math.min(...curve)) + 1;
    expect(best).toBeGreaterThanOrEqual(3);
    expect(best).toBeLessThanOrEqual(4);
  });

  it('a full covariance beats a spherical one on elongated clusters', () => {
    const cig = makeClassification('cigars', 300, 0.1, createRng(8));
    const full = new GaussianMixture({ k: 2, covariance: 'full', seed: 2 }).fit(cig.x);
    const sph = new GaussianMixture({ k: 2, covariance: 'spherical', seed: 2 }).fit(cig.x);
    expect(full.score(cig.x)).toBeGreaterThan(sph.score(cig.x));
  });

  it('samples from the fitted mixture look like the data (means and spread agree)', () => {
    const m = new GaussianMixture({ k: 3, seed: 3 }).fit(data.x);
    const s = m.sample(4000, 5);
    for (let d = 0; d < 2; d++) {
      const mean = (mx: Matrix) =>
        Array.from({ length: mx.rows }, (_, i) => mx.get(i, d)).reduce((a, b) => a + b, 0) /
        mx.rows;
      expect(Math.abs(mean(s) - mean(data.x))).toBeLessThan(0.2);
    }
  });

  it('a soft partition: a point between two clusters is split between them', () => {
    const m = new GaussianMixture({ k: 3, seed: 3 }).fit(data.x);
    const mid = Matrix.from([
      [(m.means.get(0, 0) + m.means.get(1, 0)) / 2, (m.means.get(0, 1) + m.means.get(1, 1)) / 2],
    ]);
    const p = m.predictProba(mid);
    expect(Math.max(p.get(0, 0), p.get(0, 1), p.get(0, 2))).toBeLessThan(0.99);
  });
});
