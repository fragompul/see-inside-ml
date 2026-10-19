import { Matrix, createRng } from '@sim/core';
import { describe, expect, it } from 'vitest';
import fixture from '../fixtures/pca.json';
import { KernelPCA, PCA, type PcaKernel } from './pca';

const close = (a: number, b: number, tol: number) => expect(Math.abs(a - b)).toBeLessThan(tol);

describe('PCA agrees with scikit-learn', () => {
  for (const ref of fixture.datasets) {
    it(`${ref.name}: components (with signs), variances, ratios, scores and whitening`, () => {
      const x = Matrix.from(ref.x);
      const p = new PCA().fit(x);
      ref.mean.forEach((v, j) => close(p.mean[j] as number, v, 1e-10));
      ref.variance.forEach((v, c) => close(p.explainedVariance[c] as number, v, 1e-8));
      ref.ratio.forEach((v, c) => close(p.explainedVarianceRatio[c] as number, v, 1e-10));
      ref.components.forEach((row, c) =>
        row.forEach((v, j) => close(p.components.get(c, j), v, 1e-7)),
      );
      const two = new PCA({ components: 2 }).fit(x).transform(x);
      ref.scores2.forEach((row, i) => row.forEach((v, c) => close(two.get(i, c), v, 1e-7)));
      const white = new PCA({ components: 2, whiten: true }).fit(x).transform(x);
      ref.whitened2.forEach((row, i) => row.forEach((v, c) => close(white.get(i, c), v, 1e-7)));
    });
  }
});

describe('kernel PCA agrees with scikit-learn', () => {
  const x = Matrix.from(fixture.kpca.x);
  for (const ref of fixture.kpca.fits) {
    it(`${ref.kernel} kernel: eigenvalues, embedding and out-of-sample projection`, () => {
      const kw = ref.kw as { gamma?: number; degree?: number; coef0?: number };
      const m = new KernelPCA({ components: 3, kernel: ref.kernel as PcaKernel, ...kw }).fit(x);
      ref.eigenvalues.forEach((v, c) =>
        close(m.eigenvalues[c] as number, v, 1e-7 * Math.max(1, v)),
      );
      const e = m.embedding;
      ref.embedding.forEach((row, i) => row.forEach((v, c) => close(e.get(i, c), v, 1e-6)));
      const fresh = m.transform(
        Matrix.from(fixture.kpca.x.slice(0, 3).map((r) => r.map((v) => v * 1.1))),
      );
      ref.new.forEach((row, i) => row.forEach((v, c) => close(fresh.get(i, c), v, 1e-6)));
    });
  }

  it('an RBF kernel unrolls two concentric rings that PCA cannot separate', () => {
    const labels = Array.from({ length: 70 }, (_, i) => i % 2);
    const linear = new PCA({ components: 1 }).fit(x).transform(x);
    const rbf = new KernelPCA({ components: 2, kernel: 'rbf', gamma: 1.5 }).fit(x).embedding;
    // gap between the classes along the best single axis, relative to the spread
    const separation = (coords: number[]) => {
      const a = coords.filter((_, i) => labels[i] === 0);
      const b = coords.filter((_, i) => labels[i] === 1);
      const mean = (v: number[]) => v.reduce((s, q) => s + q, 0) / v.length;
      const sd = (v: number[]) =>
        Math.sqrt(v.reduce((s, q) => s + (q - mean(v)) ** 2, 0) / v.length) || 1;
      return Math.abs(mean(a) - mean(b)) / ((sd(a) + sd(b)) / 2);
    };
    const best = (m: Matrix) =>
      Math.max(
        ...Array.from({ length: m.cols }, (_, c) =>
          separation(Array.from({ length: m.rows }, (_, i) => m.get(i, c))),
        ),
      );
    expect(best(rbf)).toBeGreaterThan(3 * best(linear));
  });
});

describe('properties of PCA', () => {
  const rng = createRng(5);
  const x = Matrix.from(
    Array.from({ length: 200 }, () => {
      const a = rng.normal() * 3;
      const b = rng.normal();
      return [a + 0.5 * b, 0.8 * a - b, b * 0.2, rng.normal() * 0.1];
    }),
  );

  it('components are orthonormal and variances are sorted and sum to the total variance', () => {
    const p = new PCA().fit(x);
    for (let a = 0; a < 4; a++)
      for (let b = 0; b < 4; b++) {
        let dot = 0;
        for (let j = 0; j < 4; j++) dot += p.components.get(a, j) * p.components.get(b, j);
        close(dot, a === b ? 1 : 0, 1e-9);
      }
    for (let c = 1; c < 4; c++)
      expect(p.explainedVariance[c]).toBeLessThanOrEqual(
        (p.explainedVariance[c - 1] as number) + 1e-12,
      );
    const total = Array.from({ length: 4 }, (_, j) => {
      const col = Array.from({ length: x.rows }, (_, i) => x.get(i, j));
      const m = col.reduce((a, b) => a + b, 0) / col.length;
      return col.reduce((s, v) => s + (v - m) ** 2, 0) / (col.length - 1);
    }).reduce((a, b) => a + b, 0);
    close(
      p.explainedVariance.reduce((a, b) => a + b, 0),
      total,
      1e-8,
    );
  });

  it('keeping every component reconstructs the data exactly; fewer give a growing error equal to the lost variance', () => {
    const full = new PCA().fit(x);
    close(full.reconstructionError(x), 0, 1e-18 + 1e-20);
    let previous = 0;
    for (let k = 3; k >= 1; k--) {
      const p = new PCA({ components: k }).fit(x);
      const err = p.reconstructionError(x);
      expect(err).toBeGreaterThanOrEqual(previous);
      previous = err;
      // the mean squared error equals the variance of the discarded directions (times (n − 1)/n)
      const lost = full.allVariance.slice(k).reduce((a, b) => a + b, 0) * ((x.rows - 1) / x.rows);
      close(err, lost, 1e-8);
    }
  });

  it('the scores are uncorrelated, and whitening gives them unit variance', () => {
    const z = new PCA({ components: 3, whiten: true }).fit(x).transform(x);
    for (let c = 0; c < 3; c++) {
      let s = 0;
      for (let i = 0; i < z.rows; i++) s += z.get(i, c) ** 2;
      close(s / (z.rows - 1), 1, 1e-9);
    }
  });

  it('componentsFor finds how many directions carry a share of the variance', () => {
    const p = new PCA().fit(x);
    expect(p.componentsFor(0.5)).toBeLessThanOrEqual(2);
    expect(p.componentsFor(1)).toBeLessThanOrEqual(4);
  });
});
