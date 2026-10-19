import { Matrix, createRng } from '@sim/core';
import { describe, expect, it } from 'vitest';
import { FastICA, kurtosis, sourceRecovery } from './ica';
import { NMF, nmfError, nmfSteps } from './nmf';

/** Three independent, clearly non-Gaussian signals mixed by a fixed matrix. */
function mixtures(n = 800, seed = 3) {
  const rng = createRng(seed);
  const s = new Matrix(n, 3);
  for (let i = 0; i < n; i++) {
    const t = i / 40;
    s.set(i, 0, Math.sin(2 * t)); // sine
    s.set(i, 1, Math.sign(Math.sin(3.3 * t))); // square wave
    s.set(i, 2, rng.next() * 2 - 1); // uniform noise
  }
  const a = [
    [1, 0.5, 0.8],
    [0.6, 1.1, 0.3],
    [0.4, 0.7, 1.2],
  ];
  const x = new Matrix(n, 3);
  for (let i = 0; i < n; i++)
    for (let j = 0; j < 3; j++) {
      let v = 0;
      for (let k = 0; k < 3; k++) v += s.get(i, k) * (a[j] as number[])[k]!;
      x.set(i, j, v);
    }
  return { s, x };
}

describe('FastICA', () => {
  it('unmixes three mixed signals (recovered up to sign, scale and order)', () => {
    const { s, x } = mixtures();
    const ica = new FastICA({ components: 3, seed: 2, tolerance: 1e-8, maxIterations: 500 }).fit(x);
    expect(ica.converged).toBe(true);
    for (const r of sourceRecovery(s, ica.transform(x))) expect(r).toBeGreaterThan(0.99);
  });

  it('PCA alone does not unmix them: its components stay mixed', () => {
    const { s, x } = mixtures();
    const ica = new FastICA({ components: 3, seed: 2, maxIterations: 0 }).fit(x); // whitening only (random rotation)
    const worst = Math.min(...sourceRecovery(s, ica.transform(x)));
    expect(worst).toBeLessThan(0.98);
  });

  it('the whitened data have identity covariance and the unmixing rows are decorrelated', () => {
    const { x } = mixtures();
    const ica = new FastICA({ components: 3, seed: 2 }).fit(x);
    const y = ica.transform(x);
    for (let a = 0; a < 3; a++)
      for (let b = 0; b < 3; b++) {
        let s = 0;
        for (let i = 0; i < y.rows; i++) s += y.get(i, a) * y.get(i, b);
        expect(Math.abs(s / y.rows - (a === b ? 1 : 0))).toBeLessThan(1e-6);
      }
  });

  it('works with each non-linearity and the sources are far from Gaussian', () => {
    const { s, x } = mixtures();
    for (const nonlinearity of ['logcosh', 'exp', 'cube'] as const) {
      const ica = new FastICA({ components: 3, nonlinearity, seed: 4, maxIterations: 500 }).fit(x);
      expect(Math.min(...sourceRecovery(s, ica.transform(x)))).toBeGreaterThan(0.97);
    }
    expect(Math.abs(kurtosis(Array.from({ length: 800 }, (_, i) => s.get(i, 1))))).toBeGreaterThan(
      1,
    );
  });

  it('is reproducible for a seed', () => {
    const { x } = mixtures(300);
    const a = new FastICA({ components: 3, seed: 7 }).fit(x).components;
    const b = new FastICA({ components: 3, seed: 7 }).fit(x).components;
    expect(Array.from(a.data)).toEqual(Array.from(b.data));
  });
});

describe('kurtosis', () => {
  it('is about 0 for Gaussian data, negative for uniform and positive for heavy tails', () => {
    const rng = createRng(1);
    const g = Array.from({ length: 20000 }, () => rng.normal());
    const u = Array.from({ length: 20000 }, () => rng.next());
    const l = Array.from({ length: 20000 }, () => Math.log(rng.next() / rng.next()));
    expect(Math.abs(kurtosis(g))).toBeLessThan(0.15);
    expect(kurtosis(u)).toBeCloseTo(-1.2, 1);
    expect(kurtosis(l)).toBeGreaterThan(1);
  });
});

describe('NMF', () => {
  /** Two "parts" combined with non-negative amounts. */
  const parts = Matrix.from([
    [1, 1, 0, 0, 0, 0],
    [0, 0, 0, 1, 1, 0],
  ]);
  const rng = createRng(5);
  const amounts = Array.from({ length: 40 }, () => [rng.next() * 3, rng.next() * 3]);
  const v = new Matrix(40, 6);
  amounts.forEach((a, i) => {
    for (let j = 0; j < 6; j++)
      v.set(i, j, (a[0] as number) * parts.get(0, j) + (a[1] as number) * parts.get(1, j));
  });

  for (const loss of ['frobenius', 'kl'] as const) {
    it(`${loss}: the error never increases and the factors stay non-negative`, () => {
      const m = new NMF({ components: 2, loss, iterations: 150, seed: 2 }).fit(v);
      for (let i = 1; i < m.history.length; i++)
        expect(m.history[i]).toBeLessThanOrEqual((m.history[i - 1] as number) + 1e-9);
      expect(m.w.data.every((a) => a >= 0)).toBe(true);
      expect(m.h.data.every((a) => a >= 0)).toBe(true);
      expect(m.error).toBeCloseTo(nmfError(v, m.w, m.h, loss), 9);
    });
  }

  it('recovers a rank-2 non-negative table almost exactly', () => {
    const m = new NMF({ components: 2, iterations: 600, seed: 3 }).fit(v);
    expect(m.error / (40 * 6)).toBeLessThan(1e-3);
  });

  it('finds the parts: each row of H is concentrated on one of the two groups of columns', () => {
    const m = new NMF({ components: 2, iterations: 600, seed: 3 }).fit(v);
    const shares = [0, 1].map((r) => {
      const left = m.h.get(r, 0) + m.h.get(r, 1);
      const right = m.h.get(r, 3) + m.h.get(r, 4);
      return Math.max(left, right) / (left + right + 1e-12);
    });
    for (const s of shares) expect(s).toBeGreaterThan(0.85);
  });

  it('one component cannot explain two independent parts', () => {
    const one = new NMF({ components: 1, iterations: 300, seed: 3 }).fit(v);
    const two = new NMF({ components: 2, iterations: 300, seed: 3 }).fit(v);
    expect(one.error).toBeGreaterThan(10 * two.error);
  });

  it('streams its iterations', () => {
    const states = Array.from(nmfSteps(v, { components: 2, iterations: 5, seed: 1 }));
    expect(states.map((s) => s.iteration)).toEqual([0, 1, 2, 3, 4, 5]);
  });
});
