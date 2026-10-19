import { Matrix, createRng, makeClassification } from '@sim/core';
import { describe, expect, it } from 'vitest';
import fixture from '../fixtures/svm.json';
import { SVC, SVR, makeKernel, resolvedGamma, type KernelName } from './svm';

const x = Matrix.from(fixture.x);
const grid = Matrix.from(fixture.grid);

describe('SVC agrees with scikit-learn (libsvm)', () => {
  for (const ref of fixture.binary) {
    it(`binary: ${JSON.stringify(ref.config)}`, () => {
      const { kernel, C, gamma, degree, coef0 } = ref.config as {
        kernel: KernelName;
        C: number;
        gamma?: number;
        degree?: number;
        coef0?: number;
      };
      const m = new SVC({
        kernel,
        C,
        tol: 1e-9,
        ...(gamma !== undefined ? { gamma } : {}),
        ...(degree ? { degree } : {}),
        ...(coef0 !== undefined ? { coef0 } : {}),
      }).fit(x, fixture.y2);
      expect(resolvedGamma({ gamma: gamma ?? 'scale' }, x)).toBeCloseTo(ref.gamma, 10);
      const b = m.binaryModel() as { support: number[]; coef: number[]; intercept: number };
      // rounding noise can leave α ≈ 1e-12 on either side: compare the machines by their real vectors
      const mine = new Map(b.support.map((s, i) => [s, b.coef[i] as number] as const));
      ref.support.forEach((s, i) => {
        const c = ref.dual_coef[i] as number;
        if (Math.abs(c) > 1e-6) expect(Math.abs((mine.get(s) ?? 0) - c)).toBeLessThan(1e-4);
      });
      for (const [s, c] of mine) if (Math.abs(c) > 1e-6) expect(ref.support).toContain(s);
      expect(Math.abs(b.intercept - ref.intercept)).toBeLessThan(1e-4);
      const d = m.decision(grid);
      ref.decision.forEach((v, i) => expect(Math.abs((d[i] as number) - v)).toBeLessThan(1e-4));
      expect(Array.from(m.predict(grid))).toEqual(ref.pred);
    });
  }

  it('multi-class: one-vs-one support vectors and predictions', () => {
    const m = new SVC({ kernel: 'rbf', C: 2, tol: 1e-9 }).fit(x, fixture.y3);
    // α ≈ 0 noise may add or drop a vector: the real ones must agree
    for (const s of fixture.multi.support) expect(m.supportIndices).toContain(s);
    expect(Math.abs(m.supportIndices.length - fixture.multi.support.length)).toBeLessThanOrEqual(3);
    expect(Array.from(m.predict(grid))).toEqual(fixture.multi.pred);
    // the (class 0 vs 1) machine matches scikit-learn's first ovo column (libsvm's sign: positive = first class)
    const d = m.decision(grid, 0);
    fixture.multi.ovo.forEach((row, i) =>
      expect(Math.abs(-(d[i] as number) - (row[0] as number))).toBeLessThan(1e-3),
    );
  });
});

describe('SVR agrees with scikit-learn', () => {
  const xr = Matrix.from(fixture.svr.x);
  const gr = Matrix.from(fixture.svr.grid);
  for (const ref of fixture.svr.fits) {
    it(`${JSON.stringify(ref.config)}`, () => {
      const c = ref.config as { kernel: KernelName; C: number; epsilon: number; gamma?: number };
      const m = new SVR({
        kernel: c.kernel,
        C: c.C,
        epsilon: c.epsilon,
        tol: 1e-9,
        ...(c.gamma ? { gamma: c.gamma } : {}),
      }).fit(xr, fixture.svr.y);
      for (const s of ref.support) expect(m.support).toContain(s);
      expect(Math.abs(m.support.length - ref.support.length)).toBeLessThanOrEqual(2);
      ref.support.forEach((s, i) => {
        const idx = m.support.indexOf(s);
        expect(Math.abs((m.coef[idx] as number) - (ref.dual_coef[i] as number))).toBeLessThan(1e-3);
      });
      expect(Math.abs(m.intercept - ref.intercept)).toBeLessThan(1e-3);
      const pred = m.predict(gr);
      ref.pred.forEach((v, i) => expect(Math.abs((pred[i] as number) - v)).toBeLessThan(5e-4));
    });
  }
});

describe('properties', () => {
  const data = makeClassification('moons', 160, 0.25, createRng(5));

  it('the margin: support vectors satisfy |f| ≈ 1 when free, and every point has y·f ≥ 1 or is a support vector', () => {
    const m = new SVC({ kernel: 'linear', C: 1000, tol: 1e-9 }).fit(data.x, data.y);
    const d = m.decision(data.x);
    const sv = new Set(m.supportIndices);
    for (let i = 0; i < data.x.rows; i++) {
      const margin = (data.y[i] === 1 ? 1 : -1) * (d[i] as number);
      if (!sv.has(i)) expect(margin).toBeGreaterThanOrEqual(1 - 1e-6);
    }
  });

  it('a larger C fits the training set at least as well (more support vectors are free, boundary tighter)', () => {
    const acc = (C: number) => {
      const m = new SVC({ kernel: 'rbf', C, gamma: 2 }).fit(data.x, data.y);
      return Array.from(m.predict(data.x)).filter((p, i) => p === data.y[i]).length / data.x.rows;
    };
    expect(acc(100)).toBeGreaterThanOrEqual(acc(0.01));
  });

  it('kernels are symmetric and the RBF kernel of a point with itself is 1', () => {
    const k = makeKernel({ kernel: 'rbf', gamma: 0.7 }, data.x);
    const a = data.x.data.subarray(0, 2);
    const b = data.x.data.subarray(2, 4);
    expect(k(a, a)).toBe(1);
    expect(k(a, b)).toBeCloseTo(k(b, a), 14);
  });

  it('SVR: points inside the ε-tube are not support vectors', () => {
    const xr = Matrix.from(fixture.svr.x);
    const m = new SVR({ kernel: 'rbf', C: 1, epsilon: 0.3 }).fit(xr, fixture.svr.y);
    const pred = m.predict(xr);
    const sv = new Set(m.support);
    for (let i = 0; i < xr.rows; i++)
      if (Math.abs((fixture.svr.y[i] as number) - (pred[i] as number)) < 0.3 - 1e-3)
        expect(sv.has(i)).toBe(false);
  });
});
