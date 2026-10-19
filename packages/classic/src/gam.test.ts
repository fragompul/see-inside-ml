import { Matrix, createRng } from '@sim/core';
import { describe, expect, it } from 'vitest';
import { AdditiveModel, PSpline, bsplineBasis, differencePenalty } from './gam';
import { LinearRegression } from './linear';

const col = (v: number[]) => new Matrix(v.length, 1, Float64Array.from(v));

describe('B-spline basis', () => {
  it('the bumps add up to one at every point and are never negative', () => {
    const x = Array.from({ length: 101 }, (_, i) => i / 100);
    const b = bsplineBasis(x, 0, 1, 9);
    for (let r = 0; r < x.length; r++) {
      let sum = 0;
      for (let k = 0; k < 9; k++) {
        expect(b.data[r * 9 + k]!).toBeGreaterThanOrEqual(0);
        sum += b.data[r * 9 + k]!;
      }
      expect(sum).toBeCloseTo(1, 12);
    }
  });

  it('a cubic spline reproduces a straight line exactly (coefficients along the line)', () => {
    const x = Array.from({ length: 50 }, (_, i) => i / 49);
    const b = bsplineBasis(x, 0, 1, 8);
    // Greville abscissae: for equally spaced knots the coefficients of the line f(x) = x are the knot averages
    const inner = 8 - 3;
    const knots = Array.from({ length: 8 + 4 }, (_, i) => (i - 3) / inner);
    const coef = Array.from(
      { length: 8 },
      (_, k) => (knots[k + 1]! + knots[k + 2]! + knots[k + 3]!) / 3,
    );
    x.forEach((v, r) => {
      let s = 0;
      for (let k = 0; k < 8; k++) s += b.data[r * 8 + k]! * coef[k]!;
      expect(s).toBeCloseTo(v, 10);
    });
  });

  it('the penalty is zero for a straight line of coefficients and positive for a bend', () => {
    const p = differencePenalty(6);
    const quad = (c: number[]) =>
      c.reduce((a, ci, i) => a + ci * c.reduce((s, cj, j) => s + p.data[i * 6 + j]! * cj, 0), 0);
    expect(quad([1, 2, 3, 4, 5, 6])).toBeCloseTo(0, 12);
    expect(quad([0, 0, 1, 0, 0, 0])).toBeGreaterThan(0);
  });
});

describe('P-spline', () => {
  const rng = createRng(3);
  const x = Array.from({ length: 120 }, () => rng.next() * 6);
  const y = x.map((v) => Math.sin(v) * 2 + 0.3 * rng.normal());

  it('a huge penalty leaves the straight-line least-squares fit', () => {
    const line = new LinearRegression().fit(col(x), y).predict(col(x));
    const smooth = new PSpline({ bumps: 12, lambda: 1e9 }).fit(col(x), y).predict(col(x));
    for (let i = 0; i < x.length; i++) expect(smooth[i]).toBeCloseTo(line[i]!, 3);
  });

  it('the effective degrees of freedom fall from the number of bumps towards 2 as the penalty grows', () => {
    const df = [1e-6, 1e-2, 1, 100, 1e4, 1e9].map(
      (lambda) => new PSpline({ bumps: 12, lambda }).fit(col(x), y).df,
    );
    expect(df[0]!).toBeGreaterThan(11.5);
    for (let i = 1; i < df.length; i++) expect(df[i]!).toBeLessThan(df[i - 1]! + 1e-9);
    expect(df[df.length - 1]!).toBeCloseTo(2, 2);
  });

  it('cross-validation prefers a moderate penalty: wiggles and straight lines both score worse', () => {
    const gcv = (lambda: number) => new PSpline({ bumps: 15, lambda }).fit(col(x), y).gcv;
    expect(gcv(1)).toBeLessThan(gcv(1e-6));
    expect(gcv(1)).toBeLessThan(gcv(1e9));
  });
});

describe('additive model', () => {
  const rng = createRng(8);
  const n = 400;
  const f1 = (v: number) => Math.sin(2 * v);
  const f2 = (v: number) => v * v - 1;
  const data = new Matrix(
    n,
    2,
    Float64Array.from({ length: 2 * n }, () => rng.uniform(-2, 2)),
  );
  const y = Array.from(
    { length: n },
    (_, i) => 1 + f1(data.data[2 * i]!) + f2(data.data[2 * i + 1]!) + 0.2 * rng.normal(),
  );

  it('recovers the shape of each input up to a constant', () => {
    const m = new AdditiveModel({ bumps: 12, lambda: 1, iterations: 100 }).fit(data, y);
    const grid = Array.from({ length: 41 }, (_, i) => -2 + i / 10);
    const corr = (a: Float64Array, b: number[]) => {
      const ma = a.reduce((s, v) => s + v, 0) / a.length;
      const mb = b.reduce((s, v) => s + v, 0) / b.length;
      let s = 0,
        sa = 0,
        sb = 0;
      for (let i = 0; i < a.length; i++) {
        s += (a[i]! - ma) * (b[i]! - mb);
        sa += (a[i]! - ma) ** 2;
        sb += (b[i]! - mb) ** 2;
      }
      return s / Math.sqrt(sa * sb);
    };
    expect(corr(m.shape(0, grid), grid.map(f1))).toBeGreaterThan(0.98);
    expect(corr(m.shape(1, grid), grid.map(f2))).toBeGreaterThan(0.98);
    const pred = m.predict(data);
    const rmse = Math.sqrt(y.reduce((a, v, i) => a + (v - pred[i]!) ** 2, 0) / n);
    expect(rmse).toBeLessThan(0.3);
  });

  it('with one input it is the same as a single centred P-spline', () => {
    const one = new Matrix(
      n,
      1,
      Float64Array.from({ length: n }, (_, i) => data.data[2 * i]!),
    );
    const am = new AdditiveModel({ bumps: 10, lambda: 3, iterations: 20 }).fit(one, y);
    const ps = new PSpline({ bumps: 10, lambda: 3 }).fit(
      one,
      y.map((v) => v - y.reduce((a, b) => a + b, 0) / n),
    );
    const a = am.predict(one);
    const b = ps.predict(one);
    const shift = a[0]! - b[0]!;
    for (let i = 0; i < n; i++) expect(a[i]! - b[i]!).toBeCloseTo(shift, 6);
  });

  it('shape functions have zero mean over the training data', () => {
    const m = new AdditiveModel().fit(data, y);
    const col0 = Array.from({ length: n }, (_, i) => data.data[2 * i]!);
    const f = m.shape(0, col0);
    expect(f.reduce((a, v) => a + v, 0) / n).toBeCloseTo(0, 6);
  });
});
