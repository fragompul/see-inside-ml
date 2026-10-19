import { Matrix, createRng } from '@sim/core';
import { describe, expect, it } from 'vitest';
import fx from '../fixtures/isotonic.json';
import { IsotonicRegression, pava } from './isotonic';

const col = (v: number[]) => new Matrix(v.length, 1, Float64Array.from(v));

describe('isotonic regression', () => {
  it.each([
    ['increasing', true, fx.up],
    ['decreasing', false, fx.down],
  ] as const)('matches scikit-learn (%s)', (_name, increasing, ref) => {
    const m = new IsotonicRegression({ increasing }).fit(col(fx.x), ref.y);
    const fitted = m.predict(col(fx.x));
    ref.fitted.forEach((v, i) => expect(fitted[i]).toBeCloseTo(v, 9));
    const grid = m.predict(col(fx.grid));
    ref.grid.forEach((v, i) => expect(grid[i]).toBeCloseTo(v, 9));
  });

  it('the fit is monotone and is the least-squares projection: no monotone sequence is closer', () => {
    const rng = createRng(4);
    const x = Array.from({ length: 40 }, (_, i) => i);
    const y = x.map((v) => Math.sin(v / 6) * 3 + rng.normal());
    const m = new IsotonicRegression().fit(col(x), y);
    const f = Array.from(m.predict(col(x)));
    for (let i = 1; i < f.length; i++) expect(f[i]!).toBeGreaterThanOrEqual(f[i - 1]! - 1e-12);
    const sse = (g: number[]) => g.reduce((a, v, i) => a + (v - y[i]!) ** 2, 0);
    // perturbed monotone candidates never beat it
    for (let t = 0; t < 200; t++) {
      const cand = [...f].map((v) => v + 0.3 * rng.normal()).sort((a, b) => a - b);
      expect(sse(cand)).toBeGreaterThanOrEqual(sse(f) - 1e-9);
    }
  });

  it('already monotone data is reproduced exactly, and a block takes the weighted mean of its points', () => {
    const m = new IsotonicRegression().fit(col([1, 2, 3, 4]), [1, 2, 3, 4]);
    expect(Array.from(m.predict(col([1, 2, 3, 4])))).toEqual([1, 2, 3, 4]);
    expect(m.pieces).toBe(4);
    const pooled = pava([1, 2, 3], [5, 1, 2], [1, 1, 2]);
    expect(pooled.blocks).toHaveLength(1);
    expect(pooled.blocks[0]!.value).toBeCloseTo((5 + 1 + 2 * 2) / 4, 12);
  });

  it('keeps one step per run of pooled points and records every stage for stepping through', () => {
    const m = new IsotonicRegression().fit(col([1, 2, 3, 4, 5]), [1, 3, 2, 2, 5]);
    expect(m.blocks.map((b) => b.value)).toEqual(
      [1, 7 / 3, 7 / 3, 7 / 3, 5].filter((v, i, a) => a.indexOf(v) === i),
    );
    expect(m.steps).toHaveLength(5);
  });

  it('clips outside the data and refuses more than one input', () => {
    const m = new IsotonicRegression().fit(col([1, 2, 3]), [1, 2, 3]);
    expect(m.predictOne(-10)).toBe(1);
    expect(m.predictOne(10)).toBe(3);
    expect(() => new IsotonicRegression().fit(new Matrix(3, 2), [1, 2, 3])).toThrow();
  });
});
