import { Matrix, createRng } from '@sim/core';
import { describe, expect, it } from 'vitest';
import { LinearRegression } from './linear';
import { Mars, hingeValue, termValue } from './mars';

const col = (v: number[]) => new Matrix(v.length, 1, Float64Array.from(v));

describe('hinge functions', () => {
  it('are zero on one side of the knot and linear on the other', () => {
    expect(hingeValue({ variable: 0, knot: 2, sign: 1 }, 1)).toBe(0);
    expect(hingeValue({ variable: 0, knot: 2, sign: 1 }, 5)).toBe(3);
    expect(hingeValue({ variable: 0, knot: 2, sign: -1 }, 5)).toBe(0);
    expect(hingeValue({ variable: 0, knot: 2, sign: -1 }, 0.5)).toBe(1.5);
    expect(termValue([], [3])).toBe(1);
    expect(
      termValue(
        [
          { variable: 0, knot: 1, sign: 1 },
          { variable: 1, knot: 1, sign: 1 },
        ],
        [3, 4],
      ),
    ).toBe(6);
  });
});

describe('MARS', () => {
  it('recovers a piecewise-linear function with its two knots', () => {
    const x = Array.from({ length: 81 }, (_, i) => i / 80);
    const f = (v: number) => 1 + 3 * Math.max(0, v - 0.25) - 5 * Math.max(0, v - 0.7);
    const m = new Mars({ maxTerms: 9, degree: 1, knots: 80, penalty: 2 }).fit(col(x), x.map(f));
    const pred = m.predict(col(x));
    for (let i = 0; i < x.length; i++) expect(pred[i]).toBeCloseTo(f(x[i]!), 3);
    const k = m.knotsUsed(0);
    expect(k.some((v) => Math.abs(v - 0.25) < 0.02)).toBe(true);
    expect(k.some((v) => Math.abs(v - 0.7) < 0.02)).toBe(true);
  });

  it('prunes by cross-validation: on pure noise the model collapses to (almost) a constant', () => {
    const rng = createRng(2);
    const x = Array.from({ length: 150 }, () => rng.next());
    const y = x.map(() => rng.normal());
    const m = new Mars({ maxTerms: 11, degree: 1, knots: 12, penalty: 3 }).fit(col(x), y);
    expect(m.terms.length).toBeLessThanOrEqual(3);
  });

  it('products of hinges capture an interaction that an additive model misses', () => {
    const rng = createRng(5);
    const n = 300;
    const data = new Matrix(
      n,
      2,
      Float64Array.from({ length: 2 * n }, () => rng.next()),
    );
    const y = Array.from(
      { length: n },
      (_, i) => 6 * Math.max(0, data.data[2 * i]! - 0.4) * Math.max(0, data.data[2 * i + 1]! - 0.4),
    );
    const sse = (m: Mars) => {
      const p = m.predict(data);
      return y.reduce((a, v, i) => a + (v - p[i]!) ** 2, 0);
    };
    const additive = new Mars({ maxTerms: 11, degree: 1, knots: 10, penalty: 2 }).fit(data, y);
    const interacting = new Mars({ maxTerms: 11, degree: 2, knots: 10, penalty: 3 }).fit(data, y);
    expect(sse(interacting)).toBeLessThan(0.3 * sse(additive));
    expect(interacting.terms.some((t) => t.length === 2)).toBe(true);
  });

  it('the forward pass lowers the error at every step and the backward pass records a cross-validation score for every size', () => {
    const rng = createRng(9);
    const x = Array.from({ length: 120 }, () => rng.next() * 4);
    const y = x.map((v) => Math.sin(v) + 0.1 * rng.normal());
    const m = new Mars({ maxTerms: 9, degree: 1, knots: 12, penalty: 2 }).fit(col(x), y);
    for (let i = 1; i < m.forwardPath.length; i++)
      expect(m.forwardPath[i]!.rss).toBeLessThan(m.forwardPath[i - 1]!.rss);
    expect(m.backwardPath[0]!.terms).toBe(m.forwardPath[m.forwardPath.length - 1]!.terms);
    expect(m.backwardPath[m.backwardPath.length - 1]!.terms).toBe(1);
    expect(m.gcv).toBe(Math.min(...m.backwardPath.map((p) => p.gcv)));
  });

  it('a straight-line truth is fitted at least as well as ordinary regression with no extra terms needed to keep the error low', () => {
    const x = Array.from({ length: 60 }, (_, i) => i / 10);
    const y = x.map((v) => 2 + 0.5 * v);
    const m = new Mars().fit(col(x), y);
    const ols = new LinearRegression().fit(col(x), y).predict(col(x));
    const p = m.predict(col(x));
    for (let i = 0; i < x.length; i++) expect(p[i]).toBeCloseTo(ols[i]!, 5);
  });
});
