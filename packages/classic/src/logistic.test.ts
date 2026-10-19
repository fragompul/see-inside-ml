import { Matrix, accuracy } from '@sim/core';
import { describe, expect, it } from 'vitest';
import fixture from '../fixtures/logistic.json';
import { LogisticRegression, binaryLogLoss, logisticGradientDescent } from './logistic';

const near = (a: number, b: number, tol: number) =>
  expect(Math.abs(a - b)).toBeLessThanOrEqual(tol * Math.max(1, Math.abs(b)));

describe('binary logistic regression agrees with scikit-learn', () => {
  const x = Matrix.from(fixture.x);
  for (const ref of fixture.binary) {
    it(`C = ${ref.C}`, () => {
      const m = new LogisticRegression({ C: ref.C }).fit(x, fixture.y);
      ref.coef.forEach((w, j) => near(m.weights.get(0, j), w, 1e-6));
      near(m.intercepts[0] as number, ref.intercept, 1e-6);
      const p = m.predictProba(Matrix.from(fixture.x.slice(0, 5)));
      ref.proba.forEach((row, i) => row.forEach((v, c) => near(p.get(i, c), v, 1e-6)));
    });
  }
});

describe('multinomial logistic regression agrees with scikit-learn', () => {
  const x = Matrix.from(fixture.multi.x);
  for (const ref of fixture.multi.fits) {
    it(`C = ${ref.C}`, () => {
      const m = new LogisticRegression({ C: ref.C }).fit(x, fixture.multi.y);
      expect(m.classes).toBe(3);
      ref.coef.forEach((row, c) => row.forEach((w, j) => near(m.weights.get(c, j), w, 1e-4)));
      ref.intercept.forEach((b, c) => near(m.intercepts[c] as number, b, 1e-4));
      const p = m.predictProba(Matrix.from(fixture.multi.x.slice(0, 6)));
      ref.proba.forEach((row, i) => row.forEach((v, c) => near(p.get(i, c), v, 1e-4)));
      expect(Array.from(m.predict(x))).toEqual(ref.pred);
    });
  }
});

describe('properties', () => {
  const x = Matrix.from(fixture.x);

  it('probabilities sum to one and predictions are the most probable class', () => {
    const m = new LogisticRegression().fit(x, fixture.y);
    const p = m.predictProba(x);
    const pred = m.predict(x);
    for (let i = 0; i < x.rows; i++) {
      expect(p.get(i, 0) + p.get(i, 1)).toBeCloseTo(1, 12);
      expect(pred[i]).toBe(p.get(i, 1) > 0.5 ? 1 : 0);
    }
  });

  it('a smaller C (stronger penalty) shrinks the weights', () => {
    const norm = (c: number) => {
      const m = new LogisticRegression({ C: c }).fit(x, fixture.y);
      return Math.hypot(...m.weights.data);
    };
    expect(norm(0.01)).toBeLessThan(norm(1));
    expect(norm(1)).toBeLessThan(norm(100));
  });

  it('gradient descent heads for the Newton solution and the loss never rises', () => {
    const newton = new LogisticRegression({ C: Infinity }).fit(x, fixture.y);
    const target = binaryLogLoss(
      x,
      fixture.y,
      Array.from(newton.weights.data),
      newton.intercepts[0] as number,
    );
    const states = Array.from(logisticGradientDescent(x, fixture.y, { lr: 0.5, steps: 4000 }));
    for (let k = 1; k < states.length; k++)
      expect(states[k]!.loss).toBeLessThanOrEqual(states[k - 1]!.loss + 1e-12);
    const last = states[states.length - 1]!;
    expect(last.loss - target).toBeLessThan(1e-4);
    newton.weights.data.forEach((w, j) =>
      expect(Math.abs((last.weights[j] as number) - w)).toBeLessThan(0.05),
    );
  });

  it('separable data: a weak penalty still gives a finite, accurate model', () => {
    const sx = Matrix.from([[-2], [-1], [1], [2]]);
    const m = new LogisticRegression({ C: 10 }).fit(sx, [0, 0, 1, 1]);
    expect(accuracy([0, 0, 1, 1], m.predict(sx))).toBe(1);
    expect(Number.isFinite(m.weights.get(0, 0))).toBe(true);
  });
});
