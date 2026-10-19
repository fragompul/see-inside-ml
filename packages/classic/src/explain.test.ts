import { Matrix, createRng } from '@sim/core';
import { describe, expect, it } from 'vitest';
import { instanceOf, kernelShap, lime, shapleyExact, type Model } from './explain';

const rng = createRng(3);
const n = 80;
const p = 4;
const data = new Matrix(
  n,
  p,
  Float64Array.from({ length: n * p }, () => rng.normal()),
);
const columnMean = (j: number) =>
  Array.from({ length: n }, (_, i) => data.get(i, j)).reduce((a, b) => a + b, 0) / n;

const linear =
  (w: number[], b = 0): Model =>
  (x) =>
    Float64Array.from(
      { length: x.rows },
      (_, i) => b + w.reduce((s, wj, j) => s + wj * x.get(i, j), 0),
    );

describe('exact Shapley values', () => {
  const instance = [0.5, -1, 2, 0.3];

  it('for a linear model each contribution is the weight times how far the feature is from its average', () => {
    const w = [2, -1, 0.5, 0];
    const phi = shapleyExact(linear(w, 3), instance, data);
    w.forEach((wj, j) => expect(phi.values[j]).toBeCloseTo(wj * (instance[j]! - columnMean(j)), 9));
  });

  it('efficiency: the contributions add up to the prediction minus the base value', () => {
    const f: Model = (x) =>
      Float64Array.from(
        { length: x.rows },
        (_, i) => x.get(i, 0) * x.get(i, 1) + Math.sin(x.get(i, 2)) + x.get(i, 3) ** 2,
      );
    const phi = shapleyExact(f, instance, data);
    expect(phi.values.reduce((a, b) => a + b, 0)).toBeCloseTo(phi.prediction - phi.base, 9);
    expect(phi.prediction).toBeCloseTo(f(new Matrix(1, p, Float64Array.from(instance)))[0]!, 9);
  });

  it('dummy: a feature the model ignores gets exactly zero', () => {
    const f = linear([1, 2, 0, -1]);
    expect(shapleyExact(f, instance, data).values[2]).toBeCloseTo(0, 12);
  });

  it('symmetry: two features that play the same role get the same contribution', () => {
    // f depends on x0 + x1 only; give the instance and the background the same statistics for them
    const sym = new Matrix(60, 2);
    const r = createRng(5);
    for (let i = 0; i < 60; i++) {
      const v = r.normal();
      sym.set(i, 0, v);
      sym.set(i, 1, v);
    }
    const f: Model = (x) =>
      Float64Array.from({ length: x.rows }, (_, i) => (x.get(i, 0) + x.get(i, 1)) ** 2);
    const phi = shapleyExact(f, [1.2, 1.2], sym);
    expect(phi.values[0]).toBeCloseTo(phi.values[1]!, 9);
  });

  it('additivity: the values of a sum of two models are the sums of the values', () => {
    const f1 = linear([1, 0, 2, 0]);
    const f2: Model = (x) =>
      Float64Array.from({ length: x.rows }, (_, i) => x.get(i, 1) * x.get(i, 3));
    const both: Model = (x) => {
      const a = f1(x);
      const b = f2(x);
      return Float64Array.from({ length: x.rows }, (_, i) => (a[i] as number) + (b[i] as number));
    };
    const a = shapleyExact(f1, instance, data).values;
    const b = shapleyExact(f2, instance, data).values;
    const c = shapleyExact(both, instance, data).values;
    c.forEach((v, j) => expect(v).toBeCloseTo(a[j]! + b[j]!, 9));
  });

  it('an interaction is shared fairly between the two features that make it', () => {
    const f: Model = (x) =>
      Float64Array.from({ length: x.rows }, (_, i) => x.get(i, 0) * x.get(i, 1));
    const zero = new Matrix(10, 2); // background all zeros: f is 0 unless both features are known
    const phi = shapleyExact(f, [2, 3], zero);
    expect(phi.values[0]).toBeCloseTo(3, 9);
    expect(phi.values[1]).toBeCloseTo(3, 9);
  });
});

describe('KernelSHAP', () => {
  const instance = [0.5, -1, 2, 0.3];
  const f: Model = (x) =>
    Float64Array.from(
      { length: x.rows },
      (_, i) => x.get(i, 0) * x.get(i, 1) + 2 * x.get(i, 2) - x.get(i, 3) ** 2,
    );
  const exact = shapleyExact(f, instance, data).values;

  it('with enough samples it recovers the exact values and respects efficiency exactly', () => {
    const k = kernelShap(f, instance, data, 600, 1);
    k.values.forEach((v, j) => expect(v).toBeCloseTo(exact[j]!, 6));
    expect(k.values.reduce((a, b) => a + b, 0)).toBeCloseTo(k.prediction - k.base, 9);
  });

  it('with few samples it is approximate but still satisfies efficiency', () => {
    const k = kernelShap(f, instance, data, 6, 2);
    expect(k.values.reduce((a, b) => a + b, 0)).toBeCloseTo(k.prediction - k.base, 9);
  });
});

describe('LIME', () => {
  it('recovers the weights of a linear model', () => {
    const w = [1.5, -2, 0.5, 0];
    const fit = lime(linear(w, 1), [0.2, 0.1, -0.4, 0.8], data, {
      samples: 800,
      width: 0.5,
      kernelWidth: 1.5,
      ridge: 1e-6,
      seed: 1,
    });
    w.forEach((wj, j) => expect(fit.coefficients[j]).toBeCloseTo(wj, 1));
    expect(fit.fidelity).toBeGreaterThan(0.999);
  });

  it('describes the local slope of a curved model, which differs from place to place', () => {
    const f: Model = (x) => Float64Array.from({ length: x.rows }, (_, i) => x.get(i, 0) ** 2);
    const at = (v: number) =>
      lime(f, [v, 0, 0, 0], data, {
        samples: 1500,
        width: 0.15,
        kernelWidth: 1,
        ridge: 1e-6,
        seed: 2,
      }).coefficients[0]!;
    expect(at(1.5)).toBeCloseTo(3, 0);
    expect(at(-1.5)).toBeCloseTo(-3, 0);
  });

  it('instanceOf reads one row', () => {
    expect(instanceOf(data, 0)).toHaveLength(p);
  });
});
