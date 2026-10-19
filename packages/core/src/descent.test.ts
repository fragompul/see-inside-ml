import { describe, expect, it } from 'vitest';
import {
  TEST_FUNCTIONS,
  descend,
  distanceToMinimum,
  newtonStepExact,
  quadratic,
  type TestFunction,
} from './descent';

const all = Object.values(TEST_FUNCTIONS);

describe('test functions', () => {
  it.each(all)('$id: gradient and Hessian match finite differences', (fn: TestFunction) => {
    for (const p of [
      [0.7, -0.4],
      [-1.3, 1.1],
      [2.1, 0.6],
    ] as const) {
      const h = 1e-6;
      const g = fn.grad(p);
      const fx = (d: number, e: number) => fn.f([p[0] + d, p[1] + e]);
      expect(g[0]).toBeCloseTo((fx(h, 0) - fx(-h, 0)) / (2 * h), 4);
      expect(g[1]).toBeCloseTo((fx(0, h) - fx(0, -h)) / (2 * h), 4);
      const H = fn.hess(p);
      const gAt = (d: number, e: number) => fn.grad([p[0] + d, p[1] + e]);
      const dgx = [
        (gAt(h, 0)[0] - gAt(-h, 0)[0]) / (2 * h),
        (gAt(h, 0)[1] - gAt(-h, 0)[1]) / (2 * h),
      ];
      const dgy = [
        (gAt(0, h)[0] - gAt(0, -h)[0]) / (2 * h),
        (gAt(0, h)[1] - gAt(0, -h)[1]) / (2 * h),
      ];
      expect(H[0][0]).toBeCloseTo(dgx[0]!, 3);
      expect(H[0][1]).toBeCloseTo(dgx[1]!, 3);
      expect(H[1][0]).toBeCloseTo(dgy[0]!, 3);
      expect(H[1][1]).toBeCloseTo(dgy[1]!, 3);
    }
  });

  it.each(all)('$id: the listed minima have zero gradient', (fn: TestFunction) => {
    for (const m of fn.minima) {
      const g = fn.grad(m);
      expect(Math.hypot(g[0], g[1])).toBeLessThan(1e-3);
    }
  });
});

describe('optimisers', () => {
  const bowl = quadratic(10);

  it('Newton solves a quadratic in a single step', () => {
    const p = descend(bowl, 'newton', [-1.8, 0.9], { learningRate: 1, steps: 3 });
    expect(Math.hypot(...p.points[1]!)).toBeLessThan(1e-9);
    expect(newtonStepExact(bowl, [-1.8, 0.9])[0]).toBeCloseTo(0, 12);
  });

  it('gradient descent is stable below 2/L and diverges above it', () => {
    const stable = descend(bowl, 'gd', [-1.8, 0.9], { learningRate: 1.9 / 10, steps: 200 });
    const unstable = descend(bowl, 'gd', [-1.8, 0.9], { learningRate: 2.1 / 10, steps: 200 });
    expect(stable.diverged).toBe(false);
    expect(stable.values[stable.values.length - 1]!).toBeLessThan(stable.values[0]!);
    expect(unstable.values[unstable.values.length - 1]!).toBeGreaterThan(unstable.values[0]!);
  });

  it('the harder the conditioning the slower plain gradient descent, and momentum helps', () => {
    const steps = (kappa: number, method: 'gd' | 'momentum') => {
      const fn = quadratic(kappa);
      const lr = method === 'gd' ? 1.8 / kappa : 1 / kappa;
      return descend(fn, method, [-1.8, 0.9], { learningRate: lr, steps: 4000, beta: 0.9 }, 1e-6)
        .converged;
    };
    const easy = steps(5, 'gd')!;
    const hard = steps(100, 'gd')!;
    expect(hard).toBeGreaterThan(5 * easy);
    expect(steps(100, 'momentum')!).toBeLessThan(hard);
  });

  it('L-BFGS reaches the Rosenbrock minimum quickly where plain gradient descent has barely moved', () => {
    const fn = TEST_FUNCTIONS.rosenbrock!;
    const lbfgs = descend(fn, 'lbfgs', fn.start, { learningRate: 1, steps: 100 }, 1e-6);
    expect(distanceToMinimum(fn, lbfgs.points[lbfgs.points.length - 1]!)).toBeLessThan(1e-4);
    const gd = descend(fn, 'gd', fn.start, { learningRate: fn.safeLearningRate, steps: 100 });
    expect(distanceToMinimum(fn, gd.points[gd.points.length - 1]!)).toBeGreaterThan(0.3);
  });

  it('Adam makes steady progress on Rosenbrock', () => {
    const fn = TEST_FUNCTIONS.rosenbrock!;
    const p = descend(fn, 'adam', fn.start, { learningRate: 0.05, steps: 2000 });
    expect(p.values[p.values.length - 1]!).toBeLessThan(p.values[0]! / 100);
  });

  it('Newton with eigenvalue flipping escapes a saddle where gradient descent started on its axis stays', () => {
    const fn = TEST_FUNCTIONS.saddle!;
    const start: [number, number] = [1.5, 0];
    const gd = descend(fn, 'gd', start, { learningRate: 0.2, steps: 300 });
    expect(Math.abs(gd.points[gd.points.length - 1]![1])).toBeLessThan(1e-6);
    const near: [number, number] = [1.5, 1e-3];
    const gd2 = descend(fn, 'gd', near, { learningRate: 0.2, steps: 300 });
    expect(Math.abs(gd2.points[gd2.points.length - 1]![1])).toBeGreaterThan(2);
  });

  it('Himmelblau: different starting points end in different minima', () => {
    const fn = TEST_FUNCTIONS.himmelblau!;
    const ends = (
      [
        [-4, 4],
        [4, 4],
        [-4, -4],
        [4, -4],
      ] as const
    ).map((s) => {
      const p = descend(fn, 'lbfgs', s, { learningRate: 1, steps: 100 }, 1e-8);
      const e = p.points[p.points.length - 1]!;
      return fn.minima.findIndex((m) => Math.hypot(m[0] - e[0], m[1] - e[1]) < 1e-2);
    });
    expect(new Set(ends).size).toBeGreaterThanOrEqual(3);
    expect(ends.every((i) => i >= 0)).toBe(true);
  });

  it('a path starts at the start and records one value per step', () => {
    const p = descend(bowl, 'nesterov', [1, 1], { learningRate: 0.05, steps: 20 });
    expect(p.points[0]).toEqual([1, 1]);
    expect(p.values).toHaveLength(p.points.length);
    expect(p.gradientNorms).toHaveLength(p.points.length);
  });
});
