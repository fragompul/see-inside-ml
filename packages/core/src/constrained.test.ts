import { describe, expect, it } from 'vitest';
import {
  CONSTRAINED_PROBLEMS,
  distanceToSolution,
  kktResidual,
  minimiseBfgs,
  solveConstrained,
  type ConstrainedMethod,
} from './constrained';

const run = (
  id: string,
  method: ConstrainedMethod,
  outer: number,
  weight: number,
  factor: number,
) => solveConstrained(CONSTRAINED_PROBLEMS[id]!, { method, outer, weight, factor });

describe('the unconstrained solver', () => {
  it('finds the minimum of a stretched bowl and of Rosenbrock', () => {
    const bowl = minimiseBfgs(
      (x) => x[0] ** 2 + 20 * x[1] ** 2,
      (x) => [2 * x[0], 40 * x[1]],
      [3, -2],
    );
    expect(bowl.x[0]).toBeCloseTo(0, 6);
    expect(bowl.x[1]).toBeCloseTo(0, 6);
    const p = CONSTRAINED_PROBLEMS.rosenbrock!;
    const banana = minimiseBfgs(
      (x) => (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2,
      p.grad,
      [-1.2, 1],
      2000,
    );
    expect(banana.x[0]).toBeCloseTo(1, 4);
    expect(banana.x[1]).toBeCloseTo(1, 4);
  });
});

describe('augmented Lagrangian', () => {
  it.each(['disk', 'line', 'polygon'])(
    'reaches the known solution and multipliers on "%s" with a modest weight',
    (id) => {
      const steps = run(id, 'augmented', 12, 10, 1);
      const last = steps[steps.length - 1]!;
      const p = CONSTRAINED_PROBLEMS[id]!;
      expect(distanceToSolution(p, last.x)).toBeLessThan(1e-6);
      expect(last.violation).toBeLessThan(1e-6);
      last.multipliers.forEach((m, i) => expect(m).toBeCloseTo(p.solution.multipliers[i]!, 4));
      expect(kktResidual(p, last.x, last.multipliers)).toBeLessThan(1e-4);
    },
  );

  it('on the banana in a disc the result satisfies the KKT conditions', () => {
    const p = CONSTRAINED_PROBLEMS.rosenbrock!;
    const last = run('rosenbrock', 'augmented', 25, 20, 1).pop()!;
    expect(Math.hypot(last.x[0], last.x[1])).toBeCloseTo(1, 4);
    expect(kktResidual(p, last.x, last.multipliers)).toBeLessThan(1e-4);
    expect(last.multipliers[0]).toBeGreaterThan(0);
  });
});

describe('quadratic penalty', () => {
  it('is never quite feasible, and the error shrinks roughly in proportion to the weight', () => {
    const steps = run('disk', 'penalty', 5, 1, 10);
    const errors = steps.map((s) => distanceToSolution(CONSTRAINED_PROBLEMS.disk!, s.x));
    for (let i = 1; i < errors.length; i++) {
      expect(errors[i]!).toBeLessThan(errors[i - 1]!);
      expect(errors[i - 1]! / errors[i]!).toBeGreaterThan(5);
      expect(errors[i - 1]! / errors[i]!).toBeLessThan(15);
    }
    for (const s of steps) expect(s.violation).toBeGreaterThan(1e-9);
  });
});

describe('logarithmic barrier', () => {
  it('stays strictly inside the region and approaches the boundary as the weight shrinks', () => {
    const steps = run('disk', 'barrier', 8, 1, 10);
    for (const s of steps) expect(s.violation).toBeLessThan(0);
    const gaps = steps.map((s) => -s.violation);
    for (let i = 1; i < gaps.length; i++) expect(gaps[i]!).toBeLessThan(gaps[i - 1]!);
    expect(distanceToSolution(CONSTRAINED_PROBLEMS.disk!, steps[steps.length - 1]!.x)).toBeLessThan(
      1e-4,
    );
  });

  it('refuses a start outside the region', () => {
    expect(() =>
      solveConstrained(
        { ...CONSTRAINED_PROBLEMS.disk!, start: [3, 3] },
        { method: 'barrier', outer: 2, weight: 1, factor: 10 },
      ),
    ).toThrow();
  });
});
