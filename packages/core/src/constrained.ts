/**
 * Constrained optimisation on small problems of two variables: minimise f(x) subject to equality constraints h(x) = 0 and
 * inequality constraints g(x) <= 0. Three ways of turning the constrained problem into a sequence of unconstrained ones:
 *  - quadratic penalty: add (mu / 2) (sum h^2 + sum max(0, g)^2) and let mu grow; the solution is never exactly feasible
 *    and the problems get harder as mu grows;
 *  - augmented Lagrangian: the same penalty plus multiplier terms that are updated after each solve, so that a moderate mu
 *    is enough and the multipliers converge to those of the KKT conditions;
 *  - logarithmic barrier (inequalities only): add -t sum log(-g) and let t shrink; every iterate is strictly feasible.
 */
export type Vec2 = [number, number];

export interface Constraint {
  kind: 'eq' | 'ineq';
  label: string;
  /** Value of h (must be 0) or g (must be at most 0). */
  value: (x: Vec2) => number;
  gradient: (x: Vec2) => Vec2;
}

export interface ConstrainedProblem {
  id: string;
  label: string;
  note: string;
  f: (x: Vec2) => number;
  grad: (x: Vec2) => Vec2;
  constraints: Constraint[];
  bounds: { x0: number; x1: number; y0: number; y1: number };
  start: Vec2;
  /** The known solution and the multiplier of each constraint (0 for inactive ones), with the convention grad f + sum multiplier * grad constraint = 0. */
  solution: { x: Vec2; multipliers: number[] };
}

const disk = (cx: number, cy: number, r2: number): Constraint => ({
  kind: 'ineq',
  label: `x² + y² ≤ ${r2}`,
  value: (x) => (x[0] - cx) ** 2 + (x[1] - cy) ** 2 - r2,
  gradient: (x) => [2 * (x[0] - cx), 2 * (x[1] - cy)],
});

export const CONSTRAINED_PROBLEMS: Record<string, ConstrainedProblem> = {
  disk: {
    id: 'disk',
    label: 'A target outside a disc',
    note: 'The best point of the unit disc is the one nearest the target (2, 1): on the rim, along the line to it.',
    f: (x) => (x[0] - 2) ** 2 + (x[1] - 1) ** 2,
    grad: (x) => [2 * (x[0] - 2), 2 * (x[1] - 1)],
    constraints: [disk(0, 0, 1)],
    bounds: { x0: -1.6, x1: 2.6, y0: -1.4, y1: 2 },
    start: [-0.5, -0.5],
    solution: { x: [2 / Math.sqrt(5), 1 / Math.sqrt(5)], multipliers: [Math.sqrt(5) - 1] },
  },
  line: {
    id: 'line',
    label: 'Nearest point on a line',
    note: 'Minimise x² + y² on the line x + y = 1: the foot of the perpendicular from the origin, (½, ½).',
    f: (x) => x[0] ** 2 + x[1] ** 2,
    grad: (x) => [2 * x[0], 2 * x[1]],
    constraints: [
      {
        kind: 'eq',
        label: 'x + y = 1',
        value: (x) => x[0] + x[1] - 1,
        gradient: () => [1, 1],
      },
    ],
    bounds: { x0: -1.2, x1: 1.8, y0: -1.2, y1: 1.8 },
    start: [1.4, -0.8],
    solution: { x: [0.5, 0.5], multipliers: [-1] },
  },
  polygon: {
    id: 'polygon',
    label: 'A triangle',
    note: 'Minimise (x − 1)² + (y − 2.5)² inside a triangle whose slanted side cuts the target off: the solution is on that side, at (1.4, 1.7).',
    f: (x) => (x[0] - 1) ** 2 + (x[1] - 2.5) ** 2,
    grad: (x) => [2 * (x[0] - 1), 2 * (x[1] - 2.5)],
    constraints: [
      {
        kind: 'ineq',
        label: 'x − 2y + 2 ≥ 0',
        value: (x) => -(x[0] - 2 * x[1] + 2),
        gradient: () => [-1, 2],
      },
      {
        kind: 'ineq',
        label: '−x − 2y + 6 ≥ 0',
        value: (x) => -(-x[0] - 2 * x[1] + 6),
        gradient: () => [1, 2],
      },
      {
        kind: 'ineq',
        label: '−x + 2y + 2 ≥ 0',
        value: (x) => -(-x[0] + 2 * x[1] + 2),
        gradient: () => [1, -2],
      },
      { kind: 'ineq', label: 'x ≥ 0', value: (x) => -x[0], gradient: () => [-1, 0] },
      { kind: 'ineq', label: 'y ≥ 0', value: (x) => -x[1], gradient: () => [0, -1] },
    ],
    bounds: { x0: -0.6, x1: 4.6, y0: -0.6, y1: 3.6 },
    start: [1, 0.6],
    solution: { x: [1.4, 1.7], multipliers: [0.8, 0, 0, 0, 0] },
  },
  rosenbrock: {
    id: 'rosenbrock',
    label: 'The banana inside a disc',
    note: 'The valley of Rosenbrock’s function leads out of the unit disc: the constrained minimum sits on the rim, near (0.786, 0.618).',
    f: (x) => (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2,
    grad: (x) => [-2 * (1 - x[0]) - 400 * x[0] * (x[1] - x[0] ** 2), 200 * (x[1] - x[0] ** 2)],
    constraints: [disk(0, 0, 1)],
    bounds: { x0: -1.5, x1: 1.5, y0: -1.1, y1: 1.5 },
    start: [-0.8, 0.3],
    // refined below: no closed form is known
    solution: { x: [0.7864, 0.6177], multipliers: [0.12] },
  },
};

export type ConstrainedMethod = 'penalty' | 'augmented' | 'barrier';
export const CONSTRAINED_METHODS: ConstrainedMethod[] = ['penalty', 'augmented', 'barrier'];

export interface ConstrainedOptions {
  method: ConstrainedMethod;
  /** Outer iterations (penalty and barrier: stages with a changed weight; augmented: multiplier updates). */
  outer: number;
  /** The first penalty weight mu (penalty and augmented) or the first barrier weight t. */
  weight: number;
  /** Factor by which mu grows (penalty) or t shrinks (barrier) each stage. */
  factor: number;
}

export interface OuterStep {
  x: Vec2;
  weight: number;
  f: number;
  /** Largest violation: max over |h| and max(0, g); for the barrier method it is negative (distance inside). */
  violation: number;
  /** Multiplier estimates, one per constraint. */
  multipliers: number[];
  /** Inner (unconstrained) iterations used in this stage. */
  inner: number;
}

/** Minimises a smooth function of two variables with BFGS and backtracking; returns the point and the iterations used. */
export function minimiseBfgs(
  f: (x: Vec2) => number,
  grad: (x: Vec2) => Vec2,
  start: Vec2,
  maxIterations = 400,
  tolerance = 1e-9,
  feasible: (x: Vec2) => boolean = () => true,
): { x: Vec2; iterations: number } {
  let x: Vec2 = [...start];
  let g = grad(x);
  let h = [
    [1, 0],
    [0, 1],
  ];
  let iterations = 0;
  for (; iterations < maxIterations; iterations++) {
    if (Math.hypot(g[0], g[1]) < tolerance) break;
    let d: Vec2 = [-(h[0]![0]! * g[0] + h[0]![1]! * g[1]), -(h[1]![0]! * g[0] + h[1]![1]! * g[1])];
    if (d[0] * g[0] + d[1] * g[1] >= 0) {
      d = [-g[0], -g[1]];
      h = [
        [1, 0],
        [0, 1],
      ];
    }
    let step = 1;
    const fx = f(x);
    const slope = d[0] * g[0] + d[1] * g[1];
    let next: Vec2 = [x[0] + step * d[0], x[1] + step * d[1]];
    let tries = 0;
    while ((!feasible(next) || !(f(next) <= fx + 1e-4 * step * slope)) && tries < 60) {
      step *= 0.5;
      next = [x[0] + step * d[0], x[1] + step * d[1]];
      tries++;
    }
    if (tries >= 60) break;
    const gn = grad(next);
    const s: Vec2 = [next[0] - x[0], next[1] - x[1]];
    const y: Vec2 = [gn[0] - g[0], gn[1] - g[1]];
    const sy = s[0] * y[0] + s[1] * y[1];
    if (sy > 1e-14) {
      const rho = 1 / sy;
      // H <- (I - rho s y^T) H (I - rho y s^T) + rho s s^T
      const a = [
        [1 - rho * s[0] * y[0], -rho * s[0] * y[1]],
        [-rho * s[1] * y[0], 1 - rho * s[1] * y[1]],
      ];
      const ah = [
        [
          a[0]![0]! * h[0]![0]! + a[0]![1]! * h[1]![0]!,
          a[0]![0]! * h[0]![1]! + a[0]![1]! * h[1]![1]!,
        ],
        [
          a[1]![0]! * h[0]![0]! + a[1]![1]! * h[1]![0]!,
          a[1]![0]! * h[0]![1]! + a[1]![1]! * h[1]![1]!,
        ],
      ];
      h = [
        [
          ah[0]![0]! * a[0]![0]! + ah[0]![1]! * a[0]![1]! + rho * s[0] * s[0],
          ah[0]![0]! * a[1]![0]! + ah[0]![1]! * a[1]![1]! + rho * s[0] * s[1],
        ],
        [
          ah[1]![0]! * a[0]![0]! + ah[1]![1]! * a[0]![1]! + rho * s[1] * s[0],
          ah[1]![0]! * a[1]![0]! + ah[1]![1]! * a[1]![1]! + rho * s[1] * s[1],
        ],
      ];
    }
    x = next;
    g = gn;
  }
  return { x, iterations };
}

const violationOf = (p: ConstrainedProblem, x: Vec2): number =>
  Math.max(
    0,
    ...p.constraints.map((c) => (c.kind === 'eq' ? Math.abs(c.value(x)) : Math.max(0, c.value(x)))),
  );

export function solveConstrained(p: ConstrainedProblem, o: ConstrainedOptions): OuterStep[] {
  const steps: OuterStep[] = [];
  let x: Vec2 = [...p.start];
  let weight = o.weight;
  const lambda = p.constraints.map(() => 0);
  if (o.method === 'barrier') {
    // the start has to be strictly feasible: pull it towards the centre of the region if it is not
    const inside = (z: Vec2) => p.constraints.every((c) => c.kind === 'ineq' && c.value(z) < 0);
    if (!inside(x))
      throw new RangeError(
        'The log barrier needs a start strictly inside the region and no equality constraints',
      );
  }
  for (let k = 0; k < o.outer; k++) {
    let objective: (z: Vec2) => number;
    let gradient: (z: Vec2) => Vec2;
    if (o.method === 'barrier') {
      const t = weight;
      objective = (z) => p.f(z) - t * p.constraints.reduce((a, c) => a + Math.log(-c.value(z)), 0);
      gradient = (z) => {
        const gr = p.grad(z);
        let gx = gr[0];
        let gy = gr[1];
        for (const c of p.constraints) {
          const v = c.value(z);
          const dg = c.gradient(z);
          gx += (-t / v) * dg[0];
          gy += (-t / v) * dg[1];
        }
        return [gx, gy];
      };
    } else {
      const mu = weight;
      const augmented = o.method === 'augmented';
      // for the augmented Lagrangian an inequality contributes (max(0, lambda + mu g)^2 - lambda^2) / (2 mu), an equality lambda h + mu h^2 / 2
      objective = (z) => {
        let v = p.f(z);
        p.constraints.forEach((c, i) => {
          const val = c.value(z);
          const l = augmented ? (lambda[i] as number) : 0;
          if (c.kind === 'eq') v += l * val + (mu / 2) * val * val;
          else v += (Math.max(0, l + mu * val) ** 2 - l * l) / (2 * mu);
        });
        return v;
      };
      gradient = (z) => {
        const gr = p.grad(z);
        let gx = gr[0];
        let gy = gr[1];
        p.constraints.forEach((c, i) => {
          const val = c.value(z);
          const l = augmented ? (lambda[i] as number) : 0;
          const dg = c.gradient(z);
          const factor = c.kind === 'eq' ? l + mu * val : Math.max(0, l + mu * val);
          gx += factor * dg[0];
          gy += factor * dg[1];
        });
        return [gx, gy];
      };
    }
    const feasible =
      o.method === 'barrier' ? (z: Vec2) => p.constraints.every((c) => c.value(z) < 0) : undefined;
    const result = minimiseBfgs(objective, gradient, x, 600, 1e-10, feasible);
    x = result.x;
    // multiplier estimates for this stage
    const estimates = p.constraints.map((c, i) => {
      const val = c.value(x);
      if (o.method === 'barrier') return -weight / val;
      const l = o.method === 'augmented' ? (lambda[i] as number) : 0;
      return c.kind === 'eq' ? l + weight * val : Math.max(0, l + weight * val);
    });
    if (o.method === 'augmented') estimates.forEach((e, i) => (lambda[i] = e));
    steps.push({
      x: [...x],
      weight,
      f: p.f(x),
      violation:
        o.method === 'barrier'
          ? Math.max(...p.constraints.map((c) => c.value(x)))
          : violationOf(p, x),
      multipliers: estimates,
      inner: result.iterations,
    });
    if (o.method === 'penalty') weight *= o.factor;
    if (o.method === 'barrier') weight /= o.factor;
  }
  return steps;
}

export const distanceToSolution = (p: ConstrainedProblem, x: Vec2): number =>
  Math.hypot(x[0] - p.solution.x[0], x[1] - p.solution.x[1]);

/** Largest violation of the KKT conditions at x with multipliers m: stationarity, feasibility, sign and complementary slackness. */
export function kktResidual(p: ConstrainedProblem, x: Vec2, m: readonly number[]): number {
  const gr = p.grad(x);
  let sx = gr[0];
  let sy = gr[1];
  p.constraints.forEach((c, i) => {
    const dg = c.gradient(x);
    sx += (m[i] as number) * dg[0];
    sy += (m[i] as number) * dg[1];
  });
  let worst = Math.hypot(sx, sy);
  p.constraints.forEach((c, i) => {
    const v = c.value(x);
    worst = Math.max(worst, c.kind === 'eq' ? Math.abs(v) : Math.max(0, v));
    if (c.kind === 'ineq')
      worst = Math.max(worst, Math.max(0, -(m[i] as number)), Math.abs((m[i] as number) * v));
  });
  return worst;
}

// The one problem without a closed-form answer gets its reference solution from a long augmented-Lagrangian run.
{
  const p = CONSTRAINED_PROBLEMS.rosenbrock as ConstrainedProblem;
  const last = solveConstrained(p, {
    method: 'augmented',
    outer: 40,
    weight: 50,
    factor: 1,
  }).pop() as OuterStep;
  p.solution = { x: last.x, multipliers: last.multipliers };
}
