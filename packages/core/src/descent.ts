import { Matrix } from './matrix';
import { solve, symmetricEigen } from './linalg';

/**
 * Optimisers on small smooth functions of two variables, written the textbook way so that their paths
 * can be compared: plain gradient descent, momentum, Nesterov, RMSprop, Adam, Newton and L-BFGS.
 */
export interface TestFunction {
  id: string;
  label: string;
  note: string;
  f(x: readonly number[]): number;
  grad(x: readonly number[]): [number, number];
  hess(x: readonly number[]): [[number, number], [number, number]];
  /** Known local minima (the lab checks convergence against them). */
  minima: [number, number][];
  /** Plotting window. */
  bounds: { x0: number; x1: number; y0: number; y1: number };
  /** A sensible starting point. */
  start: [number, number];
  /** Learning rate that works for plain gradient descent from `start`. */
  safeLearningRate: number;
}

export function quadratic(condition: number): TestFunction {
  // f = 0.5 (x^2 + kappa y^2): the textbook ill-conditioned bowl
  return {
    id: 'quadratic',
    label: `Stretched bowl (condition ${condition})`,
    note: 'A bowl much steeper in one direction than the other: gradient descent zig-zags across the narrow valley.',
    f: (x) => 0.5 * ((x[0] as number) ** 2 + condition * (x[1] as number) ** 2),
    grad: (x) => [x[0] as number, condition * (x[1] as number)],
    hess: () => [
      [1, 0],
      [0, condition],
    ],
    minima: [[0, 0]],
    bounds: { x0: -2.2, x1: 2.2, y0: -1.2, y1: 1.2 },
    start: [-1.8, 0.9],
    safeLearningRate: 1 / condition,
  };
}

export const TEST_FUNCTIONS: Record<string, TestFunction> = {
  quadratic: quadratic(30),
  rosenbrock: {
    id: 'rosenbrock',
    label: 'Rosenbrock banana',
    note: 'A long curved valley with a flat floor: easy to find, hard to follow to the minimum at (1, 1).',
    f: (x) => (1 - (x[0] as number)) ** 2 + 100 * ((x[1] as number) - (x[0] as number) ** 2) ** 2,
    grad: (x) => {
      const a = x[0] as number;
      const b = x[1] as number;
      return [-2 * (1 - a) - 400 * a * (b - a * a), 200 * (b - a * a)];
    },
    hess: (x) => {
      const a = x[0] as number;
      const b = x[1] as number;
      return [
        [2 - 400 * (b - a * a) + 800 * a * a, -400 * a],
        [-400 * a, 200],
      ];
    },
    minima: [[1, 1]],
    bounds: { x0: -2, x1: 2, y0: -1, y1: 3 },
    start: [-1.5, 1.5],
    safeLearningRate: 0.0015,
  },
  himmelblau: {
    id: 'himmelblau',
    label: 'Himmelblau (four minima)',
    note: 'Four equally good valleys: where you end up depends on where you start.',
    f: (x) =>
      ((x[0] as number) ** 2 + (x[1] as number) - 11) ** 2 +
      ((x[0] as number) + (x[1] as number) ** 2 - 7) ** 2,
    grad: (x) => {
      const a = x[0] as number;
      const b = x[1] as number;
      return [
        4 * a * (a * a + b - 11) + 2 * (a + b * b - 7),
        2 * (a * a + b - 11) + 4 * b * (a + b * b - 7),
      ];
    },
    hess: (x) => {
      const a = x[0] as number;
      const b = x[1] as number;
      return [
        [12 * a * a + 4 * b - 42, 4 * a + 4 * b],
        [4 * a + 4 * b, 12 * b * b + 4 * a - 26],
      ];
    },
    minima: [
      [3, 2],
      [-2.805118, 3.131312],
      [-3.77931, -3.283186],
      [3.584428, -1.848126],
    ],
    bounds: { x0: -5, x1: 5, y0: -5, y1: 5 },
    start: [0.5, -0.5],
    safeLearningRate: 0.01,
  },
  saddle: {
    id: 'saddle',
    label: 'Saddle point',
    note: 'Flat in one direction at the centre, with two minima beyond: a model of a saddle in a loss landscape.',
    // minima at (0, +-sqrt(10)); saddle at the origin
    f: (x) =>
      0.5 * (x[0] as number) ** 2 - 0.5 * (x[1] as number) ** 2 + 0.05 * (x[1] as number) ** 4,
    grad: (x) => [x[0] as number, -(x[1] as number) + 0.2 * (x[1] as number) ** 3],
    hess: (x) => [
      [1, 0],
      [0, -1 + 0.6 * (x[1] as number) ** 2],
    ],
    minima: [
      [0, Math.sqrt(5)],
      [0, -Math.sqrt(5)],
    ],
    bounds: { x0: -3, x1: 3, y0: -3.2, y1: 3.2 },
    start: [1.5, 0],
    safeLearningRate: 0.2,
  },
  beale: {
    id: 'beale',
    label: 'Beale',
    note: 'Steep walls and a nearly flat floor leading to the minimum at (3, 0.5).',
    f: (x) => {
      const a = x[0] as number;
      const b = x[1] as number;
      return (1.5 - a + a * b) ** 2 + (2.25 - a + a * b * b) ** 2 + (2.625 - a + a * b ** 3) ** 2;
    },
    grad: (x) => {
      const a = x[0] as number;
      const b = x[1] as number;
      const t1 = 1.5 - a + a * b;
      const t2 = 2.25 - a + a * b * b;
      const t3 = 2.625 - a + a * b ** 3;
      return [
        2 * t1 * (b - 1) + 2 * t2 * (b * b - 1) + 2 * t3 * (b ** 3 - 1),
        2 * t1 * a + 2 * t2 * 2 * a * b + 2 * t3 * 3 * a * b * b,
      ];
    },
    hess: (x) => {
      const a = x[0] as number;
      const b = x[1] as number;
      const t1 = 1.5 - a + a * b;
      const t2 = 2.25 - a + a * b * b;
      const t3 = 2.625 - a + a * b ** 3;
      const d1a = b - 1;
      const d2a = b * b - 1;
      const d3a = b ** 3 - 1;
      const d1b = a;
      const d2b = 2 * a * b;
      const d3b = 3 * a * b * b;
      const haa = 2 * (d1a * d1a + d2a * d2a + d3a * d3a);
      const hab =
        2 * (d1a * d1b + d2a * d2b + d3a * d3b) + 2 * (t1 * 1 + t2 * 2 * b + t3 * 3 * b * b);
      const hbb = 2 * (d1b * d1b + d2b * d2b + d3b * d3b) + 2 * (t2 * 2 * a + t3 * 6 * a * b);
      return [
        [haa, hab],
        [hab, hbb],
      ];
    },
    minima: [[3, 0.5]],
    bounds: { x0: -4.5, x1: 4.5, y0: -4.5, y1: 4.5 },
    start: [-2, -1.5],
    safeLearningRate: 0.002,
  },
};

export type DescentMethod =
  'gd' | 'momentum' | 'nesterov' | 'rmsprop' | 'adam' | 'newton' | 'lbfgs';

export const DESCENT_METHODS: DescentMethod[] = [
  'gd',
  'momentum',
  'nesterov',
  'rmsprop',
  'adam',
  'newton',
  'lbfgs',
];

export interface DescentOptions {
  learningRate: number;
  steps: number;
  /** Momentum coefficient (momentum, Nesterov). */
  beta?: number;
}

export interface DescentPath {
  method: DescentMethod;
  /** The point after each step (entry 0 is the start). */
  points: [number, number][];
  values: number[];
  gradientNorms: number[];
  /** Steps taken until the gradient norm fell below `tolerance` (null if it never did). */
  converged: number | null;
  diverged: boolean;
}

const norm = (g: readonly number[]) => Math.hypot(g[0] as number, g[1] as number);

export function descend(
  fn: TestFunction,
  method: DescentMethod,
  start: readonly [number, number],
  options: DescentOptions,
  tolerance = 1e-4,
): DescentPath {
  const { learningRate: lr, steps, beta = 0.9 } = options;
  let x: [number, number] = [start[0], start[1]];
  const points: [number, number][] = [[...x]];
  const values = [fn.f(x)];
  const gradientNorms = [norm(fn.grad(x))];
  let converged: number | null = gradientNorms[0]! < tolerance ? 0 : null;
  let diverged = false;
  let v: [number, number] = [0, 0];
  let m: [number, number] = [0, 0];
  let s: [number, number] = [0, 0];
  // L-BFGS memory
  const memory: { s: [number, number]; y: [number, number]; rho: number }[] = [];
  let prevX: [number, number] | null = null;
  let prevG: [number, number] | null = null;

  for (let t = 1; t <= steps; t++) {
    let next: [number, number];
    if (method === 'gd') {
      const g = fn.grad(x);
      next = [x[0] - lr * g[0], x[1] - lr * g[1]];
    } else if (method === 'momentum') {
      const g = fn.grad(x);
      v = [beta * v[0] - lr * g[0], beta * v[1] - lr * g[1]];
      next = [x[0] + v[0], x[1] + v[1]];
    } else if (method === 'nesterov') {
      const look: [number, number] = [x[0] + beta * v[0], x[1] + beta * v[1]];
      const g = fn.grad(look);
      v = [beta * v[0] - lr * g[0], beta * v[1] - lr * g[1]];
      next = [x[0] + v[0], x[1] + v[1]];
    } else if (method === 'rmsprop') {
      const g = fn.grad(x);
      s = [0.9 * s[0] + 0.1 * g[0] * g[0], 0.9 * s[1] + 0.1 * g[1] * g[1]];
      next = [
        x[0] - (lr * g[0]) / (Math.sqrt(s[0]) + 1e-8),
        x[1] - (lr * g[1]) / (Math.sqrt(s[1]) + 1e-8),
      ];
    } else if (method === 'adam') {
      const g = fn.grad(x);
      m = [0.9 * m[0] + 0.1 * g[0], 0.9 * m[1] + 0.1 * g[1]];
      s = [0.999 * s[0] + 0.001 * g[0] * g[0], 0.999 * s[1] + 0.001 * g[1] * g[1]];
      const mh = [m[0] / (1 - 0.9 ** t), m[1] / (1 - 0.9 ** t)];
      const sh = [s[0] / (1 - 0.999 ** t), s[1] / (1 - 0.999 ** t)];
      next = [
        x[0] - (lr * (mh[0] as number)) / (Math.sqrt(sh[0] as number) + 1e-8),
        x[1] - (lr * (mh[1] as number)) / (Math.sqrt(sh[1] as number) + 1e-8),
      ];
    } else if (method === 'newton') {
      const g = fn.grad(x);
      const h = fn.hess(x);
      // make the Hessian positive definite by flipping/flooring its eigenvalues (a saddle becomes a descent direction)
      const eig = symmetricEigen(
        Matrix.from([
          [h[0][0], h[0][1]],
          [h[1][0], h[1][1]],
        ]),
      );
      const lam = [0, 1].map((i) => Math.max(Math.abs(eig.values[i] as number), 1e-3));
      let dx = 0;
      let dy = 0;
      for (let i = 0; i < 2; i++) {
        const proj =
          (eig.vectors.get(0, i) * g[0] + eig.vectors.get(1, i) * g[1]) / (lam[i] as number);
        dx -= proj * eig.vectors.get(0, i);
        dy -= proj * eig.vectors.get(1, i);
      }
      // backtracking so that a full step never makes things worse
      let step = 1;
      const f0 = fn.f(x);
      while (step > 1e-6 && fn.f([x[0] + step * dx, x[1] + step * dy]) > f0 + 1e-12) step *= 0.5;
      next = [x[0] + step * dx, x[1] + step * dy];
    } else {
      const g = fn.grad(x);
      if (prevX && prevG) {
        const sk: [number, number] = [x[0] - prevX[0], x[1] - prevX[1]];
        const yk: [number, number] = [g[0] - prevG[0], g[1] - prevG[1]];
        const sy = sk[0] * yk[0] + sk[1] * yk[1];
        if (sy > 1e-12) {
          memory.push({ s: sk, y: yk, rho: 1 / sy });
          if (memory.length > 10) memory.shift();
        }
      }
      // two-loop recursion
      let q: [number, number] = [g[0], g[1]];
      const alphas: number[] = [];
      for (let i = memory.length - 1; i >= 0; i--) {
        const mem = memory[i]!;
        const a = mem.rho * (mem.s[0] * q[0] + mem.s[1] * q[1]);
        alphas[i] = a;
        q = [q[0] - a * mem.y[0], q[1] - a * mem.y[1]];
      }
      const last = memory[memory.length - 1];
      const gamma = last
        ? (last.s[0] * last.y[0] + last.s[1] * last.y[1]) / (last.y[0] ** 2 + last.y[1] ** 2)
        : 1 / Math.max(norm(g), 1);
      let r: [number, number] = [gamma * q[0], gamma * q[1]];
      for (let i = 0; i < memory.length; i++) {
        const mem = memory[i]!;
        const b = mem.rho * (mem.y[0] * r[0] + mem.y[1] * r[1]);
        r = [
          r[0] + mem.s[0] * ((alphas[i] as number) - b),
          r[1] + mem.s[1] * ((alphas[i] as number) - b),
        ];
      }
      let dir: [number, number] = [-r[0], -r[1]];
      if (dir[0] * g[0] + dir[1] * g[1] >= 0) dir = [-g[0], -g[1]];
      // Armijo backtracking line search
      let step = 1;
      const f0 = fn.f(x);
      const slope = dir[0] * g[0] + dir[1] * g[1];
      while (
        step > 1e-10 &&
        fn.f([x[0] + step * dir[0], x[1] + step * dir[1]]) > f0 + 1e-4 * step * slope
      )
        step *= 0.5;
      prevX = x;
      prevG = g;
      next = [x[0] + step * dir[0], x[1] + step * dir[1]];
    }
    x = next;
    if (
      !Number.isFinite(x[0]) ||
      !Number.isFinite(x[1]) ||
      Math.abs(x[0]) > 1e6 ||
      Math.abs(x[1]) > 1e6
    ) {
      diverged = true;
      break;
    }
    points.push([...x]);
    values.push(fn.f(x));
    const gn = norm(fn.grad(x));
    gradientNorms.push(gn);
    if (converged === null && gn < tolerance) converged = t;
  }
  return { method, points, values, gradientNorms, converged, diverged };
}

/** Distance from a point to the nearest known minimum. */
export function distanceToMinimum(fn: TestFunction, x: readonly number[]): number {
  return Math.min(
    ...fn.minima.map((m) => Math.hypot((x[0] as number) - m[0], (x[1] as number) - m[1])),
  );
}

// kept for completeness: a direct solve used by tests to cross-check Newton on quadratics
export function newtonStepExact(fn: TestFunction, x: readonly [number, number]): [number, number] {
  const g = fn.grad(x);
  const h = fn.hess(x);
  const d = solve(
    Matrix.from([
      [h[0][0], h[0][1]],
      [h[1][0], h[1][1]],
    ]),
    new Matrix(2, 1, Float64Array.from([-g[0], -g[1]])),
  );
  return [x[0] + d.get(0, 0), x[1] + d.get(1, 0)];
}
