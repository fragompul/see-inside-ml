import {
  DESCENT_METHODS,
  TEST_FUNCTIONS,
  descend,
  distanceToMinimum,
  type DescentMethod,
  type DescentPath,
  type TestFunction,
} from '@sim/core';

export type FunctionId = 'quadratic' | 'rosenbrock' | 'himmelblau' | 'saddle' | 'beale';

export interface OptimizerScenario {
  fn: FunctionId;
  /** Learning rate of the first-order methods (Newton and L-BFGS choose their own steps). */
  learningRate: number;
  steps: number;
  /** Starting point; null = the function's default. */
  start: [number, number] | null;
  methods: DescentMethod[];
  momentum: number;
}

export const DEFAULT_OPTIMIZERS: OptimizerScenario = {
  fn: 'quadratic',
  learningRate: 0.03,
  steps: 200,
  start: null,
  methods: [...DESCENT_METHODS],
  momentum: 0.9,
};

export const METHOD_LABELS: Record<DescentMethod, string> = {
  gd: 'Gradient descent',
  momentum: 'Momentum',
  nesterov: 'Nesterov',
  rmsprop: 'RMSprop',
  adam: 'Adam',
  newton: 'Newton',
  lbfgs: 'L-BFGS',
};

export const METHOD_NOTES: Record<DescentMethod, string> = {
  gd: 'Steps straight downhill, scaled by the learning rate.',
  momentum:
    'Keeps a velocity: an average of past gradients, so it builds speed along consistent directions.',
  nesterov:
    'Momentum that looks ahead: takes the gradient where the velocity is about to carry it.',
  rmsprop: 'Divides each coordinate by a running average of its recent gradient size.',
  adam: 'Momentum plus per-coordinate scaling: the default of deep learning.',
  newton: 'Uses the curvature (the Hessian) to jump straight to the bottom of the local quadratic.',
  lbfgs:
    'Builds an approximate curvature from the last ten steps and a line search: no Hessian needed.',
};

export interface OptimizerRun {
  scenario: OptimizerScenario;
  fn: TestFunction;
  start: [number, number];
  paths: DescentPath[];
}

export function runOptimizers(s: OptimizerScenario): OptimizerRun {
  const fn = TEST_FUNCTIONS[s.fn] as TestFunction;
  const start = s.start ?? fn.start;
  const paths = s.methods.map((m) =>
    descend(fn, m, start, { learningRate: s.learningRate, steps: s.steps, beta: s.momentum }, 1e-4),
  );
  return { scenario: s, fn, start, paths };
}

export const pathOf = (run: OptimizerRun, m: DescentMethod): DescentPath =>
  run.paths.find((p) => p.method === m) as DescentPath;

export const endOf = (p: DescentPath): [number, number] =>
  p.points[p.points.length - 1] as [number, number];

export const finalDistance = (run: OptimizerRun, m: DescentMethod): number =>
  distanceToMinimum(run.fn, endOf(pathOf(run, m)));

/** Index of the known minimum a path ended in (−1 if none is close). */
export function minimumReached(run: OptimizerRun, m: DescentMethod): number {
  const e = endOf(pathOf(run, m));
  return run.fn.minima.findIndex((mm) => Math.hypot(mm[0] - e[0], mm[1] - e[1]) < 0.05);
}
