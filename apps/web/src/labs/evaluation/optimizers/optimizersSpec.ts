import { TEST_FUNCTIONS } from '@sim/core';
import type { ParamSpec } from '@/shared/lab-kit';
import {
  endOf,
  finalDistance,
  minimumReached,
  pathOf,
  type FunctionId,
  type OptimizerRun,
  type OptimizerScenario,
} from './optimizersCore';

export const optimizerParams: ParamSpec[] = [
  {
    key: 'fn',
    label: 'Function to minimise',
    kind: 'select',
    options: (['quadratic', 'rosenbrock', 'himmelblau', 'saddle', 'beale'] as FunctionId[]).map(
      (id) => ({
        value: id,
        label: TEST_FUNCTIONS[id]!.label,
      }),
    ),
  },
  {
    key: 'learningRate',
    label: 'Learning rate (first-order methods)',
    kind: 'ladder',
    values: [0.0005, 0.0015, 0.002, 0.01, 0.02, 0.03, 0.05, 0.06, 0.08, 0.2],
    format: (v) => String(v),
    hint: 'Newton and L-BFGS pick their own step sizes.',
  },
  {
    key: 'steps',
    label: 'Steps',
    kind: 'ladder',
    values: [20, 50, 100, 200, 300, 500],
    format: (v) => String(v),
  },
  {
    key: 'momentum',
    label: 'Momentum coefficient',
    kind: 'slider',
    min: 0,
    max: 0.99,
    step: 0.01,
    format: (v) => v.toFixed(2),
  },
];

export interface OptimizerExperiment {
  id: string;
  title: string;
  text: string;
  scenario: Partial<OptimizerScenario>;
  claim: (run: OptimizerRun) => boolean;
}

const conv = (r: OptimizerRun, m: Parameters<typeof pathOf>[1]) => pathOf(r, m).converged;

export const optimizerExperiments: OptimizerExperiment[] = [
  {
    id: 'zigzag',
    title: 'Zig-zagging down a narrow valley',
    text: 'The bowl is thirty times steeper across than along. Plain gradient descent must keep its step small enough not to bounce out across the steep direction, so it crawls along the shallow one and has not converged after 200 steps. Momentum and Nesterov average out the bouncing; Newton uses the curvature and solves it in one step.',
    scenario: { fn: 'quadratic', learningRate: 0.03, steps: 200 },
    claim: (r) =>
      conv(r, 'gd') === null &&
      (conv(r, 'momentum') ?? 999) <= 200 &&
      (conv(r, 'nesterov') ?? 999) <= 100 &&
      conv(r, 'newton') === 1,
  },
  {
    id: 'too-greedy',
    title: 'A learning rate that is too large',
    text: 'Step size 0.08 is just past the stability limit of the steep direction (2 divided by its curvature of 30). Plain gradient descent and Nesterov blow up; momentum with the same rate and the adaptive methods stay stable.',
    scenario: { fn: 'quadratic', learningRate: 0.08, steps: 200 },
    claim: (r) =>
      pathOf(r, 'gd').diverged &&
      pathOf(r, 'nesterov').diverged &&
      !pathOf(r, 'momentum').diverged &&
      !pathOf(r, 'adam').diverged,
  },
  {
    id: 'valley',
    title: 'The banana valley',
    text: "Rosenbrock's valley has a nearly flat floor curving to (1, 1). After 500 steps plain gradient descent has not even reached the floor; Newton and L-BFGS use curvature information and arrive in a few dozen steps.",
    scenario: { fn: 'rosenbrock', learningRate: 0.0015, steps: 500 },
    claim: (r) =>
      finalDistance(r, 'gd') > 1.5 &&
      finalDistance(r, 'newton') < 1e-3 &&
      finalDistance(r, 'lbfgs') < 1e-3 &&
      (conv(r, 'lbfgs') ?? 999) <= 60,
  },
  {
    id: 'momentum-valley',
    title: 'Momentum follows the valley',
    text: 'With momentum the same small learning rate carries the point along the valley floor: it gets within 0.05 of the minimum in 500 steps, where plain gradient descent is still far away.',
    scenario: { fn: 'rosenbrock', learningRate: 0.0015, steps: 500 },
    claim: (r) => finalDistance(r, 'momentum') < 0.05 && finalDistance(r, 'gd') > 1,
  },
  {
    id: 'adaptive',
    title: 'Adaptive step sizes tolerate a big rate',
    text: 'With a rate of 0.02 plain gradient descent and momentum fly off the banana. RMSprop and Adam normalise the step by the gradient size, so they stay bounded and keep making progress.',
    scenario: { fn: 'rosenbrock', learningRate: 0.02, steps: 500 },
    claim: (r) =>
      pathOf(r, 'gd').diverged &&
      pathOf(r, 'momentum').diverged &&
      !pathOf(r, 'rmsprop').diverged &&
      !pathOf(r, 'adam').diverged &&
      finalDistance(r, 'rmsprop') < 1,
  },
  {
    id: 'saddle-stuck',
    title: 'Stuck on a saddle',
    text: 'Start exactly on the axis through the saddle: the gradient has no component along the escape direction, so every first-order method slides to the saddle and stops, though the loss is lower elsewhere.',
    scenario: { fn: 'saddle', learningRate: 0.2, steps: 200, start: [1.5, 0] },
    claim: (r) =>
      Math.abs(endOf(pathOf(r, 'gd'))[1]) < 1e-6 &&
      minimumReached(r, 'gd') === -1 &&
      minimumReached(r, 'adam') === -1,
  },
  {
    id: 'saddle-nudge',
    title: 'A tiny nudge escapes the saddle',
    text: 'Start a thousandth off the axis and the small component along the unstable direction grows exponentially: the optimisers slip off the saddle and reach a true minimum. Noise in real training does this for free.',
    scenario: { fn: 'saddle', learningRate: 0.2, steps: 200, start: [1.5, 0.001] },
    claim: (r) =>
      minimumReached(r, 'gd') >= 0 &&
      minimumReached(r, 'adam') >= 0 &&
      minimumReached(r, 'newton') >= 0,
  },
  {
    id: 'basins',
    title: 'Where you start decides where you end',
    text: 'Himmelblau has four equally good valleys. The same optimiser started from two different corners ends in two different minima.',
    scenario: { fn: 'himmelblau', learningRate: 0.01, steps: 300, start: [-4, 4] },
    claim: (r) => minimumReached(r, 'gd') === 1 && minimumReached(r, 'lbfgs') === 1,
  },
  {
    id: 'basins-b',
    title: 'The same function, another corner',
    text: 'From the opposite corner the optimisers settle in the valley at (3, 2): neighbouring starting points can lead to different answers.',
    scenario: { fn: 'himmelblau', learningRate: 0.01, steps: 300, start: [4, 4] },
    claim: (r) => minimumReached(r, 'gd') === 0 && minimumReached(r, 'lbfgs') === 0,
  },
  {
    id: 'momentum-overshoots',
    title: 'Momentum is not always safer',
    text: "On Beale's steep walls the velocity that helps elsewhere makes momentum and Nesterov overshoot and diverge at a learning rate where plain gradient descent still works.",
    scenario: { fn: 'beale', learningRate: 0.002, steps: 500 },
    claim: (r) =>
      pathOf(r, 'momentum').diverged && pathOf(r, 'nesterov').diverged && !pathOf(r, 'gd').diverged,
  },
];
