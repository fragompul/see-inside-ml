import { createRng } from '@sim/core';
import { NeuralOde, trainNeuralOde, type Solver } from '@sim/deep';

export type OdeDataset = 'circles' | 'moons';

export const DATASET_LABEL: Record<OdeDataset, string> = {
  circles: 'A ring around a disc',
  moons: 'Two moons',
};

export interface OdeScenario {
  dataset: OdeDataset;
  solver: Solver;
  /** Solver steps for unit time, used while training and for the model's own predictions. */
  steps: number;
}

export const DEFAULT_ODE: OdeScenario = { dataset: 'circles', solver: 'rk4', steps: 8 };

export const GRID = 22;
export const RANGE = 2.2;
const FINE_STEPS = 128;
const SEEDS = [1, 2];

function sample(
  dataset: OdeDataset,
  n: number,
  seed: number,
): { xs: Float64Array; ys: Float64Array } {
  const rng = createRng(seed);
  const xs = new Float64Array(2 * n);
  const ys = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    const second = i % 2;
    if (dataset === 'circles') {
      const r = (second ? 1.5 : 0.5) + 0.1 * rng.normal();
      const a = rng.next() * 2 * Math.PI;
      xs[2 * i] = r * Math.cos(a);
      xs[2 * i + 1] = r * Math.sin(a);
    } else {
      const a = rng.next() * Math.PI;
      xs[2 * i] = (second ? 1 - Math.cos(a) : Math.cos(a)) - 0.5 + 0.1 * rng.normal();
      xs[2 * i + 1] = (second ? 0.5 - Math.sin(a) : Math.sin(a)) - 0.25 + 0.1 * rng.normal();
    }
    ys[i] = second;
  }
  return { xs, ys };
}

export interface OdeResult {
  scenario: OdeScenario;
  /** Test accuracy of the model with its own solver, and of the same weights with a very fine accurate solver (means over two runs). */
  own: number;
  fine: number;
  /** Average distance between the final states from the two solvers. */
  stateError: number;
  /** Probability of class 1 over a grid (row 0 at the top), using the model's own solver. */
  map: number[];
  /** The velocity at grid points: [x, y, vx, vy]. */
  field: number[][];
  /** Paths of 36 points (first run), each as a list of [x, y] over time, with the class of the point. */
  paths: { points: number[][]; label: number }[];
  data: { xs: number[]; ys: number[] };
  history: { epoch: number; loss: number; accuracy: number }[];
}

const cache = new Map<string, OdeResult>();

export function runOde(s: OdeScenario): OdeResult {
  const key = JSON.stringify(s);
  const hit = cache.get(key);
  if (hit) return hit;
  const train = sample(s.dataset, 300, 1);
  const test = sample(s.dataset, 300, 2);
  const config = { dim: 2, hidden: 8, solver: s.solver, steps: s.steps };
  let own = 0;
  let fine = 0;
  let stateError = 0;
  let first: ReturnType<typeof trainNeuralOde> | null = null;
  for (const seed of SEEDS) {
    const run = trainNeuralOde(config, train, { epochs: 250, learningRate: 0.03, seed });
    if (!first) first = run;
    const accurate = new NeuralOde(
      { dim: 2, hidden: 8, solver: 'rk4', steps: FINE_STEPS },
      createRng(1),
    );
    accurate.parameters.forEach((p, i) => p.data.set((run.model.parameters[i] as typeof p).data));
    own += run.model.accuracy(test.xs, test.ys) / SEEDS.length;
    fine += accurate.accuracy(test.xs, test.ys) / SEEDS.length;
    for (let i = 0; i < 100; i++) {
      const a = run.model
        .trajectory(test.xs[2 * i] as number, test.xs[2 * i + 1] as number)
        .pop() as Float64Array;
      const b = accurate
        .trajectory(test.xs[2 * i] as number, test.xs[2 * i + 1] as number)
        .pop() as Float64Array;
      stateError +=
        Math.hypot((a[0] as number) - (b[0] as number), (a[1] as number) - (b[1] as number)) /
        (100 * SEEDS.length);
    }
  }
  const model = (first as NonNullable<typeof first>).model;
  const map: number[] = [];
  const field: number[][] = [];
  for (let r = 0; r < GRID; r++)
    for (let c = 0; c < GRID; c++)
      map.push(
        model.probability(
          -RANGE + ((c + 0.5) * 2 * RANGE) / GRID,
          RANGE - ((r + 0.5) * 2 * RANGE) / GRID,
        ),
      );
  for (let r = 0; r < 9; r++)
    for (let c = 0; c < 9; c++) {
      const x = -RANGE + ((c + 0.5) * 2 * RANGE) / 9;
      const y = RANGE - ((r + 0.5) * 2 * RANGE) / 9;
      const v = model.velocity([x, y]);
      field.push([x, y, v[0] as number, v[1] as number]);
    }
  const paths = Array.from({ length: 36 }, (_, k) => {
    const i = k * 5;
    return {
      points: model
        .trajectory(train.xs[2 * i] as number, train.xs[2 * i + 1] as number)
        .map((z) => [z[0] as number, z[1] as number]),
      label: train.ys[i] as number,
    };
  });
  const out: OdeResult = {
    scenario: s,
    own,
    fine,
    stateError,
    map,
    field,
    paths,
    data: { xs: Array.from(train.xs), ys: Array.from(train.ys) },
    history: (first as NonNullable<typeof first>).history,
  };
  cache.set(key, out);
  return out;
}
