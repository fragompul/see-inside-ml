import { Matrix, createRng } from '@sim/core';
import { DEFAULT_FLOW_TRAINING, sampleReal, trainFlow, type GanDataset } from '@sim/deep';

export type FlowDataset = Extract<GanDataset, 'moons' | 'ring' | 'spiral' | 'gaussians8'>;

export const FLOW_DATASETS: Record<FlowDataset, string> = {
  moons: 'Two moons',
  ring: 'A ring',
  spiral: 'A spiral',
  gaussians8: 'Eight clusters',
};

export interface FlowScenario {
  dataset: FlowDataset;
  layers: number;
  hidden: number;
  steps: number;
  seed: number;
}

export const DEFAULT_FLOW_LAB: FlowScenario = {
  dataset: 'moons',
  layers: 6,
  hidden: 24,
  steps: 600,
  seed: 1,
};

export const GRID = 56;
export const HALF = 3.4;

export interface FlowLabRun {
  scenario: FlowScenario;
  /** Training points (n x 2) and fresh samples drawn from the flow. */
  data: number[];
  samples: number[];
  /** Density on a GRID x GRID grid spanning [-HALF, HALF]^2, row-major with x varying fastest. */
  density: number[];
  /** The same 300 data points after each layer (index 0 is the data, the last is the latent space). */
  stages: number[][];
  history: { step: number; nll: number }[];
  /** Average negative log-likelihood in nats per point on the training points and on fresh points. */
  nll: number;
  heldOutNll: number;
  /** The same for the best single Gaussian. */
  gaussianNll: number;
  /** Largest error in recovering the data from its latent code. */
  invertError: number;
}

const gaussianNllOf = (m: Matrix): number => {
  const n = m.rows;
  let mx = 0;
  let my = 0;
  for (let i = 0; i < n; i++) {
    mx += (m.data[2 * i] as number) / n;
    my += (m.data[2 * i + 1] as number) / n;
  }
  let sxx = 0;
  let syy = 0;
  let sxy = 0;
  for (let i = 0; i < n; i++) {
    const dx = (m.data[2 * i] as number) - mx;
    const dy = (m.data[2 * i + 1] as number) - my;
    sxx += (dx * dx) / n;
    syy += (dy * dy) / n;
    sxy += (dx * dy) / n;
  }
  return Math.log(2 * Math.PI * Math.E) + 0.5 * Math.log(sxx * syy - sxy * sxy);
};

const cache = new Map<string, FlowLabRun>();

export function runFlowLab(s: FlowScenario): FlowLabRun {
  const key = JSON.stringify(s);
  const hit = cache.get(key);
  if (hit) return hit;
  const data = new Matrix(600, 2, sampleReal(s.dataset, 600, createRng(s.seed * 10 + 3)));
  const heldOut = new Matrix(600, 2, sampleReal(s.dataset, 600, createRng(s.seed * 10 + 4)));
  const run = trainFlow(
    data,
    { layers: s.layers, hidden: s.hidden, scale: 2 },
    { ...DEFAULT_FLOW_TRAINING, steps: s.steps, seed: s.seed },
    Math.max(1, Math.floor(s.steps / 12)),
  );
  const flow = run.flow;
  const grid = new Matrix(GRID * GRID, 2);
  for (let j = 0; j < GRID; j++)
    for (let i = 0; i < GRID; i++) {
      grid.data[(j * GRID + i) * 2] = -HALF + ((i + 0.5) * 2 * HALF) / GRID;
      grid.data[(j * GRID + i) * 2 + 1] = -HALF + ((j + 0.5) * 2 * HALF) / GRID;
    }
  const density = Array.from(flow.logProb(grid), Math.exp);
  const subset = new Matrix(300, 2, data.data.slice(0, 600));
  const forward = flow.forward(subset, false);
  const back = flow.inverse(forward.z);
  let invertError = 0;
  for (let i = 0; i < subset.data.length; i++)
    invertError = Math.max(
      invertError,
      Math.abs((back.data[i] as number) - (subset.data[i] as number)),
    );
  const out: FlowLabRun = {
    scenario: s,
    data: Array.from(data.data.slice(0, 600)),
    samples: Array.from(flow.sample(400, createRng(s.seed + 77)).data),
    density,
    stages: forward.steps.map((m) => Array.from(m.data)),
    history: run.history,
    nll: flow.loss(data, false),
    heldOutNll: flow.loss(heldOut, false),
    gaussianNll: gaussianNllOf(data),
    invertError,
  };
  cache.set(key, out);
  return out;
}
