import {
  learningRateFinder,
  trainToolbox,
  type Activation,
  type InitScheme,
  type ToolOptimizer,
  type ToolboxResult,
} from '@sim/deep';
import { Matrix, createRng, makeClassification, type ClassificationSet } from '@sim/core';

export type ToolboxDataset = Extract<ClassificationSet, 'moons' | 'circles' | 'spirals' | 'xor'>;

export interface ToolboxScenario {
  dataset: ToolboxDataset;
  trainSize: number;
  noise: number;
  hidden: number;
  depth: number;
  activation: Activation;
  init: InitScheme;
  dropout: number;
  batchNorm: boolean;
  weightDecay: number;
  optimizer: ToolOptimizer;
  learningRate: number;
  epochs: number;
  patience: number;
  seed: number;
}

export const DEFAULT_TOOLBOX_LAB: ToolboxScenario = {
  dataset: 'moons',
  trainSize: 80,
  noise: 0.3,
  hidden: 32,
  depth: 2,
  activation: 'relu',
  init: 'he',
  dropout: 0,
  batchNorm: false,
  weightDecay: 0,
  optimizer: 'adam',
  learningRate: 0.01,
  epochs: 100,
  patience: 0,
  seed: 3,
};

export interface ToolboxRun {
  scenario: ToolboxScenario;
  result: ToolboxResult;
  /** Accuracy on a large fresh sample, with the weights the run ended with (best epoch if early stopping fired). */
  testAccuracy: number;
  testLoss: number;
  finalTrainAccuracy: number;
  gap: number;
  /** Learning-rate range test for the same architecture. */
  finder: { curve: { learningRate: number; loss: number }[]; suggested: number };
  grid: { size: number; labels: Int32Array };
  train: { x: Matrix; y: number[] };
}

const make = (s: ToolboxScenario, n: number, seed: number) => {
  const t = makeClassification(s.dataset, n, s.noise, createRng(seed));
  return { x: t.x, y: Array.from(t.y) };
};

export function runToolbox(s: ToolboxScenario): ToolboxRun {
  const train = make(s, s.trainSize, s.seed);
  const validation = make(s, 400, s.seed + 1);
  const test = make(s, 1500, s.seed + 2);
  const layers = [2, ...new Array<number>(s.depth).fill(s.hidden), 2];
  const config = {
    layers,
    activation: s.activation,
    init: s.init,
    dropout: s.dropout,
    batchNorm: s.batchNorm,
    weightDecay: s.weightDecay,
    seed: s.seed,
  };
  const result = trainToolbox(config, train, validation, {
    epochs: s.epochs,
    batchSize: 16,
    learningRate: s.learningRate,
    optimizer: s.optimizer,
    patience: s.patience,
    seed: s.seed + 3,
  });
  const ev = result.model.evaluate(test.x, test.y);
  const trainEval = result.model.evaluate(train.x, train.y);
  const finder = learningRateFinder(config, train, {
    low: 1e-4,
    high: s.optimizer === 'adam' ? 1 : 10,
    steps: 150,
    batchSize: 16,
    optimizer: s.optimizer,
    seed: s.seed,
  });
  // decision regions
  const size = 40;
  const g = new Matrix(size * size, 2);
  const bx = train.x;
  let x0 = Infinity,
    x1 = -Infinity,
    y0 = Infinity,
    y1 = -Infinity;
  for (let i = 0; i < bx.rows; i++) {
    x0 = Math.min(x0, bx.get(i, 0));
    x1 = Math.max(x1, bx.get(i, 0));
    y0 = Math.min(y0, bx.get(i, 1));
    y1 = Math.max(y1, bx.get(i, 1));
  }
  const pad = 0.5;
  for (let r = 0; r < size; r++)
    for (let c = 0; c < size; c++) {
      g.set(r * size + c, 0, x0 - pad + ((c + 0.5) / size) * (x1 - x0 + 2 * pad));
      g.set(r * size + c, 1, y1 + pad - ((r + 0.5) / size) * (y1 - y0 + 2 * pad));
    }
  return {
    scenario: s,
    result,
    testAccuracy: ev.accuracy,
    testLoss: ev.loss,
    finalTrainAccuracy: trainEval.accuracy,
    gap: trainEval.accuracy - ev.accuracy,
    finder,
    grid: { size, labels: result.model.predict(g) },
    train,
  };
}
