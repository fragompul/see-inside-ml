import { describe, expect, it } from 'vitest';
import fixtures from '../fixtures/parity.json';
import {
  type ActivationName,
  type LossType,
  type Problem,
  Matrix,
  Optimizer,
  type OptimizerName,
  createRng,
  learningRateAt,
} from '@sim/core';
import { NeuralNetwork, type LayerWeights } from './network';

/**
 * Golden-fixture parity: the TypeScript engine must reproduce the original NumPy implementation
 * (forward, gradients, losses, optimiser trajectories) on identical inputs and initial weights.
 * Fixtures come from reference/python/generate_fixtures.py.
 */

interface Case {
  name: string;
  config: {
    layerSizes: number[];
    activations: ActivationName[];
    problem: Problem;
    lossType: string;
    skipConnections: boolean;
  };
  train: { optimizer: OptimizerName; lr: number; steps: number; l1: number; l2: number };
  X: number[][];
  y: number[][];
  initialWeights: LayerWeights[];
  expected: {
    firstOutput: number[][];
    firstGrads: { dW: number[][]; db: number[] }[];
    losses: number[];
    finalOutput: number[][];
    finalLoss: number;
    finalMetric: number;
    finalWeights: LayerWeights[];
  };
}

const cases = fixtures.cases as unknown as Case[];
const TOL = 1e-8;

function close(actual: number, expected: number, label: string) {
  const bound = TOL + TOL * Math.abs(expected);
  if (!(Math.abs(actual - expected) <= bound)) {
    throw new Error(
      `${label}: expected ${expected}, got ${actual} (Δ=${Math.abs(actual - expected)})`,
    );
  }
}

function closeMatrix(actual: number[][], expected: number[][], label: string) {
  expect(actual.length, `${label} rows`).toBe(expected.length);
  expected.forEach((row, i) => {
    expect(actual[i]?.length, `${label} cols`).toBe(row.length);
    row.forEach((value, j) => close(actual[i]?.[j] as number, value, `${label}[${i}][${j}]`));
  });
}

function build(c: Case) {
  const loss: LossType | undefined =
    c.config.lossType === 'default' ? undefined : (c.config.lossType as LossType);
  const net = new NeuralNetwork(
    {
      layerSizes: c.config.layerSizes,
      activations: c.config.activations,
      problem: c.config.problem,
      loss,
      skipConnections: c.config.skipConnections,
    },
    createRng(0),
  );
  net.setWeights(c.initialWeights);
  return net;
}

describe('parity with the NumPy reference', () => {
  it('ships a meaningful fixture set', () => {
    expect(cases.length).toBeGreaterThanOrEqual(20);
  });

  describe.each(cases.map((c) => [c.name, c] as const))('%s', (_name, c) => {
    const X = Matrix.from(c.X);
    const y = Matrix.from(c.y);

    it('forward pass and first-step gradients match', () => {
      const net = build(c);
      const out = net.forward(X);
      closeMatrix(out.toArray(), c.expected.firstOutput, 'output');

      net.backward(y, out);
      net.layers.forEach((layer, i) => {
        const dW = new Matrix(layer.inputDim, layer.outputDim, layer.weightParam.grad.slice());
        closeMatrix(dW.toArray(), c.expected.firstGrads[i]?.dW as number[][], `dW[${i}]`);
        closeMatrix(
          [Array.from(layer.biasParam.grad)],
          [c.expected.firstGrads[i]?.db as number[]],
          `db[${i}]`,
        );
      });
    });

    it('training trajectory, final weights and metrics match', () => {
      const net = build(c);
      const optimizer = new Optimizer(c.train.optimizer);
      const reg = { l1: c.train.l1, l2: c.train.l2 };
      const losses: number[] = [];

      for (let step = 0; step < c.train.steps; step++) {
        const out = net.forward(X);
        losses.push(net.evaluate(out, y, reg).loss);
        net.backward(y, out);
        optimizer.step(net.parameters, c.train.lr, reg);
      }

      losses.forEach((loss, i) => close(loss, c.expected.losses[i] as number, `loss[${i}]`));

      const finalOut = net.forward(X);
      closeMatrix(finalOut.toArray(), c.expected.finalOutput, 'final output');
      const evaluation = net.evaluate(finalOut, y, reg);
      close(evaluation.loss, c.expected.finalLoss, 'final loss');
      close(evaluation.metric, c.expected.finalMetric, 'final metric');

      net.getWeights().forEach((w, i) => {
        closeMatrix(w.W, c.expected.finalWeights[i]?.W as number[][], `W[${i}]`);
        closeMatrix([w.b], [c.expected.finalWeights[i]?.b as number[]], `b[${i}]`);
      });
    });
  });

  it('learning-rate schedules match', () => {
    const { baseLr, epochs, exponential, cosine } = fixtures.schedules;
    epochs.forEach((epoch, i) => {
      close(learningRateAt('constant', baseLr, epoch), baseLr, 'constant');
      close(learningRateAt('exponential', baseLr, epoch), exponential[i] as number, `exp@${epoch}`);
      close(learningRateAt('cosine', baseLr, epoch), cosine[i] as number, `cos@${epoch}`);
    });
  });
});
