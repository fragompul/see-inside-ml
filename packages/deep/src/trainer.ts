import {
  type Matrix,
  selectRows,
  Optimizer,
  type OptimizerName,
  type Regularization,
  permutation,
  type Rng,
  learningRateAt,
  type LrSchedule,
} from '@sim/core';
import type { NeuralNetwork } from './network';

export interface TrainerConfig {
  optimizer: OptimizerName;
  lr: number;
  schedule: LrSchedule;
  batchSize: number;
  l1: number;
  l2: number;
}

export interface EpochStats {
  epoch: number;
  lr: number;
  trainLoss: number;
  testLoss: number;
  /** Accuracy (classification) or MSE (regression). */
  trainMetric: number;
  testMetric: number;
}

export interface TrainingData {
  Xtrain: Matrix;
  ytrain: Matrix;
  Xtest: Matrix;
  ytest: Matrix;
}

/** Mini-batch training loop with LR scheduling, regularisation and per-epoch statistics. */
export class Trainer {
  epoch = 0;
  private optimizer: Optimizer;

  constructor(
    readonly network: NeuralNetwork,
    readonly data: TrainingData,
    private config: TrainerConfig,
    private readonly rng: Rng,
  ) {
    assertConfig(config);
    this.optimizer = new Optimizer(config.optimizer);
  }

  get settings(): Readonly<TrainerConfig> {
    return this.config;
  }

  /** Hyper-parameters can change live; switching optimiser resets its internal state. */
  configure(patch: Partial<TrainerConfig>): void {
    const next = { ...this.config, ...patch };
    assertConfig(next);
    if (next.optimizer !== this.config.optimizer) this.optimizer = new Optimizer(next.optimizer);
    this.config = next;
  }

  get regularization(): Regularization {
    return { l1: this.config.l1, l2: this.config.l2 };
  }

  currentLr(): number {
    return learningRateAt(this.config.schedule, this.config.lr, this.epoch);
  }

  /** Runs one full pass over the training set and returns the resulting statistics. */
  runEpoch(): EpochStats {
    const { network, data } = this;
    const lr = this.currentLr();
    const reg = this.regularization;
    const n = data.Xtrain.rows;
    const order = permutation(n, this.rng);

    for (let start = 0; start < n; start += this.config.batchSize) {
      const idx = order.subarray(start, Math.min(n, start + this.config.batchSize));
      const xb = selectRows(data.Xtrain, idx);
      const yb = selectRows(data.ytrain, idx);
      const out = network.forward(xb);
      network.backward(yb, out);
      this.optimizer.step(network.parameters, lr, reg);
    }

    this.epoch += 1;
    return this.evaluate(lr);
  }

  evaluate(lr = this.currentLr()): EpochStats {
    const { network, data } = this;
    const reg = this.regularization;
    const train = network.evaluate(network.forward(data.Xtrain, false), data.ytrain, reg);
    const test = network.evaluate(network.forward(data.Xtest, false), data.ytest, reg);
    return {
      epoch: this.epoch,
      lr,
      trainLoss: train.loss,
      testLoss: test.loss,
      trainMetric: train.metric,
      testMetric: test.metric,
    };
  }
}

function assertConfig(c: TrainerConfig): void {
  if (!(c.lr > 0)) throw new RangeError('Learning rate must be positive');
  if (!Number.isInteger(c.batchSize) || c.batchSize < 1) {
    throw new RangeError('Batch size must be a positive integer');
  }
  if (c.l1 < 0 || c.l2 < 0) throw new RangeError('Regularisation rates cannot be negative');
}
