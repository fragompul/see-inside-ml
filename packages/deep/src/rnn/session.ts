import {
  Optimizer,
  OPTIMIZER_NAMES,
  type OptimizerName,
  createRng,
  permutation,
  type Rng,
} from '@sim/core';
import { RecurrentNet, validateRecurrent, type CellName, type RecurrentTrace } from './recurrent';
import {
  TASK_NAMES,
  generateSequences,
  labelOf,
  randomSequence,
  taskInfo,
  type SequenceSet,
  type TaskName,
} from './tasks';

export interface RnnLabConfig {
  cell: CellName;
  hidden: number;
  task: TaskName;
  seqLen: number;
  optimizer: OptimizerName;
  lr: number;
  batchSize: number;
  /** Gradients are rescaled so their global norm never exceeds this (0 = no clipping). */
  clip: number;
  /** Training sequences. */
  trainSize: number;
  seed: number;
}

export const DEFAULT_RNN_LAB: RnnLabConfig = {
  cell: 'lstm',
  hidden: 16,
  task: 'first',
  seqLen: 10,
  optimizer: 'adam',
  lr: 0.01,
  batchSize: 32,
  clip: 5,
  trainSize: 320,
  seed: 5,
};

export const RNN_LAB_LIMITS = {
  trainSize: [64, 960],
  batchSize: [4, 128],
  clip: [0, 20],
} as const;

const TEST_SIZE = 200;

export interface RnnEpochStats {
  epoch: number;
  trainLoss: number;
  testLoss: number;
  trainAccuracy: number;
  testAccuracy: number;
}

export function validateRnnLab(c: RnnLabConfig): void {
  if (!(TASK_NAMES as readonly string[]).includes(c.task)) throw new RangeError('Unknown task');
  const info = taskInfo[c.task];
  if (!Number.isInteger(c.seqLen) || c.seqLen < info.minLength || c.seqLen > info.maxLength)
    throw new RangeError(`Sequence length must be in [${info.minLength}, ${info.maxLength}]`);
  validateRecurrent({ cell: c.cell, hidden: c.hidden, vocab: info.vocab, classes: info.classes });
  if (!(OPTIMIZER_NAMES as readonly string[]).includes(c.optimizer))
    throw new RangeError('Unknown optimizer');
  if (!(c.lr > 0)) throw new RangeError('Learning rate must be positive');
  const [bLo, bHi] = RNN_LAB_LIMITS.batchSize;
  if (!Number.isInteger(c.batchSize) || c.batchSize < bLo || c.batchSize > bHi)
    throw new RangeError(`Batch size must be an integer in [${bLo}, ${bHi}]`);
  const [tLo, tHi] = RNN_LAB_LIMITS.trainSize;
  if (!Number.isInteger(c.trainSize) || c.trainSize < tLo || c.trainSize > tHi)
    throw new RangeError(`Training sequences must be an integer in [${tLo}, ${tHi}]`);
  const [cLo, cHi] = RNN_LAB_LIMITS.clip;
  if (!Number.isFinite(c.clip) || c.clip < cLo || c.clip > cHi)
    throw new RangeError(`Clip must be in [${cLo}, ${cHi}]`);
}

const stream = (seed: number, index: number) =>
  createRng((seed + Math.imul(index + 1, 0x9e3779b1)) >>> 0);

const dataKey = (c: RnnLabConfig) => JSON.stringify([c.task, c.seqLen, c.trainSize, c.seed]);
const netKey = (c: RnnLabConfig) => JSON.stringify([c.cell, c.hidden, c.task, c.seed]);

/** Builds a fresh network sized for a task. */
export function createNet(cell: CellName, hidden: number, task: TaskName, rng: Rng): RecurrentNet {
  const info = taskInfo[task];
  return new RecurrentNet({ cell, hidden, vocab: info.vocab, classes: info.classes }, rng);
}

/**
 * Average ‖dL/dh_t‖ over random sequences: how much learning signal reaches each step. Plain RNNs
 * (and, once trained badly, any cell) let it die away towards the start of the sequence.
 */
export function gradientProfile(
  net: RecurrentNet,
  task: TaskName,
  seqLen: number,
  samples: number,
  rng: Rng,
): Float64Array {
  const total = new Float64Array(seqLen);
  const saved = net.parameters.map((p) => Float64Array.from(p.grad));
  for (let n = 0; n < samples; n++) {
    const tokens = randomSequence(task, seqLen, rng);
    const { gradientNorms } = net.backward(net.run(tokens), labelOf(task, tokens), 1);
    for (let t = 0; t < seqLen; t++) total[t] = (total[t] as number) + (gradientNorms[t] as number);
  }
  net.parameters.forEach((p, i) => p.grad.set(saved[i] as Float64Array)); // leave training untouched
  return total.map((v) => v / samples);
}

/** A recurrent network, its data and its training loop behind one small API. */
export class RnnSession {
  private cfg: RnnLabConfig;
  private keys = { data: '', net: '' };
  network!: RecurrentNet;
  private optimizer!: Optimizer;
  private train!: SequenceSet;
  private test!: SequenceSet;
  private shuffle = createRng(0);
  private order: Uint32Array = new Uint32Array(0);
  private cursor = 0;
  epoch = 0;
  history: RnnEpochStats[] = [];

  constructor(config: RnnLabConfig) {
    validateRnnLab(config);
    this.cfg = { ...config };
    this.build(true);
  }

  get config(): Readonly<RnnLabConfig> {
    return this.cfg;
  }

  get trainCount(): number {
    return this.train.count;
  }

  get testCount(): number {
    return this.test.count;
  }

  get epochFraction(): number {
    return this.train.count > 0 ? this.cursor / this.train.count : 0;
  }

  /** Hyper-parameters apply live; data or architecture changes rebuild (and restart training). */
  update(patch: Partial<RnnLabConfig>): 'live' | 'rebuilt' {
    const next = { ...this.cfg, ...patch };
    // switching task changes what lengths are legal: nudge the length into range instead of failing
    if (patch.task && patch.seqLen === undefined) {
      const info = taskInfo[next.task];
      next.seqLen = Math.max(info.minLength, Math.min(info.maxLength, next.seqLen));
    }
    validateRnnLab(next);
    const dataChanged = dataKey(next) !== this.keys.data;
    const netChanged = netKey(next) !== this.keys.net;
    this.cfg = next;
    if (dataChanged || netChanged) {
      this.build(dataChanged);
      return 'rebuilt';
    }
    return 'live';
  }

  reset(): void {
    this.build(false);
  }

  step(epochs = 1): RnnEpochStats[] {
    const out: RnnEpochStats[] = [];
    while (out.length < epochs) out.push(...this.trainBatches(1));
    return out;
  }

  /** Advances training by `count` mini-batches; returns the stats of epochs completed meanwhile. */
  trainBatches(count: number): RnnEpochStats[] {
    const completed: RnnEpochStats[] = [];
    const { batchSize, lr, clip } = this.cfg;
    const net = this.network;
    const T = this.train.length;
    for (let n = 0; n < count; n++) {
      if (this.cursor === 0) this.order = permutation(this.train.count, this.shuffle);
      const idx = this.order.subarray(
        this.cursor,
        Math.min(this.order.length, this.cursor + batchSize),
      );
      net.zeroGrad();
      for (let k = 0; k < idx.length; k++) {
        const i = idx[k] as number;
        const tokens = this.train.tokens.subarray(i * T, (i + 1) * T);
        net.backward(net.run(tokens), this.train.labels[i] as number, 1 / idx.length);
      }
      if (clip > 0) this.clipGradients(clip);
      this.optimizer.step(net.parameters, lr);
      this.cursor += idx.length;
      if (this.cursor >= this.order.length) {
        this.cursor = 0;
        this.epoch += 1;
        const stats = { epoch: this.epoch, ...this.evaluate() };
        this.history.push(stats);
        completed.push(stats);
      }
    }
    return completed;
  }

  evaluate(): Omit<RnnEpochStats, 'epoch'> {
    const train = this.score(this.train);
    const test = this.score(this.test);
    return {
      trainLoss: train.loss,
      testLoss: test.loss,
      trainAccuracy: train.accuracy,
      testAccuracy: test.accuracy,
    };
  }

  /** Test accuracy of the current network on freshly drawn sequences of any length. */
  accuracyAtLength(length: number, count = 150): number {
    const set = generateSequences(this.cfg.task, length, count, stream(this.cfg.seed, 9));
    return this.score(set).accuracy;
  }

  /** Rows = true class, columns = predicted class, over the test set. */
  confusion(): number[][] {
    const classes = this.network.classes;
    const matrix = Array.from({ length: classes }, () => new Array<number>(classes).fill(0));
    const T = this.test.length;
    for (let i = 0; i < this.test.count; i++) {
      const probs = this.network.predict(this.test.tokens.subarray(i * T, (i + 1) * T));
      const guess = probs.indexOf(Math.max(...probs));
      const row = matrix[this.test.labels[i] as number] as number[];
      row[guess] = (row[guess] as number) + 1;
    }
    return matrix;
  }

  /** Every state and gate for one sequence (any length, e.g. one the user typed). */
  trace(tokens: ArrayLike<number>): RecurrentTrace {
    return this.network.run(tokens);
  }

  /** ‖dL/dh_t‖ for one sequence, without disturbing training. */
  gradientNorms(tokens: ArrayLike<number>): Float64Array {
    const saved = this.network.parameters.map((p) => Float64Array.from(p.grad));
    const { gradientNorms } = this.network.backward(
      this.network.run(tokens),
      labelOf(this.cfg.task, tokens),
      1,
    );
    this.network.parameters.forEach((p, i) => p.grad.set(saved[i] as Float64Array));
    return gradientNorms;
  }

  /** Mean gradient profile over random training-style sequences of the current length. */
  profile(samples = 24): Float64Array {
    return gradientProfile(
      this.network,
      this.cfg.task,
      this.cfg.seqLen,
      samples,
      stream(this.cfg.seed, 7),
    );
  }

  sample(index: number, set: 'test' | 'train' = 'test'): { tokens: Int32Array; label: number } {
    const data = set === 'test' ? this.test : this.train;
    if (!Number.isInteger(index) || index < 0 || index >= data.count)
      throw new RangeError('Sample out of range');
    const T = data.length;
    return {
      tokens: data.tokens.slice(index * T, (index + 1) * T),
      label: data.labels[index] as number,
    };
  }

  private clipGradients(maxNorm: number): void {
    let sq = 0;
    for (const p of this.network.parameters) for (const g of p.grad) sq += g * g;
    const norm = Math.sqrt(sq);
    if (norm > maxNorm) {
      const k = maxNorm / norm;
      for (const p of this.network.parameters)
        for (let i = 0; i < p.grad.length; i++) p.grad[i] = (p.grad[i] as number) * k;
    }
  }

  private build(regenerate: boolean): void {
    const c = this.cfg;
    if (regenerate) {
      this.train = generateSequences(c.task, c.seqLen, c.trainSize, stream(c.seed, 0));
      this.test = generateSequences(c.task, c.seqLen, TEST_SIZE, stream(c.seed, 1));
    }
    this.network = createNet(c.cell, c.hidden, c.task, stream(c.seed, 2));
    this.optimizer = new Optimizer(c.optimizer);
    this.shuffle = stream(c.seed, 3);
    this.epoch = 0;
    this.cursor = 0;
    this.history = [];
    this.keys = { data: dataKey(c), net: netKey(c) };
  }

  private score(set: SequenceSet): { loss: number; accuracy: number } {
    const T = set.length;
    let loss = 0;
    let correct = 0;
    for (let i = 0; i < set.count; i++) {
      const probs = this.network.predict(set.tokens.subarray(i * T, (i + 1) * T));
      const label = set.labels[i] as number;
      loss -= Math.log(Math.max(probs[label] as number, 1e-300));
      if (probs.indexOf(Math.max(...probs)) === label) correct += 1;
    }
    return { loss: loss / set.count, accuracy: correct / set.count };
  }
}
