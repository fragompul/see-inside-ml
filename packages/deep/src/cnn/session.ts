import {
  Matrix,
  Optimizer,
  OPTIMIZER_NAMES,
  type OptimizerName,
  createRng,
  permutation,
} from '@sim/core';
import { DIGIT_CLASSES, IMAGE_SIZE, generateDigits, type DigitSet } from './digits';
import {
  CnnNetwork,
  argmax,
  occlusionMap,
  validateCnn,
  type CnnConfig,
  type CnnTrace,
} from './network';

export interface CnnLabConfig extends CnnConfig {
  optimizer: OptimizerName;
  lr: number;
  batchSize: number;
  /** Training images per digit. */
  perClass: number;
  /** Pixel-noise level of the generated images (0 – 0.3). */
  noise: number;
  seed: number;
}

export const DEFAULT_CNN_LAB: CnnLabConfig = {
  filters1: 6,
  filters2: 12,
  kernel: 3,
  optimizer: 'adam',
  lr: 0.003,
  batchSize: 32,
  perClass: 60,
  noise: 0.1,
  seed: 7,
};

export const CNN_LAB_LIMITS = {
  perClass: [10, 150],
  noise: [0, 0.3],
  batchSize: [1, 128],
} as const;

const TEST_PER_CLASS = 30;

export interface CnnEpochStats {
  epoch: number;
  trainLoss: number;
  testLoss: number;
  trainAccuracy: number;
  testAccuracy: number;
}

export function validateLab(c: CnnLabConfig): void {
  validateCnn(c);
  const within = (v: number, [lo, hi]: readonly [number, number], name: string) => {
    if (!Number.isFinite(v) || v < lo || v > hi)
      throw new RangeError(`${name} must be in [${lo}, ${hi}]`);
  };
  if (!(OPTIMIZER_NAMES as readonly string[]).includes(c.optimizer))
    throw new RangeError('Unknown optimizer');
  if (!(c.lr > 0)) throw new RangeError('Learning rate must be positive');
  within(c.perClass, CNN_LAB_LIMITS.perClass, 'perClass');
  within(c.noise, CNN_LAB_LIMITS.noise, 'noise');
  within(c.batchSize, CNN_LAB_LIMITS.batchSize, 'batchSize');
  if (!Number.isInteger(c.batchSize) || !Number.isInteger(c.perClass))
    throw new RangeError('Sizes must be integers');
}

const stream = (seed: number, index: number) =>
  createRng((seed + Math.imul(index + 1, 0x9e3779b1)) >>> 0);

const dataKey = (c: CnnLabConfig) => JSON.stringify([c.perClass, c.noise, c.seed]);
const netKey = (c: CnnLabConfig) => JSON.stringify([c.filters1, c.filters2, c.kernel, c.seed]);

const pixels = IMAGE_SIZE * IMAGE_SIZE;

/** A CNN, its data and its training loop behind one small, serialisable API. */
export class CnnSession {
  private cfg: CnnLabConfig;
  private keys = { data: '', net: '' };
  network!: CnnNetwork;
  private optimizer!: Optimizer;
  private train!: DigitSet;
  private test!: DigitSet;
  private shuffle = createRng(0);
  /** Position inside the current shuffled epoch, so training can advance a few batches at a time. */
  private order: Uint32Array = new Uint32Array(0);
  private cursor = 0;
  epoch = 0;
  history: CnnEpochStats[] = [];

  constructor(config: CnnLabConfig) {
    validateLab(config);
    this.cfg = { ...config };
    this.build(true);
  }

  get config(): Readonly<CnnLabConfig> {
    return this.cfg;
  }

  get trainCount(): number {
    return this.train.count;
  }

  /** How far through the current epoch training is, 0..1. */
  get epochFraction(): number {
    return this.train.count > 0 ? this.cursor / this.train.count : 0;
  }

  get testCount(): number {
    return this.test.count;
  }

  /** Hyper-parameters apply live; data or architecture changes rebuild (and restart training). */
  update(patch: Partial<CnnLabConfig>): 'live' | 'rebuilt' {
    const next = { ...this.cfg, ...patch };
    validateLab(next);
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

  /** Trains for `epochs` complete passes over the training set. */
  step(epochs = 1): CnnEpochStats[] {
    const out: CnnEpochStats[] = [];
    while (out.length < epochs) out.push(...this.trainBatches(1));
    return out;
  }

  /**
   * Advances training by `count` mini-batches (an epoch is `ceil(train / batchSize)` of them).
   * Lets the UI animate the weights smoothly. Returns the stats of every epoch completed meanwhile.
   */
  trainBatches(count: number): CnnEpochStats[] {
    const completed: CnnEpochStats[] = [];
    const { batchSize, lr } = this.cfg;
    for (let n = 0; n < count; n++) {
      if (this.cursor === 0) this.order = permutation(this.train.count, this.shuffle);
      const idx = this.order.subarray(
        this.cursor,
        Math.min(this.order.length, this.cursor + batchSize),
      );
      const { images, targets } = this.batch(this.train, idx);
      const probs = this.network.forward(images, idx.length);
      this.network.backward(targets, probs);
      this.optimizer.step(this.network.parameters, lr);
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

  /** Loss and accuracy on both sets with the current weights. */
  evaluate(): Omit<CnnEpochStats, 'epoch'> {
    const train = this.score(this.train);
    const test = this.score(this.test);
    return {
      trainLoss: train.loss,
      testLoss: test.loss,
      trainAccuracy: train.accuracy,
      testAccuracy: test.accuracy,
    };
  }

  /** Rows = true digit, columns = predicted digit, over the test set. */
  confusion(): number[][] {
    const matrix = Array.from({ length: DIGIT_CLASSES }, () =>
      new Array<number>(DIGIT_CLASSES).fill(0),
    );
    const probs = this.predictAll(this.test);
    for (let i = 0; i < this.test.count; i++) {
      const row = matrix[this.test.labels[i] as number] as number[];
      const guess = argmax(probs.data.subarray(i * DIGIT_CLASSES, (i + 1) * DIGIT_CLASSES));
      row[guess] = (row[guess] as number) + 1;
    }
    return matrix;
  }

  /** Every intermediate tensor for one image (any 16×16 picture, e.g. one the user drew). */
  trace(image: Float64Array): CnnTrace {
    if (image.length !== pixels) throw new RangeError(`Images have ${pixels} pixels`);
    return this.network.trace(image);
  }

  occlusion(image: Float64Array, target: number, patch = 3): Float64Array {
    return occlusionMap(this.network, image, target, patch);
  }

  /** One example image from the test (or train) set. */
  sample(index: number, set: 'test' | 'train' = 'test'): { image: Float64Array; label: number } {
    const data = set === 'test' ? this.test : this.train;
    if (!Number.isInteger(index) || index < 0 || index >= data.count)
      throw new RangeError('Sample out of range');
    return {
      image: data.images.slice(index * pixels, (index + 1) * pixels),
      label: data.labels[index] as number,
    };
  }

  /** Learned kernels: `conv1[f]` is k×k; `conv2[f][c]` is k×k for input channel c. */
  kernels(): { conv1: Float64Array[]; conv2: Float64Array[][] } {
    const { conv1, conv2 } = this.network;
    const kk = conv1.k * conv1.k;
    return {
      conv1: Array.from({ length: conv1.outC }, (_, f) =>
        conv1.weights.slice(f * kk, (f + 1) * kk),
      ),
      conv2: Array.from({ length: conv2.outC }, (_, f) =>
        Array.from({ length: conv2.inC }, (_, c) =>
          conv2.weights.slice((f * conv2.inC + c) * kk, (f * conv2.inC + c + 1) * kk),
        ),
      ),
    };
  }

  private build(regenerate: boolean): void {
    const c = this.cfg;
    if (regenerate) {
      const style = { jitter: 0.025, tilt: 0.2, noise: c.noise };
      this.train = generateDigits(c.perClass, stream(c.seed, 0), style);
      this.test = generateDigits(TEST_PER_CLASS, stream(c.seed, 1), style);
    }
    this.network = new CnnNetwork(c, stream(c.seed, 2));
    this.optimizer = new Optimizer(c.optimizer);
    this.shuffle = stream(c.seed, 3);
    this.epoch = 0;
    this.cursor = 0;
    this.history = [];
    this.keys = { data: dataKey(c), net: netKey(c) };
  }

  private batch(set: DigitSet, idx: ArrayLike<number>): { images: Float64Array; targets: Matrix } {
    const images = new Float64Array(idx.length * pixels);
    const targets = new Matrix(idx.length, DIGIT_CLASSES);
    for (let n = 0; n < idx.length; n++) {
      const src = idx[n] as number;
      images.set(set.images.subarray(src * pixels, (src + 1) * pixels), n * pixels);
      targets.set(n, set.labels[src] as number, 1);
    }
    return { images, targets };
  }

  private predictAll(set: DigitSet): Matrix {
    const out = new Matrix(set.count, DIGIT_CLASSES);
    const chunk = 100;
    for (let start = 0; start < set.count; start += chunk) {
      const n = Math.min(chunk, set.count - start);
      const probs = this.network.forward(
        set.images.subarray(start * pixels, (start + n) * pixels),
        n,
      );
      out.data.set(probs.data, start * DIGIT_CLASSES);
    }
    return out;
  }

  private score(set: DigitSet): { loss: number; accuracy: number } {
    const probs = this.predictAll(set);
    const targets = new Matrix(set.count, DIGIT_CLASSES);
    let correct = 0;
    for (let i = 0; i < set.count; i++) {
      const label = set.labels[i] as number;
      targets.set(i, label, 1);
      if (argmax(probs.data.subarray(i * DIGIT_CLASSES, (i + 1) * DIGIT_CLASSES)) === label)
        correct += 1;
    }
    return { loss: this.network.loss(probs, targets), accuracy: correct / set.count };
  }
}
