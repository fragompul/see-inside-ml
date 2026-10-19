import { IMAGE_SIZE, generateDigits, type DigitSet } from '../cnn/digits';
import {
  Matrix,
  Optimizer,
  OPTIMIZER_NAMES,
  type OptimizerName,
  createRng,
  permutation,
  type Rng,
} from '@sim/core';
import { AutoencoderNet, validateAutoencoder, type AutoencoderConfig } from './network';

export interface AutoencoderLabConfig extends AutoencoderConfig {
  /** Weight of the KL term (VAE only). */
  beta: number;
  /** Standard deviation of the Gaussian noise added to the inputs while training (denoising). */
  corruption: number;
  optimizer: OptimizerName;
  lr: number;
  batchSize: number;
  /** Training images per digit. */
  perClass: number;
  seed: number;
}

export const DEFAULT_AE_LAB: AutoencoderLabConfig = {
  mode: 'ae',
  latent: 2,
  hidden: 64,
  beta: 1,
  corruption: 0,
  optimizer: 'adam',
  lr: 0.003,
  batchSize: 32,
  perClass: 60,
  seed: 3,
};

export const AE_LAB_LIMITS = {
  beta: [0, 8],
  corruption: [0, 0.8],
  batchSize: [4, 128],
  perClass: [10, 150],
} as const;

const TEST_PER_CLASS = 30;
const PIXELS = IMAGE_SIZE * IMAGE_SIZE;

export interface AutoencoderEpochStats {
  epoch: number;
  trainRecon: number;
  testRecon: number;
  trainKl: number;
  testKl: number;
  /** Reconstruction + β·KL on each set. */
  trainTotal: number;
  testTotal: number;
}

export function validateAutoencoderLab(c: AutoencoderLabConfig): void {
  validateAutoencoder(c);
  const within = (v: number, [lo, hi]: readonly [number, number], name: string) => {
    if (!Number.isFinite(v) || v < lo || v > hi)
      throw new RangeError(`${name} must be in [${lo}, ${hi}]`);
  };
  if (!(OPTIMIZER_NAMES as readonly string[]).includes(c.optimizer))
    throw new RangeError('Unknown optimizer');
  if (!(c.lr > 0)) throw new RangeError('Learning rate must be positive');
  within(c.beta, AE_LAB_LIMITS.beta, 'Beta');
  within(c.corruption, AE_LAB_LIMITS.corruption, 'Corruption');
  within(c.batchSize, AE_LAB_LIMITS.batchSize, 'Batch size');
  within(c.perClass, AE_LAB_LIMITS.perClass, 'Images per digit');
  if (!Number.isInteger(c.batchSize) || !Number.isInteger(c.perClass))
    throw new RangeError('Sizes must be integers');
}

const stream = (seed: number, index: number) =>
  createRng((seed + Math.imul(index + 1, 0x9e3779b1)) >>> 0);

const dataKey = (c: AutoencoderLabConfig) => JSON.stringify([c.perClass, c.seed]);
const netKey = (c: AutoencoderLabConfig) => JSON.stringify([c.mode, c.latent, c.hidden, c.seed]);

/** Adds Gaussian noise of standard deviation `level` and clamps back into [0, 1]. */
export function corruptImage(image: ArrayLike<number>, level: number, rng: Rng): Float64Array {
  const out = new Float64Array(image.length);
  for (let i = 0; i < image.length; i++)
    out[i] = Math.min(1, Math.max(0, (image[i] as number) + rng.normal() * level));
  return out;
}

/** Autoencoder / VAE, its data and its training loop behind one small API. */
export class AutoencoderSession {
  private cfg: AutoencoderLabConfig;
  private keys = { data: '', net: '' };
  network!: AutoencoderNet;
  private optimizer!: Optimizer;
  private train!: DigitSet;
  private test!: DigitSet;
  private shuffle = createRng(0);
  private noise = createRng(0);
  private order: Uint32Array = new Uint32Array(0);
  private cursor = 0;
  epoch = 0;
  history: AutoencoderEpochStats[] = [];

  constructor(config: AutoencoderLabConfig) {
    validateAutoencoderLab(config);
    this.cfg = { ...config };
    this.build(true);
  }

  get config(): Readonly<AutoencoderLabConfig> {
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
  update(patch: Partial<AutoencoderLabConfig>): 'live' | 'rebuilt' {
    const next = { ...this.cfg, ...patch };
    validateAutoencoderLab(next);
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

  step(epochs = 1): AutoencoderEpochStats[] {
    const out: AutoencoderEpochStats[] = [];
    while (out.length < epochs) out.push(...this.trainBatches(1));
    return out;
  }

  /** Advances training by `count` mini-batches; returns the epochs completed meanwhile. */
  trainBatches(count: number): AutoencoderEpochStats[] {
    const completed: AutoencoderEpochStats[] = [];
    const { batchSize, lr, beta, corruption, latent, mode } = this.cfg;
    const net = this.network;
    for (let n = 0; n < count; n++) {
      if (this.cursor === 0) this.order = permutation(this.train.count, this.shuffle);
      const idx = this.order.subarray(
        this.cursor,
        Math.min(this.order.length, this.cursor + batchSize),
      );
      const clean = this.batch(this.train, idx);
      const input = corruption > 0 ? this.corrupt(clean, corruption) : clean;
      const eps =
        mode === 'vae'
          ? new Matrix(
              idx.length,
              latent,
              Float64Array.from({ length: idx.length * latent }, () => this.noise.normal()),
            )
          : null;
      const out = net.forward(input, eps);
      net.backward(clean, out, beta);
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

  /** Losses on both sets with clean inputs and the mean code (no sampling). */
  evaluate(): Omit<AutoencoderEpochStats, 'epoch'> {
    const train = this.score(this.train);
    const test = this.score(this.test);
    return {
      trainRecon: train.recon,
      testRecon: test.recon,
      trainKl: train.kl,
      testKl: test.kl,
      trainTotal: train.total,
      testTotal: test.total,
    };
  }

  /** Mean codes of a whole set, `count × latent` row-major, with the digit labels. */
  latents(set: 'test' | 'train' = 'test'): { mu: Float64Array; labels: Uint8Array; count: number } {
    const data = set === 'test' ? this.test : this.train;
    const { latent } = this.cfg;
    const mu = new Float64Array(data.count * latent);
    const chunk = 100;
    for (let start = 0; start < data.count; start += chunk) {
      const n = Math.min(chunk, data.count - start);
      const x = new Matrix(n, PIXELS, data.images.slice(start * PIXELS, (start + n) * PIXELS));
      mu.set(this.network.forward(x, null, false).mu.data, start * latent);
    }
    return { mu, labels: data.labels, count: data.count };
  }

  encode(image: Float64Array) {
    return this.network.encode(image);
  }

  decode(z: ArrayLike<number>): Float64Array {
    return this.network.decode(z);
  }

  decodeMany(codes: Float64Array, count: number): Float64Array {
    return this.network.decodeMany(codes, count);
  }

  reconstruct(image: Float64Array): Float64Array {
    return this.network.reconstruct(image);
  }

  /** Decodes `count` codes drawn from the prior N(0, I): what the model "imagines". */
  generate(count: number, rng: Rng): Float64Array[] {
    return Array.from({ length: count }, () =>
      this.decode(Array.from({ length: this.cfg.latent }, () => rng.normal())),
    );
  }

  /** One example image from the test (or train) set. */
  sample(index: number, set: 'test' | 'train' = 'test'): { image: Float64Array; label: number } {
    const data = set === 'test' ? this.test : this.train;
    if (!Number.isInteger(index) || index < 0 || index >= data.count)
      throw new RangeError('Sample out of range');
    return {
      image: data.images.slice(index * PIXELS, (index + 1) * PIXELS),
      label: data.labels[index] as number,
    };
  }

  /** Indices of every test image showing `digit` (so the UI can offer "show me a 7"). */
  testIndicesOf(digit: number): number[] {
    const out: number[] = [];
    this.test.labels.forEach((label, i) => {
      if (label === digit) out.push(i);
    });
    if (out.length === 0) throw new RangeError('No such digit');
    return out;
  }

  private corrupt(clean: Matrix, level: number): Matrix {
    return new Matrix(clean.rows, clean.cols, corruptImage(clean.data, level, this.noise));
  }

  private build(regenerate: boolean): void {
    const c = this.cfg;
    if (regenerate) {
      this.train = generateDigits(c.perClass, stream(c.seed, 0));
      this.test = generateDigits(TEST_PER_CLASS, stream(c.seed, 1));
    }
    this.network = new AutoencoderNet(c, PIXELS, stream(c.seed, 2));
    this.optimizer = new Optimizer(c.optimizer);
    this.shuffle = stream(c.seed, 3);
    this.noise = stream(c.seed, 4);
    this.epoch = 0;
    this.cursor = 0;
    this.history = [];
    this.keys = { data: dataKey(c), net: netKey(c) };
  }

  private batch(set: DigitSet, idx: ArrayLike<number>): Matrix {
    const x = new Matrix(idx.length, PIXELS);
    for (let n = 0; n < idx.length; n++) {
      const src = idx[n] as number;
      x.data.set(set.images.subarray(src * PIXELS, (src + 1) * PIXELS), n * PIXELS);
    }
    return x;
  }

  private score(set: DigitSet): { recon: number; kl: number; total: number } {
    let recon = 0;
    let kl = 0;
    const chunk = 100;
    for (let start = 0; start < set.count; start += chunk) {
      const n = Math.min(chunk, set.count - start);
      const x = new Matrix(n, PIXELS, set.images.slice(start * PIXELS, (start + n) * PIXELS));
      const loss = this.network.loss(x, this.network.forward(x, null, false), this.cfg.beta);
      recon += loss.recon * n;
      kl += loss.kl * n;
    }
    recon /= set.count;
    kl /= set.count;
    return { recon, kl, total: recon + this.cfg.beta * kl };
  }
}
