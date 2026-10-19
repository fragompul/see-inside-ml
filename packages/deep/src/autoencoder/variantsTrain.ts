import { Matrix, Optimizer, createRng, permutation } from '@sim/core';
import { IMAGE_SIZE, generateDigits, type DigitSet } from '../cnn/digits';
import { ConditionalVae, VqVae } from './variants';

const PIXELS = IMAGE_SIZE * IMAGE_SIZE;

const batchOf = (set: DigitSet, idx: ArrayLike<number>): Matrix => {
  const x = new Matrix(idx.length, PIXELS);
  for (let n = 0; n < idx.length; n++) {
    const src = idx[n] as number;
    x.data.set(set.images.subarray(src * PIXELS, (src + 1) * PIXELS), n * PIXELS);
  }
  return x;
};

export interface VariantTraining {
  epochs: number;
  perClass: number;
  batchSize: number;
  learningRate: number;
  seed: number;
}

export const DEFAULT_VARIANT_TRAINING: VariantTraining = {
  epochs: 25,
  perClass: 60,
  batchSize: 32,
  learningRate: 0.003,
  seed: 3,
};

export interface ConditionalRun {
  model: ConditionalVae;
  /** Reconstruction and KL per image on the held-out digits after each epoch. */
  testRecon: number[];
  testKl: number[];
  test: DigitSet;
}

export function trainConditionalVae(
  o: VariantTraining & { latent: number; hidden: number; beta: number; conditioned: boolean },
): ConditionalRun {
  const train = generateDigits(o.perClass, createRng(o.seed * 10 + 1));
  const test = generateDigits(30, createRng(o.seed * 10 + 2));
  // an unconditioned model is the same network told the same (dummy) class for every image
  const label = (set: DigitSet, i: number) => (o.conditioned ? (set.labels[i] as number) : 0);
  const model = new ConditionalVae(PIXELS, 10, o.latent, o.hidden, createRng(o.seed * 10 + 3));
  const optimizer = new Optimizer('adam');
  const shuffle = createRng(o.seed * 10 + 4);
  const noise = createRng(o.seed * 10 + 5);
  const testRecon: number[] = [];
  const testKl: number[] = [];
  for (let epoch = 0; epoch < o.epochs; epoch++) {
    const order = permutation(train.count, shuffle);
    for (let start = 0; start < order.length; start += o.batchSize) {
      const idx = order.subarray(start, Math.min(order.length, start + o.batchSize));
      const x = batchOf(train, idx);
      const eps = new Matrix(
        idx.length,
        o.latent,
        Float64Array.from({ length: idx.length * o.latent }, () => noise.normal()),
      );
      const labels = Array.from(idx, (i) => label(train, i));
      const out = model.forward(x, labels, eps);
      model.backward(x, out, o.beta);
      optimizer.step(model.parameters, o.learningRate);
    }
    const x = batchOf(
      test,
      Array.from({ length: test.count }, (_, i) => i),
    );
    const out = model.forward(
      x,
      Array.from({ length: test.count }, (_, i) => label(test, i)),
      null,
      false,
    );
    const loss = model.loss(x, out, o.beta);
    testRecon.push(loss.recon);
    testKl.push(loss.kl);
  }
  return { model, testRecon, testKl, test };
}

export interface VqRun {
  model: VqVae;
  testRecon: number[];
  /** How many images of the training set snap to each codebook entry at the end. */
  usage: number[];
  /** Digit class most common among the images of each entry, and how pure that entry is (0..1). */
  majority: { digit: number; purity: number }[];
  train: DigitSet;
  test: DigitSet;
}

export function trainVqVae(
  o: VariantTraining & {
    entries: number;
    dim: number;
    hidden: number;
    commitment: number;
    /** Move entries that no image used during an epoch onto the codes of random images (for most of the training). */
    revive: boolean;
  },
): VqRun {
  const train = generateDigits(o.perClass, createRng(o.seed * 10 + 1));
  const test = generateDigits(30, createRng(o.seed * 10 + 2));
  const model = new VqVae(PIXELS, o.entries, o.dim, o.hidden, createRng(o.seed * 10 + 3));
  const optimizer = new Optimizer('adam');
  const shuffle = createRng(o.seed * 10 + 4);
  const revival = createRng(o.seed * 10 + 6);
  const all = (set: DigitSet) =>
    batchOf(
      set,
      Array.from({ length: set.count }, (_, i) => i),
    );
  const testX = all(test);
  const testRecon: number[] = [];
  for (let epoch = 0; epoch < o.epochs; epoch++) {
    // the codebook starts at the codes of real images; it is seeded again after the first epoch, once the encoder has settled
    if (epoch === 0 || epoch === 1) model.initialiseCodebook(all(train), createRng(o.seed + epoch));
    const order = permutation(train.count, shuffle);
    const used = new Set<number>();
    for (let start = 0; start < order.length; start += o.batchSize) {
      const idx = order.subarray(start, Math.min(order.length, start + o.batchSize));
      const x = batchOf(train, idx);
      const out = model.forward(x);
      out.index.forEach((k) => used.add(k));
      model.backward(x, out, o.commitment);
      optimizer.step(model.parameters, o.learningRate);
    }
    if (o.revive && epoch < Math.floor(o.epochs * 0.8)) {
      const z = model.forward(all(train), false).z;
      for (let k = 0; k < o.entries; k++) {
        if (used.has(k)) continue;
        const pick = Math.floor(revival.next() * z.rows);
        model.codebook.data.set(z.data.subarray(pick * o.dim, (pick + 1) * o.dim), k * o.dim);
      }
    }
    testRecon.push(model.loss(testX, model.forward(testX, false), o.commitment).recon);
  }
  const index = model.forward(all(train), false).index;
  const usage = new Array<number>(o.entries).fill(0);
  const byDigit = Array.from({ length: o.entries }, () => new Array<number>(10).fill(0));
  index.forEach((k, i) => {
    usage[k] = (usage[k] as number) + 1;
    const row = byDigit[k] as number[];
    row[train.labels[i] as number] = (row[train.labels[i] as number] as number) + 1;
  });
  const majority = byDigit.map((row, k) => {
    const top = Math.max(...row);
    return { digit: row.indexOf(top), purity: usage[k] ? top / (usage[k] as number) : 0 };
  });
  return { model, testRecon, usage, majority, train, test };
}
