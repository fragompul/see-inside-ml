import { Matrix, createRng } from '@sim/core';
import {
  DEFAULT_TOOLBOX,
  NO_AUGMENTATION,
  augmentSet,
  generateDigits,
  trainToolbox,
  type Augmentation,
} from '@sim/deep';

export type AugmentId = 'none' | 'shift' | 'rotate' | 'noise' | 'all';

export const AUGMENTATIONS: Record<AugmentId, { label: string; settings: Augmentation }> = {
  none: { label: 'None', settings: NO_AUGMENTATION },
  shift: { label: 'Shift up to 2 pixels', settings: { shift: 2, rotate: 0, noise: 0 } },
  rotate: { label: 'Rotate up to 15°', settings: { shift: 0, rotate: 15, noise: 0 } },
  noise: { label: 'Noise', settings: { shift: 0, rotate: 0, noise: 0.15 } },
  all: { label: 'All three', settings: { shift: 2, rotate: 15, noise: 0.15 } },
};

export interface AugmentScenario {
  /** Real labelled examples per digit. */
  perClass: number;
  augmentation: AugmentId;
  /** Random augmented copies made of each training image. */
  copies: number;
  epochs: number;
}

export const DEFAULT_AUGMENT: AugmentScenario = {
  perClass: 3,
  augmentation: 'shift',
  copies: 8,
  epochs: 30,
};

export const SEEDS = 3;
/** The kind of change a test picture may have suffered: a different camera position, tilt and a noisy sensor. */
export const WILD: Augmentation = { shift: 2, rotate: 15, noise: 0.15 };

export interface AugmentRun {
  scenario: AugmentScenario;
  /** Accuracy (mean over seeds) on new digits as drawn, and on new digits shifted, tilted and noisy. */
  clean: number;
  wild: number;
  /** Training accuracy on the real (unaugmented) examples. */
  train: number;
  /** Size of the training set after augmenting. */
  examples: number;
}

const cache = new Map<string, AugmentRun>();

export function runAugment(s: AugmentScenario): AugmentRun {
  const key = JSON.stringify(s);
  const hit = cache.get(key);
  if (hit) return hit;
  let clean = 0;
  let wild = 0;
  let train = 0;
  let examples = 0;
  for (let seed = 1; seed <= SEEDS; seed++) {
    const rng = createRng(1000 + seed);
    const real = generateDigits(s.perClass, rng);
    const settings = (AUGMENTATIONS[s.augmentation] as { settings: Augmentation }).settings;
    const copies = s.augmentation === 'none' ? 0 : s.copies;
    const bigger = augmentSet(real.images, real.count, settings, copies, rng);
    const labels = bigger.source.map((i) => real.labels[i] as number);
    const trainSet = {
      x: new Matrix(labels.length, 256, bigger.images),
      y: labels,
    };
    examples = labels.length;
    const test = generateDigits(30, createRng(5000 + seed));
    const wildTest = augmentSet(test.images, test.count, WILD, 1, createRng(6000 + seed));
    // keep only the augmented copies (the second of every pair) as the "in the wild" test set
    const wildX = new Float64Array(test.count * 256);
    for (let i = 0; i < test.count; i++)
      wildX.set(wildTest.images.subarray((2 * i + 1) * 256, (2 * i + 2) * 256), i * 256);
    const cleanSet = {
      x: new Matrix(test.count, 256, Float64Array.from(test.images)),
      y: Array.from(test.labels),
    };
    const wildSet = { x: new Matrix(test.count, 256, wildX), y: Array.from(test.labels) };
    const result = trainToolbox(
      { ...DEFAULT_TOOLBOX, layers: [256, 48, 10], seed },
      trainSet,
      cleanSet,
      {
        epochs: s.epochs,
        batchSize: 16,
        learningRate: 0.005,
        optimizer: 'adam',
        patience: 0,
        seed,
      },
    );
    clean += result.model.evaluate(cleanSet.x, cleanSet.y).accuracy / SEEDS;
    wild += result.model.evaluate(wildSet.x, wildSet.y).accuracy / SEEDS;
    const realSet = {
      x: new Matrix(real.count, 256, Float64Array.from(real.images)),
      y: Array.from(real.labels),
    };
    train += result.model.evaluate(realSet.x, realSet.y).accuracy / SEEDS;
  }
  const out = { scenario: s, clean, wild, train, examples };
  cache.set(key, out);
  return out;
}
