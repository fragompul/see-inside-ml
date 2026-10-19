import { Matrix, createRng } from '@sim/core';
import {
  DEFAULT_TOOLBOX,
  DEFAULT_VARIANT_TRAINING,
  generateDigits,
  trainConditionalVae,
  trainToolbox,
  trainVqVae,
  type ToolboxMLP,
} from '@sim/deep';

export type ControlMode = 'conditional' | 'vq';

export interface ControlScenario {
  mode: ControlMode;
  /** Conditional VAE: size of the code, weight of the KL term, and whether the class is given to the model at all. */
  latent: number;
  beta: number;
  conditioned: boolean;
  /** VQ-VAE: codebook size and weight of the commitment term. */
  entries: number;
  commitment: number;
  /** Move unused codebook entries onto real codes during training. */
  revive: boolean;
  epochs: number;
}

export const DEFAULT_CONTROL: ControlScenario = {
  mode: 'conditional',
  latent: 2,
  beta: 1,
  conditioned: true,
  entries: 10,
  commitment: 0.25,
  revive: true,
  epochs: 25,
};

let judge: ToolboxMLP | null = null;
/** A digit classifier, used as a judge of what the generators produce. */
export function digitJudge(): ToolboxMLP {
  if (!judge) {
    const set = generateDigits(80, createRng(901));
    const val = generateDigits(20, createRng(902));
    const wrap = (s: typeof set) => ({
      x: new Matrix(s.count, 256, Float64Array.from(s.images)),
      y: Array.from(s.labels),
    });
    judge = trainToolbox(
      { ...DEFAULT_TOOLBOX, layers: [256, 48, 10], seed: 1 },
      wrap(set),
      wrap(val),
      { epochs: 30, batchSize: 16, learningRate: 0.005, optimizer: 'adam', patience: 0, seed: 1 },
    ).model;
  }
  return judge;
}

const classify = (images: Float64Array[]): number[] => {
  const x = new Matrix(images.length, 256);
  images.forEach((img, i) => x.data.set(img, i * 256));
  return Array.from(digitJudge().predict(x));
};

export interface ConditionalResult {
  kind: 'conditional';
  scenario: ControlScenario;
  /** Fixed codes used for the grid (each row of the grid decodes the same codes with a different class). */
  codes: number[][];
  grid: Float64Array[][];
  /** Share of images generated "as digit c" that the judge reads as c, over random codes and all classes. */
  control: number;
  testRecon: number;
  testKl: number;
}

export interface VqResult {
  kind: 'vq';
  scenario: ControlScenario;
  /** Training images assigned to each entry, and the digit that most of them show with the share it has. */
  usage: number[];
  majority: { digit: number; purity: number }[];
  prototypes: Float64Array[];
  testRecon: number[];
  deadEntries: number;
  /** Average share of an entry's images that belong to its most common digit. */
  purity: number;
  /** The judge reads the prototype of an entry as the entry's most common digit. */
  prototypeAgreement: number;
}

export type ControlResult = ConditionalResult | VqResult;

const cache = new Map<string, ControlResult>();

export function runControl(s: ControlScenario): ControlResult {
  const key = JSON.stringify(
    s.mode === 'vq'
      ? { ...s, latent: 0, beta: 0, conditioned: false }
      : { ...s, entries: 0, commitment: 0, revive: false },
  );
  const hit = cache.get(key);
  if (hit) return hit;
  const base = { ...DEFAULT_VARIANT_TRAINING, epochs: s.epochs };
  let out: ControlResult;
  if (s.mode === 'conditional') {
    const run = trainConditionalVae({
      ...base,
      latent: s.latent,
      hidden: 64,
      beta: s.beta,
      conditioned: s.conditioned,
    });
    const rng = createRng(31);
    const codes = Array.from({ length: 8 }, () =>
      Array.from({ length: s.latent }, () => rng.normal()),
    );
    const grid = Array.from({ length: 10 }, (_, digit) =>
      codes.map((z) => run.model.decode(z, s.conditioned ? digit : 0)),
    );
    let right = 0;
    let total = 0;
    for (let digit = 0; digit < 10; digit++) {
      const images = Array.from({ length: 30 }, () =>
        run.model.decode(
          Array.from({ length: s.latent }, () => rng.normal()),
          s.conditioned ? digit : 0,
        ),
      );
      for (const guess of classify(images)) {
        right += guess === digit ? 1 : 0;
        total += 1;
      }
    }
    out = {
      kind: 'conditional',
      scenario: s,
      codes,
      grid,
      control: right / total,
      testRecon: run.testRecon[run.testRecon.length - 1] as number,
      testKl: run.testKl[run.testKl.length - 1] as number,
    };
  } else {
    const run = trainVqVae({
      ...base,
      entries: s.entries,
      dim: 4,
      hidden: 64,
      commitment: s.commitment,
      revive: s.revive,
    });
    const prototypes = Array.from({ length: s.entries }, (_, k) => run.model.decodeEntry(k));
    const used = run.usage.filter((u) => u > 0);
    const guesses = classify(prototypes);
    let agree = 0;
    let weight = 0;
    run.usage.forEach((u, k) => {
      if (u === 0) return;
      agree += guesses[k] === (run.majority[k] as { digit: number }).digit ? u : 0;
      weight += u;
    });
    out = {
      kind: 'vq',
      scenario: s,
      usage: run.usage,
      majority: run.majority,
      prototypes,
      testRecon: run.testRecon,
      deadEntries: s.entries - used.length,
      purity:
        run.usage.reduce((a, u, k) => a + u * (run.majority[k] as { purity: number }).purity, 0) /
        run.train.count,
      prototypeAgreement: agree / weight,
    };
  }
  cache.set(key, out);
  return out;
}
