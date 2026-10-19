import {
  cloneModel,
  distill,
  fineTunePruned,
  generateDigits,
  pruneModel,
  quantizeModel,
  quantizeValues,
  trainToolbox,
  weightBytes,
  type ToolboxConfig,
  ToolboxMLP,
} from '@sim/deep';
import { Matrix, createRng } from '@sim/core';

export type CompressionMode = 'quantize' | 'prune' | 'distill';

export interface CompressionScenario {
  mode: CompressionMode;
  /** Bits per weight (quantisation). */
  bits: number;
  /** Share of weights set to zero (pruning). */
  sparsity: number;
  fineTune: boolean;
  /** Labelled digits the small student may use on its own (distillation). */
  labelled: number;
  temperature: number;
  seed: number;
}

export const DEFAULT_COMPRESSION: CompressionScenario = {
  mode: 'quantize',
  bits: 4,
  sparsity: 0.8,
  fineTune: false,
  labelled: 60,
  temperature: 3,
  seed: 2,
};

const digits = (perClass: number, seed: number) => {
  const set = generateDigits(perClass, createRng(seed));
  return {
    x: new Matrix(set.count, 256, Float64Array.from(set.images)),
    y: Array.from(set.labels),
  };
};

export const TEACHER: ToolboxConfig = {
  layers: [256, 64, 32, 10],
  activation: 'relu',
  init: 'he',
  dropout: 0,
  batchNorm: false,
  weightDecay: 0,
  seed: 5,
};
export const STUDENT: ToolboxConfig = { ...TEACHER, layers: [256, 10, 10], seed: 9 };

let cached: {
  teacher: ToolboxMLP;
  train: ReturnType<typeof digits>;
  test: ReturnType<typeof digits>;
} | null = null;

export function base() {
  if (cached) return cached;
  const train = digits(60, 11);
  const test = digits(30, 12);
  const { model } = trainToolbox(TEACHER, train, test, {
    epochs: 25,
    batchSize: 32,
    learningRate: 0.003,
    optimizer: 'adam',
    patience: 0,
    seed: 1,
  });
  cached = { teacher: model, train, test };
  return cached;
}

export interface CompressionRun {
  scenario: CompressionScenario;
  teacherAccuracy: number;
  parameters: number;
  /** Accuracy of the compressed (or distilled) network on unseen digits. */
  accuracy: number;
  /** Accuracy of the baseline it should be compared with (a student trained on labels alone, in distillation). */
  baseline: number;
  bytes: number;
  fullBytes: number;
  detail: string;
}

export function runCompression(s: CompressionScenario): CompressionRun {
  const { teacher, train, test } = base();
  const teacherAccuracy = teacher.evaluate(test.x, test.y).accuracy;
  const weights = teacher.weights.reduce((a, w) => a + w.data.length, 0);
  const fullBytes = weightBytes(teacher, 32);
  if (s.mode === 'quantize') {
    const q = s.bits >= 32 ? cloneModel(teacher) : quantizeModel(teacher, s.bits);
    const mse =
      s.bits >= 32
        ? 0
        : teacher.weights.reduce(
            (a, w) => a + quantizeValues(w.data, s.bits).meanSquaredError * w.data.length,
            0,
          ) / weights;
    return {
      scenario: s,
      teacherAccuracy,
      parameters: weights,
      accuracy: q.evaluate(test.x, test.y).accuracy,
      baseline: teacherAccuracy,
      bytes: weightBytes(teacher, s.bits),
      fullBytes,
      detail: `mean squared rounding error per weight ${mse.toExponential(1)}`,
    };
  }
  if (s.mode === 'prune') {
    const p = pruneModel(teacher, s.sparsity);
    const before = p.model.evaluate(test.x, test.y).accuracy;
    if (s.fineTune)
      fineTunePruned(p, train, { epochs: 6, learningRate: 0.001, batchSize: 32, seed: s.seed });
    const acc = p.model.evaluate(test.x, test.y).accuracy;
    return {
      scenario: s,
      teacherAccuracy,
      parameters: weights,
      accuracy: acc,
      baseline: before,
      bytes: Math.round(weightBytes(teacher, 32) * (1 - p.sparsity)),
      fullBytes,
      detail: `${Math.round(p.sparsity * 100)}% of the weights are zero; before fine-tuning ${(before * 100).toFixed(0)}% correct`,
    };
  }
  // distillation: the student sees `labelled` digits with labels, or the teacher's softened view of the whole pool
  const per = Math.max(1, Math.round(s.labelled / 10));
  const small = digits(per, 31);
  const hard = trainToolbox(STUDENT, small, test, {
    epochs: 60,
    batchSize: 16,
    learningRate: 0.005,
    optimizer: 'adam',
    patience: 0,
    seed: 2,
  }).model;
  const soft = distill(teacher, STUDENT, train, {
    epochs: 40,
    learningRate: 0.005,
    batchSize: 32,
    temperature: s.temperature,
    seed: 3,
  });
  return {
    scenario: s,
    teacherAccuracy,
    parameters: weights,
    accuracy: soft.evaluate(test.x, test.y).accuracy,
    baseline: hard.evaluate(test.x, test.y).accuracy,
    bytes: weightBytes(soft, 32),
    fullBytes,
    detail: `student has ${soft.weights.reduce((a, w) => a + w.data.length, 0)} weights against ${weights}`,
  };
}

/**
 * Trains the teacher a few epochs at a time, giving the browser back to the page in between, so that the lab
 * stays responsive while it prepares. Equivalent to the synchronous training done by `base()`.
 */
export async function prepareTeacher(): Promise<void> {
  if (cached) return;
  const train = digits(60, 11);
  const test = digits(30, 12);
  const model = new ToolboxMLP(TEACHER);
  const rng = createRng(2);
  const n = train.x.rows;
  for (let epoch = 0; epoch < 25; epoch++) {
    const order = Array.from({ length: n }, (_, i) => i);
    for (let i = n - 1; i > 0; i--) {
      const j = Math.floor(rng.next() * (i + 1));
      [order[i], order[j]] = [order[j] as number, order[i] as number];
    }
    for (let b = 0; b < n; b += 32) {
      const idx = order.slice(b, b + 32);
      const xb = new Matrix(idx.length, train.x.cols);
      idx.forEach((r, i) =>
        xb.data.set(
          train.x.data.subarray(r * train.x.cols, (r + 1) * train.x.cols),
          i * train.x.cols,
        ),
      );
      model.zeroGrad();
      model.forward(xb, true, rng);
      model.backward(idx.map((i) => train.y[i] as number));
      model.update('adam', 0.003);
    }
    await new Promise((resolve) => setTimeout(resolve, 0));
  }
  cached = { teacher: model, train, test };
}
