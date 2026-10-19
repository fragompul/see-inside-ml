import type { ToolboxMLP } from '@sim/deep';
import { cloneModel, generateDigits } from '@sim/deep';
import { Matrix, createRng } from '@sim/core';
import { base } from '../compression/compressionCore';

/**
 * Adapting a pre-trained digit reader to a new situation (every digit shifted three pixels right and two down, as
 * if the camera had moved) from only a handful of labelled examples. Full fine-tuning updates every weight; head-only
 * updates the last layer; LoRA freezes everything and learns a low-rank correction to each layer.
 */
export type TuneMethod = 'none' | 'head' | 'full' | 'lora';

export interface LoraScenario {
  method: TuneMethod;
  rank: number;
  /** Labelled examples per digit class in the new situation. */
  perClass: number;
  epochs: number;
  learningRate: number;
  seed: number;
}

export const DEFAULT_LORA: LoraScenario = {
  method: 'lora',
  rank: 4,
  perClass: 5,
  epochs: 40,
  learningRate: 0.01,
  seed: 4,
};

export const METHOD_LABELS: Record<TuneMethod, string> = {
  none: 'No adaptation',
  head: 'Last layer only',
  full: 'Full fine-tuning',
  lora: 'LoRA',
};

function shift(images: ArrayLike<number>, count: number, dx: number, dy: number): Float64Array {
  const out = new Float64Array(count * 256);
  for (let i = 0; i < count; i++)
    for (let y = 0; y < 16; y++)
      for (let x = 0; x < 16; x++) {
        const sx = x - dx;
        const sy = y - dy;
        if (sx >= 0 && sx < 16 && sy >= 0 && sy < 16)
          out[i * 256 + y * 16 + x] = images[i * 256 + sy * 16 + sx] as number;
      }
  return out;
}

const newWorld = (perClass: number, seed: number) => {
  const set = generateDigits(perClass, createRng(seed));
  return {
    x: new Matrix(set.count, 256, shift(set.images, set.count, 3, 2)),
    y: Array.from(set.labels),
  };
};

export interface LoraRun {
  scenario: LoraScenario;
  /** Accuracy of the pre-trained network on the original and on the shifted digits, before any adaptation. */
  originalBefore: number;
  shiftedBefore: number;
  /** After adaptation. */
  shiftedAfter: number;
  originalAfter: number;
  trainable: number;
  total: number;
  trainLoss: number[];
}

export function runLora(s: LoraScenario): LoraRun {
  const { teacher, test } = base();
  const shiftedTest = {
    x: new Matrix(test.x.rows, 256, shift(test.x.data, test.x.rows, 3, 2)),
    y: test.y,
  };
  const small = newWorld(s.perClass, 77);
  const model: ToolboxMLP = cloneModel(teacher);
  const total =
    model.weights.reduce((a, w) => a + w.data.length, 0) +
    model.biases.reduce((a, b) => a + b.data.length, 0);
  const originalBefore = teacher.evaluate(test.x, test.y).accuracy;
  const shiftedBefore = teacher.evaluate(shiftedTest.x, shiftedTest.y).accuracy;
  const loss: number[] = [];
  if (s.method === 'head') {
    const last = model.weights.length - 1;
    model.weights.forEach((w, i) => (w.frozen = i !== last));
    model.biases.forEach((b, i) => (b.frozen = i !== last));
  } else if (s.method === 'lora') model.addAdapters(s.rank, s.seed);
  if (s.method !== 'none') {
    const rng = createRng(s.seed);
    const n = small.x.rows;
    for (let e = 0; e < s.epochs; e++) {
      const order = Array.from({ length: n }, (_, i) => i);
      for (let i = n - 1; i > 0; i--) {
        const j = Math.floor(rng.next() * (i + 1));
        [order[i], order[j]] = [order[j] as number, order[i] as number];
      }
      let sum = 0;
      for (let b = 0; b < n; b += 16) {
        const idx = order.slice(b, b + 16);
        const xb = new Matrix(idx.length, 256);
        idx.forEach((r, i) => xb.data.set(small.x.data.subarray(r * 256, (r + 1) * 256), i * 256));
        model.zeroGrad();
        model.forward(xb, true);
        sum += model.backward(idx.map((i) => small.y[i] as number)) * idx.length;
        model.update('adam', s.learningRate);
      }
      loss.push(sum / n);
    }
  }
  const trainable = s.method === 'none' ? 0 : model.trainableCount;
  if (s.method === 'lora') model.mergeAdapters();
  return {
    scenario: s,
    originalBefore,
    shiftedBefore,
    shiftedAfter:
      s.method === 'none' ? shiftedBefore : model.evaluate(shiftedTest.x, shiftedTest.y).accuracy,
    originalAfter: s.method === 'none' ? originalBefore : model.evaluate(test.x, test.y).accuracy,
    trainable,
    total,
    trainLoss: loss,
  };
}
