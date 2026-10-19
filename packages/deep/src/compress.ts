import { Matrix } from '@sim/core';
import { ToolboxMLP, trainToolbox, type LabeledSet, type ToolboxConfig } from './toolbox';

/**
 * Making a trained network smaller: store its weights with fewer bits (quantisation), set the smallest
 * ones to zero (pruning), or train a small network to imitate a big one (distillation).
 */

/** Copy of a network (weights, biases, batch-norm parameters and statistics). */
export function cloneModel(model: ToolboxMLP): ToolboxMLP {
  const copy = new ToolboxMLP(model.config);
  model.tensors().forEach((t, i) => (copy.tensors()[i] as { data: Float64Array }).data.set(t.data));
  model.runMean.forEach((m, i) => (copy.runMean[i] as Float64Array).set(m));
  model.runVar.forEach((m, i) => (copy.runVar[i] as Float64Array).set(m));
  return copy;
}

export interface Quantized {
  values: Float64Array;
  /** Distance between neighbouring representable numbers. */
  step: number;
  levels: number;
  maxError: number;
  meanSquaredError: number;
}

/** Symmetric uniform quantisation to `bits` bits: round to a multiple of max|w| / (2^(bits-1) - 1). */
export function quantizeValues(values: ArrayLike<number>, bits: number): Quantized {
  const top = (1 << (bits - 1)) - 1;
  let max = 0;
  for (let i = 0; i < values.length; i++) max = Math.max(max, Math.abs(values[i] as number));
  const step = max > 0 ? max / top : 1;
  const out = new Float64Array(values.length);
  let maxError = 0;
  let sq = 0;
  const seen = new Set<number>();
  for (let i = 0; i < values.length; i++) {
    const q = Math.max(-top, Math.min(top, Math.round((values[i] as number) / step)));
    seen.add(q);
    out[i] = q * step;
    const e = Math.abs(out[i]! - (values[i] as number));
    maxError = Math.max(maxError, e);
    sq += e * e;
  }
  return {
    values: out,
    step,
    levels: seen.size,
    maxError,
    meanSquaredError: sq / Math.max(values.length, 1),
  };
}

/** A copy of the network whose weight matrices are quantised (biases and statistics stay in full precision). */
export function quantizeModel(model: ToolboxMLP, bits: number): ToolboxMLP {
  const copy = cloneModel(model);
  copy.weights.forEach((w) => w.data.set(quantizeValues(w.data, bits).values));
  return copy;
}

/** Approximate storage of the weights in bytes at the given number of bits. */
export const weightBytes = (model: ToolboxMLP, bits: number): number =>
  Math.ceil((model.weights.reduce((s, w) => s + w.data.length, 0) * bits) / 8);

export interface Pruned {
  model: ToolboxMLP;
  /** 1 where a weight survives, per weight matrix. */
  masks: Uint8Array[];
  sparsity: number;
}

/** Global magnitude pruning: zero the smallest `sparsity` share of all weights. */
export function pruneModel(model: ToolboxMLP, sparsity: number): Pruned {
  const copy = cloneModel(model);
  const all = copy.weights
    .flatMap((w) => Array.from(w.data, (v) => Math.abs(v)))
    .sort((a, b) => a - b);
  const cut = all.length
    ? (all[Math.min(all.length - 1, Math.floor(sparsity * all.length))] as number)
    : 0;
  let zeros = 0;
  let total = 0;
  const masks = copy.weights.map((w) => {
    const mask = new Uint8Array(w.data.length);
    for (let i = 0; i < w.data.length; i++) {
      const keep = sparsity <= 0 || Math.abs(w.data[i] as number) >= cut ? 1 : 0;
      mask[i] = keep;
      if (!keep) w.data[i] = 0;
      zeros += keep ? 0 : 1;
      total += 1;
    }
    return mask;
  });
  return { model: copy, masks, sparsity: zeros / Math.max(total, 1) };
}

/** Continue training a pruned network while keeping the pruned weights at zero. */
export function fineTunePruned(
  pruned: Pruned,
  train: LabeledSet,
  options: { epochs: number; learningRate: number; batchSize: number; seed: number },
): void {
  const model = pruned.model;
  let seed = options.seed;
  for (let e = 0; e < options.epochs; e++) {
    // one epoch at a time so that the mask can be re-applied after every optimiser step
    const n = train.x.rows;
    const order = Array.from({ length: n }, (_, i) => i);
    let s = seed++;
    for (let i = n - 1; i > 0; i--) {
      s = (s * 1664525 + 1013904223) % 4294967296;
      const j = s % (i + 1);
      [order[i], order[j]] = [order[j] as number, order[i] as number];
    }
    for (let b = 0; b < n; b += options.batchSize) {
      const idx = order.slice(b, b + options.batchSize);
      const xb = new Matrix(idx.length, train.x.cols);
      idx.forEach((r, i) =>
        xb.data.set(
          train.x.data.subarray(r * train.x.cols, (r + 1) * train.x.cols),
          i * train.x.cols,
        ),
      );
      model.zeroGrad();
      model.forward(xb, true);
      model.backward(idx.map((i) => train.y[i] as number));
      model.update('adam', options.learningRate);
      model.weights.forEach((w, l) => {
        const m = pruned.masks[l] as Uint8Array;
        for (let i = 0; i < m.length; i++) if (!m[i]) w.data[i] = 0;
      });
    }
  }
}

/** Softened class probabilities of a network: p_i proportional to exp(log p_i / T). */
export function softTargets(model: ToolboxMLP, x: Matrix, temperature: number): Matrix {
  const p = model.forward(x, false);
  const out = new Matrix(p.rows, p.cols);
  for (let i = 0; i < p.rows; i++) {
    let z = 0;
    for (let k = 0; k < p.cols; k++) {
      const v = Math.exp(Math.log(Math.max(p.get(i, k), 1e-12)) / temperature);
      out.set(i, k, v);
      z += v;
    }
    for (let k = 0; k < p.cols; k++) out.set(i, k, out.get(i, k) / z);
  }
  return out;
}

/** Trains a student on the teacher's softened outputs (and, optionally, a share of the true labels). */
export function distill(
  teacher: ToolboxMLP,
  studentConfig: ToolboxConfig,
  train: LabeledSet,
  options: {
    epochs: number;
    learningRate: number;
    batchSize: number;
    temperature: number;
    seed: number;
  },
): ToolboxMLP {
  const student = new ToolboxMLP(studentConfig);
  const targets = softTargets(teacher, train.x, options.temperature);
  const n = train.x.rows;
  let s = options.seed;
  for (let e = 0; e < options.epochs; e++) {
    const order = Array.from({ length: n }, (_, i) => i);
    for (let i = n - 1; i > 0; i--) {
      s = (s * 1664525 + 1013904223) % 4294967296;
      const j = s % (i + 1);
      [order[i], order[j]] = [order[j] as number, order[i] as number];
    }
    for (let b = 0; b < n; b += options.batchSize) {
      const idx = order.slice(b, b + options.batchSize);
      const xb = new Matrix(idx.length, train.x.cols);
      const tb = new Matrix(idx.length, targets.cols);
      idx.forEach((r, i) => {
        xb.data.set(
          train.x.data.subarray(r * train.x.cols, (r + 1) * train.x.cols),
          i * train.x.cols,
        );
        tb.data.set(
          targets.data.subarray(r * targets.cols, (r + 1) * targets.cols),
          i * targets.cols,
        );
      });
      student.zeroGrad();
      student.forward(xb, true);
      student.backwardSoft(tb);
      student.update('adam', options.learningRate);
    }
  }
  return student;
}

export { trainToolbox };
