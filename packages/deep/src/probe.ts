import { createRng } from '@sim/core';

/**
 * Linear probes: to ask what a layer of a network "knows", train the simplest possible classifier (a softmax
 * regression) to read a property off the layer's activations. High accuracy on held-out data means the property
 * can be read linearly. A control task with random labels measures how much the probe could memorise on its own,
 * so that a high score is not mistaken for knowledge.
 */
export interface ProbeOptions {
  iterations: number;
  learningRate: number;
  /** L2 penalty on the weights. */
  decay: number;
  seed: number;
}

export const DEFAULT_PROBE: ProbeOptions = {
  iterations: 300,
  learningRate: 0.3,
  decay: 1e-3,
  seed: 1,
};

export interface ProbeResult {
  trainAccuracy: number;
  testAccuracy: number;
}

/** Features are standardised with the statistics of the training rows. */
export function standardise(
  train: ArrayLike<number>,
  test: ArrayLike<number>,
  dim: number,
): { train: Float64Array; test: Float64Array } {
  const n = train.length / dim;
  const mean = new Float64Array(dim);
  const sd = new Float64Array(dim);
  for (let i = 0; i < n; i++)
    for (let j = 0; j < dim; j++)
      mean[j] = (mean[j] as number) + (train[i * dim + j] as number) / n;
  for (let i = 0; i < n; i++)
    for (let j = 0; j < dim; j++)
      sd[j] = (sd[j] as number) + ((train[i * dim + j] as number) - (mean[j] as number)) ** 2 / n;
  for (let j = 0; j < dim; j++) sd[j] = Math.sqrt(sd[j] as number) + 1e-6;
  const apply = (x: ArrayLike<number>) =>
    Float64Array.from(x, (v, k) => (v - (mean[k % dim] as number)) / (sd[k % dim] as number));
  return { train: apply(train), test: apply(test) };
}

export function trainProbe(
  trainX: ArrayLike<number>,
  trainY: ArrayLike<number>,
  testX: ArrayLike<number>,
  testY: ArrayLike<number>,
  dim: number,
  classes: number,
  options: ProbeOptions = DEFAULT_PROBE,
): ProbeResult {
  const { train, test } = standardise(trainX, testX, dim);
  const n = trainY.length;
  const rng = createRng(options.seed);
  const w = Float64Array.from({ length: dim * classes }, () => rng.normal() * 0.01);
  const b = new Float64Array(classes);
  const logits = (x: Float64Array, i: number) => {
    const out = new Float64Array(classes);
    for (let c = 0; c < classes; c++) {
      let s = b[c] as number;
      for (let j = 0; j < dim; j++)
        s += (x[i * dim + j] as number) * (w[j * classes + c] as number);
      out[c] = s;
    }
    return out;
  };
  const softmax = (z: Float64Array) => {
    const top = Math.max(...z);
    const e = z.map((v) => Math.exp(v - top));
    const sum = e.reduce((a, v) => a + v, 0);
    return e.map((v) => v / sum);
  };
  // full-batch gradient descent with momentum
  const vw = new Float64Array(w.length);
  const vb = new Float64Array(classes);
  for (let it = 0; it < options.iterations; it++) {
    const gw = new Float64Array(w.length);
    const gb = new Float64Array(classes);
    for (let i = 0; i < n; i++) {
      const p = softmax(logits(train, i));
      for (let c = 0; c < classes; c++) {
        const d = (p[c] as number) - ((trainY[i] as number) === c ? 1 : 0);
        gb[c] = (gb[c] as number) + d / n;
        for (let j = 0; j < dim; j++)
          gw[j * classes + c] =
            (gw[j * classes + c] as number) + (d * (train[i * dim + j] as number)) / n;
      }
    }
    for (let k = 0; k < w.length; k++) {
      vw[k] =
        0.9 * (vw[k] as number) -
        options.learningRate * ((gw[k] as number) + options.decay * (w[k] as number));
      w[k] = (w[k] as number) + (vw[k] as number);
    }
    for (let c = 0; c < classes; c++) {
      vb[c] = 0.9 * (vb[c] as number) - options.learningRate * (gb[c] as number);
      b[c] = (b[c] as number) + (vb[c] as number);
    }
  }
  const accuracy = (x: Float64Array, y: ArrayLike<number>) => {
    let right = 0;
    for (let i = 0; i < y.length; i++) {
      const z = logits(x, i);
      if (z.indexOf(Math.max(...z)) === y[i]) right++;
    }
    return right / y.length;
  };
  return { trainAccuracy: accuracy(train, trainY), testAccuracy: accuracy(test, testY) };
}
