import { Matrix, Optimizer, createRng, type Parameter, type Rng } from '@sim/core';
import { DenseLayer } from '../dense';
import { TIME_FEATURES, makeSchedule, noiseTo, timeFeatures, type NoiseSchedule } from './schedule';
import { sampleDiffusion } from './sampler';

/**
 * Two refinements of the diffusion idea on two-dimensional data.
 *
 * Classifier-free guidance: a diffusion model that is told which class to draw, but is also trained now and then
 * with the class hidden, so that one network knows both the conditional and the unconditional noise. Sampling
 * pushes the prediction away from the unconditional one: e = e_c + w (e_c - e_u). A larger w follows the class more
 * closely and gives up variety.
 *
 * Flow matching: instead of predicting noise, learn a velocity field. Points on the straight line between a noise
 * sample and a data sample move at the constant velocity (data - noise); a network trained to predict that velocity
 * from the position and time turns noise into data by following the field, which takes few steps because the paths
 * are nearly straight.
 */

/** Six overlapping blobs on a circle, labelled by blob (so the classes genuinely blur into each other). */
export const BLOBS = 6;
export const BLOB_RADIUS = 1.6;
export const BLOB_STD = 0.42;
export const blobCentre = (k: number): [number, number] => [
  BLOB_RADIUS * Math.cos((2 * Math.PI * k) / BLOBS),
  BLOB_RADIUS * Math.sin((2 * Math.PI * k) / BLOBS),
];

export function sampleBlobs(n: number, rng: Rng): { points: Float64Array; labels: number[] } {
  const points = new Float64Array(n * 2);
  const labels: number[] = [];
  for (let i = 0; i < n; i++) {
    const k = Math.floor(rng.next() * BLOBS);
    const [cx, cy] = blobCentre(k);
    points[2 * i] = cx + rng.normal() * BLOB_STD;
    points[2 * i + 1] = cy + rng.normal() * BLOB_STD;
    labels.push(k);
  }
  return { points, labels };
}

export function nearestBlob(x: number, y: number): number {
  let best = 0;
  let bestD = Infinity;
  for (let k = 0; k < BLOBS; k++) {
    const [cx, cy] = blobCentre(k);
    const d = (x - cx) ** 2 + (y - cy) ** 2;
    if (d < bestD) {
      bestD = d;
      best = k;
    }
  }
  return best;
}

/** A network that reads a point, a time and optionally a class, and returns two numbers. */
export class FieldNet {
  readonly layers: DenseLayer[];
  readonly inputs: number;

  constructor(
    readonly classes: number,
    hidden: number,
    rng: Rng,
  ) {
    this.inputs = 2 + TIME_FEATURES + classes;
    this.layers = [
      new DenseLayer(this.inputs, hidden, 'swish', rng),
      new DenseLayer(hidden, hidden, 'swish', rng),
      new DenseLayer(hidden, hidden, 'swish', rng),
      new DenseLayer(hidden, 2, 'linear', rng),
    ];
  }

  get parameters(): Parameter[] {
    return this.layers.flatMap((l) => l.parameters);
  }

  /** `label` is a class number, or -1 for "class hidden" (all zeros). Time is `t` out of `scale`. */
  private input(
    x: ArrayLike<number>,
    t: ArrayLike<number>,
    scale: number,
    labels: ArrayLike<number>,
  ): Matrix {
    const n = x.length / 2;
    const m = new Matrix(n, this.inputs);
    for (let i = 0; i < n; i++) {
      m.data[i * this.inputs] = x[2 * i] as number;
      m.data[i * this.inputs + 1] = x[2 * i + 1] as number;
      m.data.set(timeFeatures(t[i] as number, scale), i * this.inputs + 2);
      const c = labels[i] as number;
      if (c >= 0) m.data[i * this.inputs + 2 + TIME_FEATURES + c] = 1;
    }
    return m;
  }

  run(
    x: ArrayLike<number>,
    t: ArrayLike<number>,
    scale: number,
    labels: ArrayLike<number>,
    cache: boolean,
  ) {
    let a = this.input(x, t, scale, labels);
    for (const layer of this.layers) a = layer.forward(a, cache);
    return a;
  }

  /** Mean squared error against `target` (n x 2) with gradients filled in. */
  gradients(
    x: ArrayLike<number>,
    t: ArrayLike<number>,
    scale: number,
    labels: ArrayLike<number>,
    target: ArrayLike<number>,
  ): number {
    const out = this.run(x, t, scale, labels, true);
    const dOut = new Matrix(out.rows, 2);
    let loss = 0;
    for (let i = 0; i < dOut.data.length; i++) {
      const diff = (out.data[i] as number) - (target[i] as number);
      dOut.data[i] = diff;
      loss += diff * diff;
    }
    let d = dOut;
    for (let i = this.layers.length - 1; i >= 0; i--)
      d = (this.layers[i] as DenseLayer).backward(d);
    return loss / dOut.data.length;
  }

  predict(x: ArrayLike<number>, t: number, scale: number, label: number): Float64Array {
    const n = x.length / 2;
    return Float64Array.from(
      this.run(x, new Float64Array(n).fill(t), scale, new Float64Array(n).fill(label), false).data,
    );
  }
}

export interface FieldTraining {
  steps: number;
  batch: number;
  hidden: number;
  learningRate: number;
  seed: number;
}

export const DEFAULT_FIELD_TRAINING: FieldTraining = {
  steps: 1500,
  batch: 128,
  hidden: 64,
  learningRate: 0.002,
  seed: 1,
};

export const DIFFUSION_T = 50;

export interface GuidedModel {
  net: FieldNet;
  schedule: NoiseSchedule;
}

/** Trains a conditional diffusion model, hiding the class with probability `dropout` for every example. */
export function trainGuidedDiffusion(
  options: FieldTraining & { dropout: number },
  data: { points: Float64Array; labels: number[] } = sampleBlobs(4000, createRng(options.seed + 5)),
): GuidedModel {
  const rng = createRng(options.seed);
  const net = new FieldNet(BLOBS, options.hidden, createRng(options.seed + 1));
  const schedule = makeSchedule('cosine', DIFFUSION_T);
  const opt = new Optimizer('adam');
  const n = data.labels.length;
  for (let step = 0; step < options.steps; step++) {
    const x0 = new Float64Array(options.batch * 2);
    const noisy = new Float64Array(options.batch * 2);
    const eps = new Float64Array(options.batch * 2);
    const t = new Float64Array(options.batch);
    const labels = new Float64Array(options.batch);
    for (let i = 0; i < options.batch; i++) {
      const pick = Math.floor(rng.next() * n);
      x0.set(data.points.subarray(2 * pick, 2 * pick + 2), 2 * i);
      eps[2 * i] = rng.normal();
      eps[2 * i + 1] = rng.normal();
      t[i] = 1 + Math.floor(rng.next() * DIFFUSION_T);
      labels[i] = rng.next() < options.dropout ? -1 : (data.labels[pick] as number);
      noisy.set(
        noiseTo(
          schedule,
          x0.subarray(2 * i, 2 * i + 2),
          eps.subarray(2 * i, 2 * i + 2),
          t[i] as number,
        ),
        2 * i,
      );
    }
    net.gradients(noisy, t, DIFFUSION_T, labels, eps);
    opt.step(net.parameters, options.learningRate);
  }
  return { net, schedule };
}

/** Draws `n` points of class `label` with guidance weight `w` (0 = the conditional model alone), by DDIM. */
export function sampleGuided(
  model: GuidedModel,
  label: number,
  w: number,
  n: number,
  seed: number,
  steps = 25,
): Float64Array {
  const rng = createRng(seed);
  const init = Float64Array.from({ length: 2 * n }, () => rng.normal());
  const predict = (x: Float64Array, t: number) => {
    const cond = model.net.predict(x, t, DIFFUSION_T, label);
    if (w === 0) return cond;
    const uncond = model.net.predict(x, t, DIFFUSION_T, -1);
    return cond.map((c, i) => c + w * (c - (uncond[i] as number)));
  };
  return sampleDiffusion(predict, model.schedule, init, { method: 'ddim', steps, rng }).samples;
}

export interface FlowMatchingModel {
  net: FieldNet;
}

const FM_SCALE = 100;

/** Trains a velocity field on the line from noise to data. */
export function trainFlowMatching(data: Float64Array, options: FieldTraining): FlowMatchingModel {
  const rng = createRng(options.seed);
  const net = new FieldNet(0, options.hidden, createRng(options.seed + 1));
  const opt = new Optimizer('adam');
  const n = data.length / 2;
  for (let step = 0; step < options.steps; step++) {
    const xt = new Float64Array(options.batch * 2);
    const target = new Float64Array(options.batch * 2);
    const t = new Float64Array(options.batch);
    const none = new Float64Array(options.batch).fill(-1);
    for (let i = 0; i < options.batch; i++) {
      const pick = Math.floor(rng.next() * n);
      const time = rng.next();
      t[i] = time * FM_SCALE;
      for (let c = 0; c < 2; c++) {
        const x1 = data[2 * pick + c] as number;
        const x0 = rng.normal();
        xt[2 * i + c] = (1 - time) * x0 + time * x1;
        target[2 * i + c] = x1 - x0;
      }
    }
    net.gradients(xt, t, FM_SCALE, none, target);
    opt.step(net.parameters, options.learningRate);
  }
  return { net };
}

/** Euler integration of a velocity field from noise (t = 0) to data (t = 1) in `steps` steps. */
export function integrateField(
  velocity: (x: Float64Array, t: number) => ArrayLike<number>,
  n: number,
  seed: number,
  steps: number,
): Float64Array {
  const rng = createRng(seed);
  const x = Float64Array.from({ length: 2 * n }, () => rng.normal());
  for (let k = 0; k < steps; k++) {
    const v = velocity(x, k / steps);
    for (let i = 0; i < x.length; i++) x[i] = (x[i] as number) + (v[i] as number) / steps;
  }
  return x;
}

/** Samples a trained flow-matching model. */
export function sampleFlowMatching(
  model: FlowMatchingModel,
  n: number,
  seed: number,
  steps: number,
): Float64Array {
  return integrateField((x, t) => model.net.predict(x, t * FM_SCALE, FM_SCALE, -1), n, seed, steps);
}

/** Samples of the unconditional diffusion model, for comparison: the guided model with the class hidden. */
export function sampleUnguided(
  model: GuidedModel,
  n: number,
  seed: number,
  steps: number,
): Float64Array {
  const rng = createRng(seed);
  const init = Float64Array.from({ length: 2 * n }, () => rng.normal());
  return sampleDiffusion((x, t) => model.net.predict(x, t, DIFFUSION_T, -1), model.schedule, init, {
    method: 'ddim',
    steps,
    rng,
  }).samples;
}
