import { LEAKY_SLOPE, Matrix, Optimizer, createRng, type Parameter, type Rng } from '@sim/core';
import { DenseLayer } from '../dense';
import { MODE_CENTERS, MODE_STD } from './datasets';
import { fractionNearData, modesCovered, slicedWasserstein } from './metrics';
import { Generator } from './network';

/**
 * Three GAN recipes on the eight-Gaussians problem.
 *  - `gan`: the original two-player game with a discriminator that outputs a probability.
 *  - `wgan-gp`: the discriminator becomes a critic that outputs an unbounded score; it is trained to widen the
 *    gap between its average score on real and generated points, while a gradient penalty keeps its slope near 1
 *    (so the gap estimates the Wasserstein distance) instead of the saturating loss of the original game.
 *  - `cgan`: both players are also told which of the eight clusters a point belongs to, so the generator can be
 *    asked for a particular one.
 */
export type GanKind = 'gan' | 'wgan-gp' | 'cgan';

const CLASSES = MODE_CENTERS.length;

const concat = (a: Matrix, b: Matrix): Matrix => {
  const out = new Matrix(a.rows, a.cols + b.cols);
  for (let r = 0; r < a.rows; r++) {
    out.data.set(a.data.subarray(r * a.cols, (r + 1) * a.cols), r * out.cols);
    out.data.set(b.data.subarray(r * b.cols, (r + 1) * b.cols), r * out.cols + a.cols);
  }
  return out;
};

const tags = (labels: ArrayLike<number>): Matrix => {
  const m = new Matrix(labels.length, CLASSES);
  for (let i = 0; i < labels.length; i++) m.data[i * CLASSES + (labels[i] as number)] = 1;
  return m;
};

/** A discriminator (probability out) or critic (raw score out): input -> hidden -> hidden -> 1. */
export class Critic {
  readonly layers: DenseLayer[];

  constructor(
    readonly inputs: number,
    readonly hidden: number,
    readonly output: 'sigmoid' | 'linear',
    rng: Rng,
  ) {
    this.layers = [
      new DenseLayer(inputs, hidden, 'leaky_relu', rng),
      new DenseLayer(hidden, hidden, 'leaky_relu', rng),
      new DenseLayer(hidden, 1, output, rng),
    ];
  }

  get parameters(): Parameter[] {
    return this.layers.flatMap((l) => l.parameters);
  }

  forward(x: Matrix, cache = true): Matrix {
    let a = x;
    for (const layer of this.layers) a = layer.forward(a, cache);
    return a;
  }

  /** Backprop from dL/dZ of the output layer; returns dL/dInput per row (not averaged). */
  backwardFromLogits(dZ: Matrix): Matrix {
    let d = (this.layers[2] as DenseLayer).backwardFromPreActivation(dZ);
    d = (this.layers[1] as DenseLayer).backward(d);
    return (this.layers[0] as DenseLayer).backward(d);
  }

  /**
   * Gradient of the output with respect to the first `dims` inputs of each row, from the cached forward pass.
   * Because the hidden activations are piecewise linear it is exactly W1 (D1 (W2 (D2 w3))).
   */
  inputGradient(dims: number): { grads: Float64Array; u1: Matrix; u2: Matrix } {
    const [l1, l2, l3] = this.layers as [DenseLayer, DenseLayer, DenseLayer];
    const z1 = l1.z as Matrix;
    const z2 = l2.z as Matrix;
    const rows = z1.rows;
    const h = this.hidden;
    const u1 = new Matrix(rows, h);
    const u2 = new Matrix(rows, h);
    const grads = new Float64Array(rows * dims);
    for (let r = 0; r < rows; r++) {
      for (let k = 0; k < h; k++) {
        const d2 = (z2.data[r * h + k] as number) > 0 ? 1 : LEAKY_SLOPE;
        u2.data[r * h + k] = d2 * (l3.weights.data[k] as number);
      }
      for (let j = 0; j < h; j++) {
        let v = 0;
        for (let k = 0; k < h; k++)
          v += (l2.weights.data[j * h + k] as number) * (u2.data[r * h + k] as number);
        const d1 = (z1.data[r * h + j] as number) > 0 ? 1 : LEAKY_SLOPE;
        u1.data[r * h + j] = d1 * v;
      }
      for (let i = 0; i < dims; i++) {
        let g = 0;
        for (let j = 0; j < h; j++)
          g += (l1.weights.data[i * h + j] as number) * (u1.data[r * h + j] as number);
        grads[r * dims + i] = g;
      }
    }
    return { grads, u1, u2 };
  }

  /**
   * Adds the gradient of  lambda * mean((||grad_x D(x)|| - 1)^2)  at the points of the last cached forward pass to
   * the parameter gradients, and returns the penalty. The gradient of the input-gradient with respect to the weights
   * is again a product of the same matrices, so no automatic differentiation is needed.
   */
  addGradientPenalty(lambda: number, dims: number): number {
    const [l1, l2, l3] = this.layers as [DenseLayer, DenseLayer, DenseLayer];
    const { grads, u1, u2 } = this.inputGradient(dims);
    const z1 = l1.z as Matrix;
    const z2 = l2.z as Matrix;
    const rows = z1.rows;
    const h = this.hidden;
    const scale = lambda / rows;
    let penalty = 0;
    for (let r = 0; r < rows; r++) {
      let norm2 = 0;
      for (let i = 0; i < dims; i++) norm2 += (grads[r * dims + i] as number) ** 2;
      const norm = Math.sqrt(norm2) + 1e-12;
      penalty += (norm - 1) ** 2 / rows;
      const v = new Float64Array(dims);
      for (let i = 0; i < dims; i++)
        v[i] = (2 * (norm - 1) * (grads[r * dims + i] as number)) / norm;
      // W1: v (outer) u1
      for (let i = 0; i < dims; i++)
        for (let j = 0; j < h; j++)
          l1.weightParam.grad[i * h + j] =
            (l1.weightParam.grad[i * h + j] as number) +
            scale * (v[i] as number) * (u1.data[r * h + j] as number);
      // r1 = D1 (W1^T v);  W2 gets r1 (outer) u2
      const r1 = new Float64Array(h);
      for (let j = 0; j < h; j++) {
        let s = 0;
        for (let i = 0; i < dims; i++)
          s += (l1.weights.data[i * h + j] as number) * (v[i] as number);
        r1[j] = ((z1.data[r * h + j] as number) > 0 ? 1 : LEAKY_SLOPE) * s;
      }
      for (let j = 0; j < h; j++)
        for (let k = 0; k < h; k++)
          l2.weightParam.grad[j * h + k] =
            (l2.weightParam.grad[j * h + k] as number) +
            scale * (r1[j] as number) * (u2.data[r * h + k] as number);
      // w3 gets D2 (W2^T r1)
      for (let k = 0; k < h; k++) {
        let s = 0;
        for (let j = 0; j < h; j++) s += (l2.weights.data[j * h + k] as number) * (r1[j] as number);
        const d2 = (z2.data[r * h + k] as number) > 0 ? 1 : LEAKY_SLOPE;
        l3.weightParam.grad[k] = (l3.weightParam.grad[k] as number) + scale * d2 * s;
      }
    }
    return lambda * penalty;
  }
}

export interface GanVariantOptions {
  kind: GanKind;
  steps: number;
  hidden: number;
  noiseDim: number;
  batch: number;
  learningRate: number;
  /** Critic updates per generator update. */
  criticSteps: number;
  /** Weight of the gradient penalty (WGAN-GP). */
  penalty: number;
  seed: number;
}

export const DEFAULT_GAN_VARIANT: GanVariantOptions = {
  kind: 'gan',
  steps: 1500,
  hidden: 32,
  noiseDim: 4,
  batch: 64,
  learningRate: 0.001,
  criticSteps: 1,
  penalty: 10,
  seed: 1,
};

export interface GanVariantPoint {
  step: number;
  modes: number;
  distance: number;
  /** Mean critic score on real minus generated points (an estimate of the Wasserstein distance for the critic). */
  gap: number;
}

export interface GanVariantRun {
  history: GanVariantPoint[];
  /** Final generated points (n × 2) and, for the conditional GAN, the class asked for. */
  samples: Float64Array;
  labels: number[];
  modes: number;
  distance: number;
  nearData: number;
  /** Conditional GAN: share of points that land nearest the cluster that was asked for (NaN otherwise). */
  classAccuracy: number;
  /** Critic updates made in total: criticSteps times the number of generator updates. */
  criticUpdates: number;
}

function sampleLabelled(n: number, rng: Rng): { points: Float64Array; labels: number[] } {
  const points = new Float64Array(n * 2);
  const labels: number[] = [];
  for (let i = 0; i < n; i++) {
    const c = Math.floor(rng.next() * CLASSES);
    const centre = MODE_CENTERS[c] as readonly [number, number];
    points[2 * i] = centre[0] + rng.normal() * MODE_STD;
    points[2 * i + 1] = centre[1] + rng.normal() * MODE_STD;
    labels.push(c);
  }
  return { points, labels };
}

export function nearestMode(x: number, y: number): number {
  let best = 0;
  let bestD = Infinity;
  MODE_CENTERS.forEach((c, k) => {
    const d = (x - c[0]) ** 2 + (y - c[1]) ** 2;
    if (d < bestD) {
      bestD = d;
      best = k;
    }
  });
  return best;
}

const EPS = 1e-12;
const clampP = (p: number) => Math.min(1 - EPS, Math.max(EPS, p));

export function trainGanVariant(o: GanVariantOptions, every = 100): GanVariantRun {
  const rng = createRng(o.seed);
  const conditional = o.kind === 'cgan';
  const wasserstein = o.kind === 'wgan-gp';
  const extra = conditional ? CLASSES : 0;
  const generator = new Generator(
    { noiseDim: o.noiseDim + extra, hidden: o.hidden },
    createRng(o.seed + 1),
  );
  const critic = new Critic(
    2 + extra,
    o.hidden,
    wasserstein ? 'linear' : 'sigmoid',
    createRng(o.seed + 2),
  );
  const adam = wasserstein ? { beta1: 0, beta2: 0.9 } : { beta1: 0.5, beta2: 0.999 };
  const gOpt = new Optimizer('adam', adam);
  const cOpt = new Optimizer('adam', adam);
  const noise = (n: number) =>
    new Matrix(
      n,
      o.noiseDim,
      Float64Array.from({ length: n * o.noiseDim }, () => rng.normal()),
    );
  const generate = (z: Matrix, labels: number[]): Matrix =>
    generator.forward(conditional ? concat(z, tags(labels)) : z, false);
  const evalZ = noise(400);
  const evalLabels = Array.from({ length: 400 }, (_, i) => i % CLASSES);
  const reference = sampleLabelled(400, createRng(o.seed + 99));
  const history: GanVariantPoint[] = [];

  const measure = (step: number) => {
    const fake = generate(evalZ, evalLabels);
    const realIn = conditional
      ? concat(new Matrix(400, 2, reference.points), tags(reference.labels))
      : new Matrix(400, 2, reference.points);
    const fakeIn = conditional ? concat(fake, tags(evalLabels)) : fake;
    const score = (m: Matrix) => {
      const out = critic.forward(m, false);
      return out.data.reduce((a, b) => a + b, 0) / out.rows;
    };
    history.push({
      step,
      modes: modesCovered(fake.data).count,
      distance: slicedWasserstein(fake.data, reference.points),
      gap: score(realIn) - score(fakeIn),
    });
  };

  let criticUpdates = 0;
  for (let step = 1; step <= o.steps; step++) {
    for (let k = 0; k < o.criticSteps; k++) {
      criticUpdates += 1;
      const real = sampleLabelled(o.batch, rng);
      const z = noise(o.batch);
      const fake = generate(z, real.labels);
      const realIn = conditional
        ? concat(new Matrix(o.batch, 2, real.points), tags(real.labels))
        : new Matrix(o.batch, 2, real.points);
      const fakeIn = conditional ? concat(fake, tags(real.labels)) : fake;
      const both = new Matrix(2 * o.batch, realIn.cols);
      both.data.set(realIn.data, 0);
      both.data.set(fakeIn.data, realIn.data.length);
      const out = critic.forward(both, true);
      const dZ = new Matrix(2 * o.batch, 1);
      for (let i = 0; i < 2 * o.batch; i++) {
        const isReal = i < o.batch;
        // Wasserstein: minimise mean score on fakes minus mean score on reals (each half has weight 1/batch).
        // BCE: the usual (p - label) at the logit.
        dZ.data[i] = wasserstein ? (isReal ? -2 : 2) : (out.data[i] as number) - (isReal ? 1 : 0);
      }
      critic.backwardFromLogits(dZ);
      if (wasserstein) {
        // penalise the slope at random points between a real and a generated point
        const mix = new Matrix(o.batch, realIn.cols);
        for (let i = 0; i < o.batch; i++) {
          const t = rng.next();
          for (let c = 0; c < 2; c++)
            mix.data[i * mix.cols + c] =
              t * (realIn.data[i * realIn.cols + c] as number) +
              (1 - t) * (fakeIn.data[i * fakeIn.cols + c] as number);
        }
        critic.forward(mix, true);
        critic.addGradientPenalty(o.penalty, 2);
      }
      cOpt.step(critic.parameters, o.learningRate);
    }
    const labels = Array.from({ length: o.batch }, () => Math.floor(rng.next() * CLASSES));
    const z = noise(o.batch);
    const gIn = conditional ? concat(z, tags(labels)) : z;
    const fake = generator.forward(gIn, true);
    const cIn = conditional ? concat(fake, tags(labels)) : fake;
    const out = critic.forward(cIn, true);
    const dZ = new Matrix(o.batch, 1);
    for (let i = 0; i < o.batch; i++) {
      const p = out.data[i] as number;
      // Wasserstein: maximise the score. Otherwise the non-saturating loss -log D(G(z)).
      dZ.data[i] = wasserstein ? -1 : clampP(p) - 1;
    }
    const dIn = critic.backwardFromLogits(dZ);
    const dPoints = new Matrix(o.batch, 2);
    for (let i = 0; i < o.batch; i++) {
      dPoints.data[2 * i] = dIn.data[i * dIn.cols] as number;
      dPoints.data[2 * i + 1] = dIn.data[i * dIn.cols + 1] as number;
    }
    generator.backward(dPoints);
    gOpt.step(generator.parameters, o.learningRate);
    if (step % every === 0 || step === o.steps) measure(step);
  }

  const labels = evalLabels;
  const final = generate(evalZ, labels);
  let right = 0;
  for (let i = 0; i < 400; i++)
    if (nearestMode(final.data[2 * i] as number, final.data[2 * i + 1] as number) === labels[i])
      right++;
  return {
    history,
    samples: final.data,
    labels,
    modes: modesCovered(final.data).count,
    distance: slicedWasserstein(final.data, reference.points),
    nearData: fractionNearData(final.data, reference.points, 0.3),
    classAccuracy: conditional ? right / 400 : Number.NaN,
    criticUpdates,
  };
}
