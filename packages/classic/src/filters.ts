import {
  Matrix,
  cholesky,
  createRng,
  identity,
  inverse,
  logDetFromCholesky,
  matMul,
  transpose,
  type Rng,
} from '@sim/core';

/**
 * Bayesian filtering: track a hidden state that moves over time from noisy measurements.
 * The Kalman filter is exact when the motion and the measurements are linear with Gaussian noise;
 * the particle filter approximates the same posterior with weighted samples and keeps working when
 * they are not (curved geometry, heavy-tailed noise).
 */

const add = (a: Matrix, b: Matrix): Matrix => {
  const out = new Matrix(a.rows, a.cols);
  for (let i = 0; i < a.data.length; i++)
    out.data[i] = (a.data[i] as number) + (b.data[i] as number);
  return out;
};
const subtract = (a: Matrix, b: Matrix): Matrix => {
  const out = new Matrix(a.rows, a.cols);
  for (let i = 0; i < a.data.length; i++)
    out.data[i] = (a.data[i] as number) - (b.data[i] as number);
  return out;
};
const column = (v: readonly number[]): Matrix => new Matrix(v.length, 1, Float64Array.from(v));
const toArray = (m: Matrix): number[] => Array.from(m.data);

export interface LinearGaussianModel {
  /** State transition `n × n`. */
  F: Matrix;
  /** Observation `m × n`. */
  H: Matrix;
  /** Process noise covariance `n × n`. */
  Q: Matrix;
  /** Measurement noise covariance `m × m`. */
  R: Matrix;
}

export interface KalmanStep {
  /** Prediction before seeing the measurement of this step. */
  predictedMean: number[];
  predictedCov: Matrix;
  /** Estimate after the measurement (equal to the prediction when the measurement is missing). */
  mean: number[];
  cov: Matrix;
  /** Kalman gain `n × m` (null when the measurement was missing). */
  gain: Matrix | null;
  innovation: number[] | null;
  /** Log-density of the measurement under the prediction. */
  logLikelihood: number;
}

export function kalmanFilter(
  model: LinearGaussianModel,
  measurements: readonly (readonly number[] | null)[],
  mean0: readonly number[],
  cov0: Matrix,
): KalmanStep[] {
  const { F, H, Q, R } = model;
  const n = F.rows;
  const steps: KalmanStep[] = [];
  let mean = column(mean0);
  let cov = cov0;
  measurements.forEach((z, t) => {
    // predict (the first step starts from the prior as given)
    const pm = t === 0 ? mean : matMul(F, mean);
    const pc = t === 0 ? cov : add(matMul(matMul(F, cov), transpose(F)), Q);
    if (z === null) {
      mean = pm;
      cov = pc;
      steps.push({
        predictedMean: toArray(pm),
        predictedCov: pc,
        mean: toArray(pm),
        cov: pc,
        gain: null,
        innovation: null,
        logLikelihood: 0,
      });
      return;
    }
    const innovation = subtract(column(z), matMul(H, pm));
    const S = add(matMul(matMul(H, pc), transpose(H)), R);
    const Sinv = inverse(S);
    const K = matMul(matMul(pc, transpose(H)), Sinv);
    mean = add(pm, matMul(K, innovation));
    // Joseph form keeps the covariance symmetric and positive semi-definite
    const IKH = subtract(identity(n), matMul(K, H));
    cov = add(matMul(matMul(IKH, pc), transpose(IKH)), matMul(matMul(K, R), transpose(K)));
    let quad = 0;
    for (let i = 0; i < innovation.rows; i++)
      for (let j = 0; j < innovation.rows; j++)
        quad += (innovation.data[i] as number) * Sinv.get(i, j) * (innovation.data[j] as number);
    const logDet = logDetFromCholesky(cholesky(S));
    steps.push({
      predictedMean: toArray(pm),
      predictedCov: pc,
      mean: toArray(mean),
      cov,
      gain: K,
      innovation: toArray(innovation),
      logLikelihood: -0.5 * (quad + logDet + innovation.rows * Math.log(2 * Math.PI)),
    });
  });
  return steps;
}

/** Rauch-Tung-Striebel smoother: re-estimate every step using the measurements that came after it. */
export function rtsSmoother(
  model: LinearGaussianModel,
  steps: readonly KalmanStep[],
): { mean: number[]; cov: Matrix }[] {
  const { F } = model;
  const n = steps.length;
  const out: { mean: number[]; cov: Matrix }[] = steps.map((s) => ({ mean: s.mean, cov: s.cov }));
  for (let t = n - 2; t >= 0; t--) {
    const filt = steps[t] as KalmanStep;
    const next = steps[t + 1] as KalmanStep;
    const G = matMul(matMul(filt.cov, transpose(F)), inverse(next.predictedCov));
    const dm = subtract(
      column((out[t + 1] as { mean: number[] }).mean),
      column(next.predictedMean),
    );
    const mean = add(column(filt.mean), matMul(G, dm));
    const dc = subtract((out[t + 1] as { cov: Matrix }).cov, next.predictedCov);
    const cov = add(filt.cov, matMul(matMul(G, dc), transpose(G)));
    out[t] = { mean: toArray(mean), cov };
  }
  return out;
}

/** The Kalman gain converges to a constant when the model is time-invariant: iterate until it does. */
export function steadyStateGain(model: LinearGaussianModel, iterations = 500): Matrix {
  const n = model.F.rows;
  let cov = identity(n);
  let K = new Matrix(n, model.H.rows);
  for (let i = 0; i < iterations; i++) {
    const pc = add(matMul(matMul(model.F, cov), transpose(model.F)), model.Q);
    const S = add(matMul(matMul(model.H, pc), transpose(model.H)), model.R);
    K = matMul(matMul(pc, transpose(model.H)), inverse(S));
    cov = matMul(subtract(identity(n), matMul(K, model.H)), pc);
  }
  return K;
}

// ---- particle filter -------------------------------------------------------------------------

/** Systematic resampling: indices of the particles to keep, with probability proportional to weight. */
export function systematicResample(weights: readonly number[], u: number): number[] {
  const n = weights.length;
  const out: number[] = [];
  let cumulative = weights[0] as number;
  let i = 0;
  for (let k = 0; k < n; k++) {
    const target = (k + u) / n;
    while (target > cumulative && i < n - 1) {
      i++;
      cumulative += weights[i] as number;
    }
    out.push(i);
  }
  return out;
}

export const effectiveSampleSize = (weights: readonly number[]): number =>
  1 / weights.reduce((s, w) => s + w * w, 0);

export interface ParticleModel<M> {
  /** Draw a state from the prior. */
  initial: (rng: Rng) => number[];
  /** Move a state one step forward (noise included). */
  propagate: (state: readonly number[], rng: Rng) => number[];
  /** Log-likelihood of a measurement given a state. */
  logLikelihood: (state: readonly number[], measurement: M) => number;
}

export interface ParticleStep {
  /** Particles after the update (and the resampling, if it happened). */
  particles: number[][];
  weights: number[];
  mean: number[];
  /** Effective sample size before resampling: how many particles still "count". */
  ess: number;
  resampled: boolean;
}

export interface ParticleOptions {
  particles: number;
  /** Resample when the effective sample size falls below this share of the particles. */
  resampleBelow?: number;
  seed?: number;
}

export function particleFilter<M>(
  model: ParticleModel<M>,
  measurements: readonly (M | null)[],
  options: ParticleOptions,
): ParticleStep[] {
  const { particles: n, resampleBelow = 0.5, seed = 1 } = options;
  const rng = createRng(seed);
  let cloud = Array.from({ length: n }, () => model.initial(rng));
  let weights = new Array<number>(n).fill(1 / n);
  const out: ParticleStep[] = [];
  measurements.forEach((z, t) => {
    if (t > 0) cloud = cloud.map((p) => model.propagate(p, rng));
    if (z !== null) {
      const logW = cloud.map(
        (p, i) => Math.log(Math.max(weights[i] as number, 1e-300)) + model.logLikelihood(p, z),
      );
      const m = Math.max(...logW);
      const w = logW.map((l) => Math.exp(l - m));
      const s = w.reduce((a, b) => a + b, 0);
      weights = w.map((v) => v / s);
    }
    const ess = effectiveSampleSize(weights);
    const dims = (cloud[0] as number[]).length;
    const mean = new Array<number>(dims).fill(0);
    cloud.forEach((p, i) =>
      p.forEach((v, d) => (mean[d] = (mean[d] as number) + v * (weights[i] as number))),
    );
    let resampled = false;
    const snapshot = cloud.map((p) => [...p]);
    const snapWeights = [...weights];
    if (ess < resampleBelow * n) {
      const idx = systematicResample(weights, rng.next());
      cloud = idx.map((i) => [...(cloud[i] as number[])]);
      weights = new Array<number>(n).fill(1 / n);
      resampled = true;
    }
    out.push({ particles: snapshot, weights: snapWeights, mean, ess, resampled });
  });
  return out;
}

// ---- helpers for simulations ---------------------------------------------------------------

/** Constant-velocity model in 2-D: state (x, y, vx, vy), measurement (x, y), time step `dt`. */
export function constantVelocity2D(
  dt: number,
  processNoise: number,
  measurementNoise: number,
): LinearGaussianModel {
  const F = Matrix.from([
    [1, 0, dt, 0],
    [0, 1, 0, dt],
    [0, 0, 1, 0],
    [0, 0, 0, 1],
  ]);
  const H = Matrix.from([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
  ]);
  // white-noise acceleration: the standard discretised covariance
  const q = processNoise ** 2;
  const a = (dt ** 4 / 4) * q;
  const b = (dt ** 3 / 2) * q;
  const c = dt ** 2 * q;
  const Q = Matrix.from([
    [a, 0, b, 0],
    [0, a, 0, b],
    [b, 0, c, 0],
    [0, b, 0, c],
  ]);
  const R = Matrix.from([
    [measurementNoise ** 2, 0],
    [0, measurementNoise ** 2],
  ]);
  return { F, H, Q, R };
}
