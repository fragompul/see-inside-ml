import {
  Matrix,
  cholesky,
  choleskySolve,
  logDetFromCholesky,
  nelderMead,
  type Regressor,
  type Rng,
} from '@sim/core';

/**
 * Gaussian-process regression: a distribution over functions. With a kernel `k` saying how similar
 * two inputs are, the prediction at a new point is a Gaussian whose mean interpolates the data and
 * whose variance grows away from it. Also kernel ridge regression, which is the GP's mean without
 * the uncertainty.
 */
export type GpKernel = 'rbf' | 'matern32' | 'matern52' | 'periodic';

export interface GpOptions {
  kernel?: GpKernel;
  lengthScale?: number;
  /** Signal variance σ_f². */
  variance?: number;
  /** Noise variance σ_n² added to the diagonal. */
  noise?: number;
  /** Period of the periodic kernel. */
  period?: number;
  /** Subtract the mean of y before fitting and add it back (a constant-mean prior). */
  normalizeY?: boolean;
}

export function kernelValue(
  a: ArrayLike<number>,
  b: ArrayLike<number>,
  options: GpOptions,
): number {
  const { kernel = 'rbf', lengthScale = 1, variance = 1, period = 1 } = options;
  let d2 = 0;
  for (let j = 0; j < a.length; j++) d2 += ((a[j] as number) - (b[j] as number)) ** 2;
  const d = Math.sqrt(d2);
  switch (kernel) {
    case 'matern32': {
      const r = (Math.sqrt(3) * d) / lengthScale;
      return variance * (1 + r) * Math.exp(-r);
    }
    case 'matern52': {
      const r = (Math.sqrt(5) * d) / lengthScale;
      return variance * (1 + r + (r * r) / 3) * Math.exp(-r);
    }
    case 'periodic':
      return variance * Math.exp((-2 * Math.sin((Math.PI * d) / period) ** 2) / lengthScale ** 2);
    default:
      return variance * Math.exp(-d2 / (2 * lengthScale ** 2));
  }
}

function gramMatrix(x: Matrix, z: Matrix, options: GpOptions): Matrix {
  const out = new Matrix(x.rows, z.rows);
  for (let i = 0; i < x.rows; i++) {
    const a = x.data.subarray(i * x.cols, (i + 1) * x.cols);
    for (let j = 0; j < z.rows; j++)
      out.set(i, j, kernelValue(a, z.data.subarray(j * z.cols, (j + 1) * z.cols), options));
  }
  return out;
}

export class GaussianProcess implements Regressor {
  private train: Matrix = new Matrix(0, 0);
  private factor: Matrix = new Matrix(0, 0);
  private weights: Float64Array = new Float64Array(0);
  private yMean = 0;
  private y: Float64Array = new Float64Array(0);
  /** The hyperparameters in use (changed by `optimize`). */
  hyper: Required<Omit<GpOptions, 'normalizeY'>> & { normalizeY: boolean };

  constructor(readonly options: GpOptions = {}) {
    this.hyper = {
      kernel: options.kernel ?? 'rbf',
      lengthScale: options.lengthScale ?? 1,
      variance: options.variance ?? 1,
      noise: options.noise ?? 1e-2,
      period: options.period ?? 1,
      normalizeY: options.normalizeY ?? false,
    };
  }

  fit(x: Matrix, y: ArrayLike<number>): this {
    this.train = x;
    this.y = Float64Array.from(y);
    this.yMean = this.hyper.normalizeY ? this.y.reduce((a, b) => a + b, 0) / this.y.length : 0;
    this.refit();
    return this;
  }

  private refit(): void {
    const k = gramMatrix(this.train, this.train, this.hyper);
    for (let i = 0; i < k.rows; i++) k.set(i, i, k.get(i, i) + this.hyper.noise);
    this.factor = cholesky(k);
    const centred = Float64Array.from(this.y, (v) => v - this.yMean);
    this.weights = choleskySolve(this.factor, new Matrix(centred.length, 1, centred)).data;
  }

  /** Log marginal likelihood of the data under the current hyperparameters: the evidence for the model. */
  logMarginalLikelihood(): number {
    const n = this.y.length;
    let quad = 0;
    for (let i = 0; i < n; i++)
      quad += ((this.y[i] as number) - this.yMean) * (this.weights[i] as number);
    return -0.5 * quad - 0.5 * logDetFromCholesky(this.factor) - 0.5 * n * Math.log(2 * Math.PI);
  }

  predict(x: Matrix): Float64Array {
    return this.posterior(x).mean;
  }

  /** Posterior mean and standard deviation of the latent function at the rows of `x`. */
  posterior(x: Matrix): { mean: Float64Array; std: Float64Array } {
    const ks = gramMatrix(this.train, x, this.hyper); // n × m
    const n = this.train.rows;
    const mean = new Float64Array(x.rows);
    const std = new Float64Array(x.rows);
    const v = choleskySolve(this.factor, ks); // K⁻¹ k*
    for (let j = 0; j < x.rows; j++) {
      let m = this.yMean;
      let reduce = 0;
      for (let i = 0; i < n; i++) {
        m += ks.get(i, j) * (this.weights[i] as number);
        reduce += ks.get(i, j) * v.get(i, j);
      }
      mean[j] = m;
      const prior = this.hyper.variance; // k(x, x) is the signal variance for every stationary kernel
      std[j] = Math.sqrt(Math.max(prior - reduce, 0));
    }
    return { mean, std };
  }

  /** Draws `count` functions from the posterior at the rows of `x` (rows of the result). */
  sample(x: Matrix, count: number, rng: Rng): Float64Array[] {
    const m = x.rows;
    const ks = gramMatrix(this.train, x, this.hyper);
    const kss = gramMatrix(x, x, this.hyper);
    const v = choleskySolve(this.factor, ks);
    const cov = new Matrix(m, m);
    for (let a = 0; a < m; a++)
      for (let b = 0; b < m; b++) {
        let s = 0;
        for (let i = 0; i < this.train.rows; i++) s += ks.get(i, a) * v.get(i, b);
        cov.set(a, b, kss.get(a, b) - s + (a === b ? 1e-8 : 0));
      }
    const l = cholesky(cov);
    const mean = this.posterior(x).mean;
    return Array.from({ length: count }, () => {
      const z = Float64Array.from({ length: m }, () => rng.normal());
      return Float64Array.from({ length: m }, (_, i) => {
        let s = mean[i] as number;
        for (let j = 0; j <= i; j++) s += l.get(i, j) * (z[j] as number);
        return s;
      });
    });
  }

  /**
   * Chooses length scale, signal variance and noise by maximising the log marginal likelihood
   * (Nelder–Mead on the logarithms, a few restarts). Returns the new likelihood.
   */
  optimize(restarts = 4): number {
    const objective = (p: readonly number[]) => {
      const h = {
        ...this.hyper,
        lengthScale: Math.exp(p[0] as number),
        variance: Math.exp(p[1] as number),
        noise: Math.exp(p[2] as number),
      };
      try {
        const saved = this.hyper;
        this.hyper = h;
        this.refit();
        const v = -this.logMarginalLikelihood();
        this.hyper = saved;
        return v;
      } catch {
        return Infinity;
      }
    };
    const startLog = [
      Math.log(this.hyper.lengthScale),
      Math.log(this.hyper.variance),
      Math.log(this.hyper.noise),
    ];
    let best = { x: startLog, value: objective(startLog) };
    for (let r = 0; r < restarts; r++) {
      const start =
        r === 0
          ? startLog
          : startLog.map((v, i) => v + (r % 2 ? 1 : -1) * (0.6 + 0.4 * r) * (i === 2 ? 1.5 : 1));
      const res = nelderMead(objective, start, { step: 0.7, maxIterations: 400, tolerance: 1e-12 });
      if (res.value < best.value) best = { x: res.x, value: res.value };
    }
    this.hyper = {
      ...this.hyper,
      lengthScale: Math.exp(best.x[0] as number),
      variance: Math.exp(best.x[1] as number),
      noise: Math.exp(best.x[2] as number),
    };
    this.refit();
    return -best.value;
  }
}

/** Kernel ridge regression: the GP mean with noise acting as a ridge penalty, no intercept. */
export class KernelRidge implements Regressor {
  private train: Matrix = new Matrix(0, 0);
  coef: Float64Array = new Float64Array(0);

  constructor(readonly options: GpOptions & { alpha?: number } = {}) {}

  fit(x: Matrix, y: ArrayLike<number>): this {
    this.train = x;
    const k = gramMatrix(x, x, { ...this.options, variance: this.options.variance ?? 1 });
    const alpha = this.options.alpha ?? 1;
    for (let i = 0; i < k.rows; i++) k.set(i, i, k.get(i, i) + alpha);
    this.coef = choleskySolve(cholesky(k), new Matrix(x.rows, 1, Float64Array.from(y))).data;
    return this;
  }

  predict(x: Matrix): Float64Array {
    const ks = gramMatrix(x, this.train, { ...this.options, variance: this.options.variance ?? 1 });
    return Float64Array.from({ length: x.rows }, (_, i) => {
      let s = 0;
      for (let j = 0; j < this.train.rows; j++) s += ks.get(i, j) * (this.coef[j] as number);
      return s;
    });
  }
}
