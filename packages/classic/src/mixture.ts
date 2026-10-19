import {
  Matrix,
  cholesky,
  choleskySolve,
  logDetFromCholesky,
  createRng,
  type Clusterer,
} from '@sim/core';
import { KMeans } from './kmeans';

/**
 * Gaussian mixture model fitted by expectation–maximisation. Each cluster is a Gaussian with its
 * own weight, mean and covariance; a point belongs to every cluster *softly* (a responsibility).
 * E-step: compute those responsibilities. M-step: re-estimate every cluster from the points,
 * weighted by them. The likelihood never decreases. Conventions follow scikit-learn's
 * `GaussianMixture`, including `reg_covar` and the convergence test on the mean log-likelihood.
 */
export type CovarianceType = 'full' | 'diag' | 'spherical' | 'tied';

export interface MixtureInit {
  weights: ArrayLike<number>;
  means: Matrix;
  /** One d×d covariance per component. */
  covariances: Matrix[];
}

export interface MixtureOptions {
  k: number;
  covariance?: CovarianceType;
  maxIterations?: number;
  /** Stop when the mean log-likelihood improves by less than this. */
  tolerance?: number;
  /** Added to the diagonal of every covariance to keep it invertible. */
  regCovar?: number;
  /** Explicit starting point, or `kmeans` (default): start from a k-means partition. */
  init?: MixtureInit | 'kmeans';
  seed?: number;
}

export interface MixtureState {
  iteration: number;
  weights: Float64Array;
  means: Matrix;
  covariances: Matrix[];
  /** Mean log-likelihood of the data under these parameters. */
  logLikelihood: number;
  responsibilities: Matrix;
}

const LOG_2PI = Math.log(2 * Math.PI);

/** Log density of one Gaussian at every row of x, via its Cholesky factor. */
function logGaussian(x: Matrix, mean: Float64Array | ArrayLike<number>, cov: Matrix): Float64Array {
  const d = x.cols;
  const l = cholesky(cov);
  const logDet = logDetFromCholesky(l);
  return Float64Array.from({ length: x.rows }, (_, i) => {
    const diff = new Matrix(
      d,
      1,
      Float64Array.from({ length: d }, (_, j) => x.get(i, j) - (mean[j] as number)),
    );
    const sol = choleskySolve(l, diff);
    let maha = 0;
    for (let j = 0; j < d; j++) maha += (diff.data[j] as number) * (sol.data[j] as number);
    return -0.5 * (d * LOG_2PI + logDet + maha);
  });
}

function eStep(
  x: Matrix,
  weights: Float64Array,
  means: Matrix,
  covs: Matrix[],
): { resp: Matrix; meanLogLik: number } {
  const n = x.rows;
  const k = weights.length;
  const logp = Array.from({ length: k }, (_, c) =>
    logGaussian(x, means.data.subarray(c * x.cols, (c + 1) * x.cols), covs[c] as Matrix),
  );
  const resp = new Matrix(n, k);
  let total = 0;
  for (let i = 0; i < n; i++) {
    let max = -Infinity;
    const row = new Float64Array(k);
    for (let c = 0; c < k; c++) {
      row[c] = Math.log(weights[c] as number) + (logp[c] as Float64Array)[i]!;
      max = Math.max(max, row[c] as number);
    }
    let s = 0;
    for (let c = 0; c < k; c++) s += Math.exp((row[c] as number) - max);
    const lse = max + Math.log(s);
    total += lse;
    for (let c = 0; c < k; c++) resp.set(i, c, Math.exp((row[c] as number) - lse));
  }
  return { resp, meanLogLik: total / n };
}

function mStep(
  x: Matrix,
  resp: Matrix,
  type: CovarianceType,
  regCovar: number,
): { weights: Float64Array; means: Matrix; covs: Matrix[] } {
  const n = x.rows;
  const d = x.cols;
  const k = resp.cols;
  const nk = Float64Array.from({ length: k }, (_, c) => {
    let s = 0;
    for (let i = 0; i < n; i++) s += resp.get(i, c);
    return s + 10 * Number.EPSILON;
  });
  const means = new Matrix(k, d);
  for (let c = 0; c < k; c++)
    for (let j = 0; j < d; j++) {
      let s = 0;
      for (let i = 0; i < n; i++) s += resp.get(i, c) * x.get(i, j);
      means.set(c, j, s / (nk[c] as number));
    }
  const full = (c: number): Matrix => {
    const cov = new Matrix(d, d);
    for (let a = 0; a < d; a++)
      for (let b = a; b < d; b++) {
        let s = 0;
        for (let i = 0; i < n; i++)
          s += resp.get(i, c) * (x.get(i, a) - means.get(c, a)) * (x.get(i, b) - means.get(c, b));
        cov.set(a, b, s / (nk[c] as number));
        cov.set(b, a, s / (nk[c] as number));
      }
    return cov;
  };
  let covs: Matrix[];
  if (type === 'tied') {
    // pooled covariance: weighted average of the per-component scatter
    const pooled = new Matrix(d, d);
    for (let c = 0; c < k; c++) {
      const cv = full(c);
      for (let i = 0; i < d * d; i++)
        pooled.data[i] =
          (pooled.data[i] as number) + ((nk[c] as number) * (cv.data[i] as number)) / n;
    }
    covs = Array.from({ length: k }, () => pooled);
  } else {
    covs = Array.from({ length: k }, (_, c) => {
      const cv = full(c);
      if (type === 'full') return cv;
      const m = new Matrix(d, d);
      if (type === 'diag') for (let j = 0; j < d; j++) m.set(j, j, cv.get(j, j));
      else {
        let tr = 0;
        for (let j = 0; j < d; j++) tr += cv.get(j, j);
        for (let j = 0; j < d; j++) m.set(j, j, tr / d);
      }
      return m;
    });
  }
  // keep every covariance invertible: widen each diagonal by regCovar (a tied covariance only once)
  const widen = (cv: Matrix): Matrix => {
    const out = new Matrix(d, d, Float64Array.from(cv.data));
    for (let j = 0; j < d; j++) out.set(j, j, out.get(j, j) + regCovar);
    return out;
  };
  if (type === 'tied') {
    const shared = widen(covs[0] as Matrix);
    covs = covs.map(() => shared);
  } else covs = covs.map(widen);
  return { weights: Float64Array.from(nk, (v) => v / n), means, covs };
}

export class GaussianMixture implements Clusterer {
  weights: Float64Array = new Float64Array(0);
  means: Matrix = new Matrix(0, 0);
  covariances: Matrix[] = [];
  labels: Int32Array = new Int32Array(0);
  /** Responsibilities of the training points (`n × k`). */
  responsibilities: Matrix = new Matrix(0, 0);
  /** Mean log-likelihood at the last E-step of fitting. */
  lowerBound = -Infinity;
  iterations = 0;
  converged = false;

  constructor(readonly options: MixtureOptions) {}

  private get type(): CovarianceType {
    return this.options.covariance ?? 'full';
  }

  private startingPoint(x: Matrix): { weights: Float64Array; means: Matrix; covs: Matrix[] } {
    const init = this.options.init ?? 'kmeans';
    if (init !== 'kmeans') {
      return {
        weights: Float64Array.from(init.weights),
        means: init.means,
        covs: init.covariances,
      };
    }
    const km = new KMeans({ k: this.options.k, restarts: 3, seed: this.options.seed ?? 1 }).fit(x);
    const resp = new Matrix(x.rows, this.options.k);
    km.labels.forEach((l, i) => resp.set(i, l, 1));
    return mStep(x, resp, this.type, this.options.regCovar ?? 1e-6);
  }

  /** The EM iterations as a generator: state 0 is the starting point, then one state per M-step. */
  *states(x: Matrix): Generator<MixtureState, void, undefined> {
    const { maxIterations = 100, tolerance = 1e-3, regCovar = 1e-6 } = this.options;
    let { weights, means, covs } = this.startingPoint(x);
    const first = eStep(x, weights, means, covs);
    yield {
      iteration: 0,
      weights,
      means,
      covariances: covs,
      logLikelihood: first.meanLogLik,
      responsibilities: first.resp,
    };
    let lower = -Infinity;
    for (let it = 1; it <= maxIterations; it++) {
      const previous = lower;
      const e = eStep(x, weights, means, covs);
      lower = e.meanLogLik;
      ({ weights, means, covs } = mStep(x, e.resp, this.type, regCovar));
      const after = eStep(x, weights, means, covs);
      yield {
        iteration: it,
        weights,
        means,
        covariances: covs,
        logLikelihood: after.meanLogLik,
        responsibilities: after.resp,
      };
      if (Math.abs(lower - previous) < tolerance) break;
    }
  }

  fit(x: Matrix): this {
    const { maxIterations = 100, tolerance = 1e-3, regCovar = 1e-6 } = this.options;
    let { weights, means, covs } = this.startingPoint(x);
    let lower = -Infinity;
    this.converged = false;
    let it = 1;
    for (; it <= maxIterations; it++) {
      const previous = lower;
      const e = eStep(x, weights, means, covs);
      ({ weights, means, covs } = mStep(x, e.resp, this.type, regCovar));
      lower = e.meanLogLik;
      if (Math.abs(lower - previous) < tolerance) {
        this.converged = true;
        break;
      }
    }
    this.iterations = Math.min(it, maxIterations);
    this.lowerBound = lower;
    this.weights = weights;
    this.means = means;
    this.covariances = covs;
    // a final E-step makes the labels consistent with the final parameters
    const final = eStep(x, weights, means, covs);
    this.responsibilities = final.resp;
    this.labels = Int32Array.from({ length: x.rows }, (_, i) => argmaxRow(final.resp, i));
    return this;
  }

  /** Mean log-likelihood of `x` under the fitted mixture. */
  score(x: Matrix): number {
    return eStep(x, this.weights, this.means, this.covariances).meanLogLik;
  }

  predictProba(x: Matrix): Matrix {
    return eStep(x, this.weights, this.means, this.covariances).resp;
  }

  predict(x: Matrix): Int32Array {
    const p = this.predictProba(x);
    return Int32Array.from({ length: x.rows }, (_, i) => argmaxRow(p, i));
  }

  /** Number of free parameters (scikit-learn's convention). */
  parameterCount(d: number): number {
    const k = this.options.k;
    const cov =
      this.type === 'full'
        ? (k * d * (d + 1)) / 2
        : this.type === 'diag'
          ? k * d
          : this.type === 'spherical'
            ? k
            : (d * (d + 1)) / 2;
    return cov + k * d + k - 1;
  }

  bic(x: Matrix): number {
    return -2 * this.score(x) * x.rows + this.parameterCount(x.cols) * Math.log(x.rows);
  }

  aic(x: Matrix): number {
    return -2 * this.score(x) * x.rows + 2 * this.parameterCount(x.cols);
  }

  /** Draws `count` new points from the fitted mixture. */
  sample(count: number, seed = 1): Matrix {
    const rng = createRng(seed);
    const d = this.means.cols;
    const chol = this.covariances.map((c) => cholesky(c));
    const out = new Matrix(count, d);
    for (let i = 0; i < count; i++) {
      let u = rng.next();
      let c = 0;
      while (c < this.weights.length - 1 && u > (this.weights[c] as number)) {
        u -= this.weights[c] as number;
        c++;
      }
      const z = Float64Array.from({ length: d }, () => rng.normal());
      for (let a = 0; a < d; a++) {
        let s = this.means.get(c, a);
        for (let b = 0; b <= a; b++) s += (chol[c] as Matrix).get(a, b) * (z[b] as number);
        out.set(i, a, s);
      }
    }
    return out;
  }
}

function argmaxRow(m: Matrix, i: number): number {
  let best = 0;
  for (let c = 1; c < m.cols; c++) if (m.get(i, c) > m.get(i, best)) best = c;
  return best;
}

/** BIC for k = 1 … maxK (lower is better): the usual way to choose the number of components. */
export function bicCurve(
  x: Matrix,
  maxK: number,
  options: Omit<MixtureOptions, 'k'> = {},
): number[] {
  return Array.from({ length: maxK }, (_, i) =>
    new GaussianMixture({ ...options, k: i + 1 }).fit(x).bic(x),
  );
}
