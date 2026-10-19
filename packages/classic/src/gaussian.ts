import { Matrix, cholesky, choleskySolve, logDetFromCholesky, type Classifier } from '@sim/core';

/**
 * Classifiers that model each class as a Gaussian and apply Bayes' rule:
 *
 *   P(c | x) ∝ π_c · N(x; μ_c, Σ_c)
 *
 * They differ only in the covariance they allow: `GaussianNB` a diagonal one per class (features
 * independent given the class), `LDA` one covariance shared by all classes (straight boundaries),
 * `QDA` a full covariance per class (curved boundaries).
 */

interface ClassStats {
  count: number[];
  prior: number[];
  mean: Float64Array[];
}

function classStats(x: Matrix, y: ArrayLike<number>, priors?: readonly number[]): ClassStats {
  let k = 0;
  for (let i = 0; i < y.length; i++) k = Math.max(k, (y[i] as number) + 1);
  k = Math.max(k, 2);
  const count = new Array<number>(k).fill(0);
  const mean = Array.from({ length: k }, () => new Float64Array(x.cols));
  for (let i = 0; i < x.rows; i++) {
    const c = y[i] as number;
    count[c] = (count[c] as number) + 1;
    for (let j = 0; j < x.cols; j++) mean[c]![j] = (mean[c]![j] as number) + x.get(i, j);
  }
  for (let c = 0; c < k; c++)
    for (let j = 0; j < x.cols; j++)
      mean[c]![j] = (mean[c]![j] as number) / Math.max(1, count[c] as number);
  const prior = priors ? [...priors] : count.map((n) => n / x.rows);
  return { count, prior, mean };
}

/** Covariance (divisor `n`, as scikit-learn's `empirical_covariance`) of the rows of class `c`. */
function classCovariance(
  x: Matrix,
  y: ArrayLike<number>,
  mean: Float64Array,
  c: number,
  ddof: number,
): Matrix {
  const d = x.cols;
  const cov = new Matrix(d, d);
  let n = 0;
  for (let i = 0; i < x.rows; i++) {
    if (y[i] !== c) continue;
    n++;
    for (let a = 0; a < d; a++)
      for (let b = 0; b < d; b++)
        cov.data[a * d + b] =
          (cov.data[a * d + b] as number) +
          (x.get(i, a) - (mean[a] as number)) * (x.get(i, b) - (mean[b] as number));
  }
  const div = Math.max(1, n - ddof);
  for (let i = 0; i < d * d; i++) cov.data[i] = (cov.data[i] as number) / div;
  return cov;
}

/** Turns per-class log-joint scores into probabilities (log-sum-exp for stability). */
function softmaxRows(scores: Matrix): Matrix {
  const out = new Matrix(scores.rows, scores.cols);
  for (let i = 0; i < scores.rows; i++) {
    let max = -Infinity;
    for (let c = 0; c < scores.cols; c++) max = Math.max(max, scores.get(i, c));
    let z = 0;
    for (let c = 0; c < scores.cols; c++) z += Math.exp(scores.get(i, c) - max);
    for (let c = 0; c < scores.cols; c++) out.set(i, c, Math.exp(scores.get(i, c) - max) / z);
  }
  return out;
}

abstract class GaussianClassifier implements Classifier {
  classes = 2;
  priors: number[] = [];
  means: Float64Array[] = [];
  /** Covariance of each class (the same matrix repeated for LDA, diagonal for naive Bayes). */
  covariances: Matrix[] = [];

  abstract fit(x: Matrix, y: ArrayLike<number>): this;

  /** `n × classes` log of `π_c · N(x; μ_c, Σ_c)` up to a constant shared by the classes. */
  abstract logJoint(x: Matrix): Matrix;

  predictProba(x: Matrix): Matrix {
    return softmaxRows(this.logJoint(x));
  }

  predict(x: Matrix): Int32Array {
    const s = this.logJoint(x);
    return Int32Array.from({ length: x.rows }, (_, i) => {
      let best = 0;
      for (let c = 1; c < this.classes; c++) if (s.get(i, c) > s.get(i, best)) best = c;
      return best;
    });
  }
}

export class GaussianNB extends GaussianClassifier {
  variances: Float64Array[] = [];

  /** `varSmoothing` adds that fraction of the largest feature variance to every variance. */
  constructor(readonly options: { varSmoothing?: number; priors?: readonly number[] } = {}) {
    super();
  }

  fit(x: Matrix, y: ArrayLike<number>): this {
    const s = classStats(x, y, this.options.priors);
    this.classes = s.prior.length;
    this.priors = s.prior;
    this.means = s.mean;
    const d = x.cols;
    let maxVar = 0;
    for (let j = 0; j < d; j++) {
      let m = 0;
      for (let i = 0; i < x.rows; i++) m += x.get(i, j);
      m /= x.rows;
      let v = 0;
      for (let i = 0; i < x.rows; i++) v += (x.get(i, j) - m) ** 2;
      maxVar = Math.max(maxVar, v / x.rows);
    }
    const eps = (this.options.varSmoothing ?? 1e-9) * maxVar;
    this.variances = s.mean.map((mu, c) => {
      const v = new Float64Array(d);
      for (let i = 0; i < x.rows; i++)
        if (y[i] === c)
          for (let j = 0; j < d; j++)
            v[j] = (v[j] as number) + (x.get(i, j) - (mu[j] as number)) ** 2;
      return v.map((a) => a / Math.max(1, s.count[c] as number) + eps);
    });
    this.covariances = this.variances.map((v) => {
      const m = new Matrix(d, d);
      v.forEach((a, j) => m.set(j, j, a));
      return m;
    });
    return this;
  }

  logJoint(x: Matrix): Matrix {
    const out = new Matrix(x.rows, this.classes);
    for (let i = 0; i < x.rows; i++)
      for (let c = 0; c < this.classes; c++) {
        let s = Math.log(this.priors[c] as number);
        const mu = this.means[c] as Float64Array;
        const v = this.variances[c] as Float64Array;
        for (let j = 0; j < x.cols; j++)
          s -=
            0.5 *
            (Math.log(2 * Math.PI * (v[j] as number)) +
              (x.get(i, j) - (mu[j] as number)) ** 2 / (v[j] as number));
        out.set(i, c, s);
      }
    return out;
  }
}

export class LDA extends GaussianClassifier {
  /** Shared covariance Σ = Σ_c π_c Σ_c (scikit-learn's `lsqr` convention). */
  pooled: Matrix = new Matrix(0, 0);
  private solved: Matrix[] = [];
  private offsets: number[] = [];

  /**
   * `shrinkage` ∈ [0, 1] pulls each class covariance towards a spherical one , which stabilises it when there are few points per feature.
   */
  constructor(readonly options: { shrinkage?: number; priors?: readonly number[] } = {}) {
    super();
  }

  fit(x: Matrix, y: ArrayLike<number>): this {
    const s = classStats(x, y, this.options.priors);
    this.classes = s.prior.length;
    this.priors = s.prior;
    this.means = s.mean;
    const d = x.cols;
    const shrink = this.options.shrinkage ?? 0;
    const pooled = new Matrix(d, d);
    for (let c = 0; c < this.classes; c++) {
      const cov = shrinkCovariance(classCovariance(x, y, s.mean[c] as Float64Array, c, 0), shrink);
      for (let i = 0; i < d * d; i++)
        pooled.data[i] =
          (pooled.data[i] as number) + (s.prior[c] as number) * (cov.data[i] as number);
    }
    this.pooled = pooled;
    this.covariances = Array.from({ length: this.classes }, () => pooled);
    const l = cholesky(pooled);
    this.solved = s.mean.map((mu) => choleskySolve(l, new Matrix(d, 1, Float64Array.from(mu))));
    this.offsets = s.mean.map((mu, c) => {
      let q = 0;
      (this.solved[c] as Matrix).data.forEach((v, j) => (q += v * (mu[j] as number)));
      return Math.log(s.prior[c] as number) - 0.5 * q;
    });
    return this;
  }

  logJoint(x: Matrix): Matrix {
    const out = new Matrix(x.rows, this.classes);
    for (let i = 0; i < x.rows; i++)
      for (let c = 0; c < this.classes; c++) {
        let s = this.offsets[c] as number;
        const w = (this.solved[c] as Matrix).data;
        for (let j = 0; j < x.cols; j++) s += x.get(i, j) * (w[j] as number);
        out.set(i, c, s);
      }
    return out;
  }

  /** The weights `w_c = Σ⁻¹μ_c` and offsets of the linear scores, for drawing the boundary. */
  linearScores(): { weights: Float64Array[]; offsets: number[] } {
    return {
      weights: this.solved.map((m) => Float64Array.from(m.data)),
      offsets: [...this.offsets],
    };
  }
}

/** scikit-learn's `shrunk_covariance`: (1 − s)·Σ + s·(trace(Σ)/d)·I. */
function shrinkCovariance(cov: Matrix, shrinkage: number): Matrix {
  if (shrinkage <= 0) return cov;
  const d = cov.rows;
  let trace = 0;
  for (let j = 0; j < d; j++) trace += cov.get(j, j);
  const mu = trace / d;
  const out = new Matrix(d, d);
  for (let a = 0; a < d; a++)
    for (let b = 0; b < d; b++)
      out.set(a, b, (1 - shrinkage) * cov.get(a, b) + (a === b ? shrinkage * mu : 0));
  return out;
}

export class QDA extends GaussianClassifier {
  private factors: Matrix[] = [];
  private logDets: number[] = [];

  /** `regParam` ∈ [0, 1] blends each class covariance with the identity: (1 − r)·Σ + r·I. */
  constructor(readonly options: { regParam?: number; priors?: readonly number[] } = {}) {
    super();
  }

  fit(x: Matrix, y: ArrayLike<number>): this {
    const s = classStats(x, y, this.options.priors);
    this.classes = s.prior.length;
    this.priors = s.prior;
    this.means = s.mean;
    const d = x.cols;
    const r = this.options.regParam ?? 0;
    this.covariances = s.mean.map((mu, c) => {
      const cov = classCovariance(x, y, mu, c, 1); // unbiased, like scikit-learn's SVD of the class
      for (let i = 0; i < d * d; i++) cov.data[i] = (1 - r) * (cov.data[i] as number);
      for (let j = 0; j < d; j++) cov.set(j, j, cov.get(j, j) + r);
      return cov;
    });
    this.factors = this.covariances.map((cov) => cholesky(cov));
    this.logDets = this.factors.map((l) => logDetFromCholesky(l));
    return this;
  }

  logJoint(x: Matrix): Matrix {
    const d = x.cols;
    const out = new Matrix(x.rows, this.classes);
    for (let i = 0; i < x.rows; i++)
      for (let c = 0; c < this.classes; c++) {
        const mu = this.means[c] as Float64Array;
        const diff = new Matrix(
          d,
          1,
          Float64Array.from({ length: d }, (_, j) => x.get(i, j) - (mu[j] as number)),
        );
        const sol = choleskySolve(this.factors[c] as Matrix, diff);
        let maha = 0;
        for (let j = 0; j < d; j++) maha += (diff.data[j] as number) * (sol.data[j] as number);
        out.set(
          i,
          c,
          Math.log(this.priors[c] as number) - 0.5 * ((this.logDets[c] as number) + maha),
        );
      }
    return out;
  }
}
