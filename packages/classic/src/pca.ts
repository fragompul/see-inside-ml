import { Matrix, symmetricEigen, type Projector } from '@sim/core';

/**
 * Principal component analysis: rotate the data to the directions of greatest variance and keep the
 * first few. The components are the eigenvectors of the covariance matrix; the eigenvalues are the
 * variance along each. Kernel PCA does the same in an implicit feature space, which lets it unroll
 * curved shapes. Conventions (signs, ddof) follow scikit-learn.
 */
export interface PcaOptions {
  /** Components to keep (default all). */
  components?: number;
  /** Divide each score by √variance so every component has unit variance. */
  whiten?: boolean;
}

/** Flips a vector so its entry of largest magnitude is positive (scikit-learn's `svd_flip` convention). */
function flipSign(v: Float64Array): Float64Array {
  let best = 0;
  for (let i = 1; i < v.length; i++)
    if (Math.abs(v[i] as number) > Math.abs(v[best] as number)) best = i;
  return (v[best] as number) < 0 ? v.map((a) => -a) : v;
}

export class PCA implements Projector {
  mean: Float64Array = new Float64Array(0);
  /** Rows are components (`k × d`), unit length, ordered by decreasing variance. */
  components: Matrix = new Matrix(0, 0);
  /** Variance along each component (covariance eigenvalues, divisor n − 1). */
  explainedVariance: Float64Array = new Float64Array(0);
  explainedVarianceRatio: Float64Array = new Float64Array(0);
  /** Variance of every principal direction, kept ones or not. */
  allVariance: Float64Array = new Float64Array(0);

  constructor(readonly options: PcaOptions = {}) {}

  fit(x: Matrix): this {
    const n = x.rows;
    const d = x.cols;
    this.mean = new Float64Array(d);
    for (let j = 0; j < d; j++) {
      let s = 0;
      for (let i = 0; i < n; i++) s += x.get(i, j);
      this.mean[j] = s / n;
    }
    const cov = new Matrix(d, d);
    for (let a = 0; a < d; a++)
      for (let b = a; b < d; b++) {
        let s = 0;
        for (let i = 0; i < n; i++)
          s += (x.get(i, a) - (this.mean[a] as number)) * (x.get(i, b) - (this.mean[b] as number));
        cov.set(a, b, s / (n - 1));
        cov.set(b, a, s / (n - 1));
      }
    const { values, vectors } = symmetricEigen(cov);
    const k = Math.min(this.options.components ?? d, d);
    this.allVariance = Float64Array.from(values, (v) => Math.max(v, 0));
    const total = this.allVariance.reduce((a, b) => a + b, 0);
    this.components = new Matrix(k, d);
    for (let c = 0; c < k; c++) {
      const v = flipSign(Float64Array.from({ length: d }, (_, j) => vectors.get(j, c)));
      this.components.data.set(v, c * d);
    }
    this.explainedVariance = this.allVariance.slice(0, k);
    this.explainedVarianceRatio = this.explainedVariance.map((v) => (total > 0 ? v / total : 0));
    return this;
  }

  /** Scores: the coordinates of every row along the kept components. */
  transform(x: Matrix): Matrix {
    const k = this.components.rows;
    const out = new Matrix(x.rows, k);
    for (let i = 0; i < x.rows; i++)
      for (let c = 0; c < k; c++) {
        let s = 0;
        for (let j = 0; j < x.cols; j++)
          s += (x.get(i, j) - (this.mean[j] as number)) * this.components.get(c, j);
        out.set(i, c, this.options.whiten ? s / Math.sqrt(this.explainedVariance[c] as number) : s);
      }
    return out;
  }

  /** Maps scores back to the original space (the best reconstruction from the kept components). */
  inverseTransform(z: Matrix): Matrix {
    const out = new Matrix(z.rows, this.components.cols);
    for (let i = 0; i < z.rows; i++)
      for (let j = 0; j < out.cols; j++) {
        let s = this.mean[j] as number;
        for (let c = 0; c < this.components.rows; c++) {
          const score =
            z.get(i, c) *
            (this.options.whiten ? Math.sqrt(this.explainedVariance[c] as number) : 1);
          s += score * this.components.get(c, j);
        }
        out.set(i, j, s);
      }
    return out;
  }

  /** Mean squared distance between the data and their reconstruction. */
  reconstructionError(x: Matrix): number {
    const r = this.inverseTransform(this.transform(x));
    let s = 0;
    for (let i = 0; i < x.data.length; i++)
      s += ((x.data[i] as number) - (r.data[i] as number)) ** 2;
    return s / x.rows;
  }

  /** Smallest number of components that explains at least `share` of the variance. */
  componentsFor(share: number): number {
    const total = this.allVariance.reduce((a, b) => a + b, 0);
    let acc = 0;
    for (let c = 0; c < this.allVariance.length; c++) {
      acc += this.allVariance[c] as number;
      if (acc / total >= share - 1e-12) return c + 1;
    }
    return this.allVariance.length;
  }
}

// ---- kernel PCA ----------------------------------------------------------------------------------

export type PcaKernel = 'rbf' | 'poly' | 'linear' | 'cosine';

export interface KernelPcaOptions {
  components: number;
  kernel: PcaKernel;
  gamma?: number;
  degree?: number;
  coef0?: number;
}

function kernelFn(
  o: KernelPcaOptions,
  d: number,
): (a: ArrayLike<number>, b: ArrayLike<number>) => number {
  const gamma = o.gamma ?? 1 / d;
  const dot = (a: ArrayLike<number>, b: ArrayLike<number>) => {
    let s = 0;
    for (let j = 0; j < a.length; j++) s += (a[j] as number) * (b[j] as number);
    return s;
  };
  switch (o.kernel) {
    case 'linear':
      return dot;
    case 'poly':
      return (a, b) => (gamma * dot(a, b) + (o.coef0 ?? 1)) ** (o.degree ?? 3);
    case 'cosine':
      return (a, b) => dot(a, b) / (Math.sqrt(dot(a, a) * dot(b, b)) || 1);
    default:
      return (a, b) => {
        let s = 0;
        for (let j = 0; j < a.length; j++) s += ((a[j] as number) - (b[j] as number)) ** 2;
        return Math.exp(-gamma * s);
      };
  }
}

export class KernelPCA implements Projector {
  eigenvalues: Float64Array = new Float64Array(0);
  /** Eigenvectors of the centred kernel matrix as columns (`n × k`). */
  alphas: Matrix = new Matrix(0, 0);
  private train: Matrix = new Matrix(0, 0);
  private kernel: (a: ArrayLike<number>, b: ArrayLike<number>) => number = () => 0;
  private rowMeans: Float64Array = new Float64Array(0);
  private totalMean = 0;

  constructor(readonly options: KernelPcaOptions) {}

  fit(x: Matrix): this {
    const n = x.rows;
    this.train = x;
    this.kernel = kernelFn(this.options, x.cols);
    const rows = Array.from({ length: n }, (_, i) => x.data.subarray(i * x.cols, (i + 1) * x.cols));
    const k = new Matrix(n, n);
    for (let i = 0; i < n; i++)
      for (let j = i; j < n; j++) {
        const v = this.kernel(rows[i] as Float64Array, rows[j] as Float64Array);
        k.set(i, j, v);
        k.set(j, i, v);
      }
    // centre in feature space: K − 1ₙK − K1ₙ + 1ₙK1ₙ
    this.rowMeans = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      let s = 0;
      for (let j = 0; j < n; j++) s += k.get(i, j);
      this.rowMeans[i] = s / n;
    }
    this.totalMean = this.rowMeans.reduce((a, b) => a + b, 0) / n;
    const centred = new Matrix(n, n);
    for (let i = 0; i < n; i++)
      for (let j = 0; j < n; j++)
        centred.set(
          i,
          j,
          k.get(i, j) -
            (this.rowMeans[i] as number) -
            (this.rowMeans[j] as number) +
            this.totalMean,
        );
    const { values, vectors } = symmetricEigen(centred);
    const kk = Math.min(this.options.components, n);
    this.eigenvalues = values.slice(0, kk);
    this.alphas = new Matrix(n, kk);
    for (let c = 0; c < kk; c++) {
      // scikit-learn's svd_flip on the eigenvectors: the entry of largest magnitude is positive
      let best = 0;
      for (let i = 1; i < n; i++)
        if (Math.abs(vectors.get(i, c)) > Math.abs(vectors.get(best, c))) best = i;
      const sign = vectors.get(best, c) < 0 ? -1 : 1;
      for (let i = 0; i < n; i++) this.alphas.set(i, c, sign * vectors.get(i, c));
    }
    return this;
  }

  /** The training points in kernel-principal coordinates (eigenvectors × √eigenvalue). */
  get embedding(): Matrix {
    const out = new Matrix(this.alphas.rows, this.alphas.cols);
    for (let i = 0; i < out.rows; i++)
      for (let c = 0; c < out.cols; c++)
        out.set(
          i,
          c,
          this.alphas.get(i, c) * Math.sqrt(Math.max(this.eigenvalues[c] as number, 0)),
        );
    return out;
  }

  transform(x: Matrix): Matrix {
    const n = this.train.rows;
    const out = new Matrix(x.rows, this.alphas.cols);
    for (let i = 0; i < x.rows; i++) {
      const row = x.data.subarray(i * x.cols, (i + 1) * x.cols);
      const kv = Float64Array.from({ length: n }, (_, j) =>
        this.kernel(row, this.train.data.subarray(j * this.train.cols, (j + 1) * this.train.cols)),
      );
      const mean = kv.reduce((a, b) => a + b, 0) / n;
      for (let c = 0; c < this.alphas.cols; c++) {
        let s = 0;
        for (let j = 0; j < n; j++)
          s +=
            ((kv[j] as number) - mean - (this.rowMeans[j] as number) + this.totalMean) *
            this.alphas.get(j, c);
        const lambda = Math.max(this.eigenvalues[c] as number, 1e-12);
        out.set(i, c, s / Math.sqrt(lambda));
      }
    }
    return out;
  }
}
