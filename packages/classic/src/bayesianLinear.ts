import { Matrix, inverse, symmetricEigen, type Regressor } from '@sim/core';

/**
 * Bayesian linear regression with evidence maximisation (scikit-learn's `BayesianRidge`): a
 * Gaussian prior on the weights with precision λ and Gaussian noise with precision α. Instead of one
 * weight vector it keeps a whole posterior, so every prediction comes with an uncertainty, and α, λ
 * are tuned to the data by maximising the marginal likelihood.
 */
export interface BayesianRidgeOptions {
  maxIterations?: number;
  /** Stop when the coefficients move by less than this (sum of absolute changes). */
  tolerance?: number;
  /** Gamma hyper-priors on the precisions (scikit-learn defaults: 1e-6). */
  alpha1?: number;
  alpha2?: number;
  lambda1?: number;
  lambda2?: number;
  intercept?: boolean;
}

export class BayesianRidge implements Regressor {
  coef: Float64Array = new Float64Array(0);
  intercept = 0;
  /** Noise precision α = 1/σ². */
  alpha = 1;
  /** Weight precision λ. */
  lambda = 1;
  iterations = 0;
  /** Posterior covariance of the weights. */
  covariance: Matrix = new Matrix(0, 0);
  private xMean: Float64Array = new Float64Array(0);

  constructor(readonly options: BayesianRidgeOptions = {}) {}

  fit(x: Matrix, y: ArrayLike<number>): this {
    const {
      maxIterations = 300,
      tolerance = 1e-3,
      alpha1 = 1e-6,
      alpha2 = 1e-6,
      lambda1 = 1e-6,
      lambda2 = 1e-6,
      intercept = true,
    } = this.options;
    const n = x.rows;
    const d = x.cols;
    this.xMean = new Float64Array(d);
    let yMean = 0;
    if (intercept) {
      for (let j = 0; j < d; j++) {
        for (let i = 0; i < n; i++) this.xMean[j] = (this.xMean[j] as number) + x.get(i, j) / n;
      }
      for (let i = 0; i < n; i++) yMean += (y[i] as number) / n;
    }
    const xc = new Matrix(n, d);
    for (let i = 0; i < n; i++)
      for (let j = 0; j < d; j++) xc.set(i, j, x.get(i, j) - (this.xMean[j] as number));
    const yc = Float64Array.from({ length: n }, (_, i) => (y[i] as number) - yMean);

    // XᵀX and Xᵀy, and the eigenvalues of XᵀX (used for the effective number of parameters γ)
    const xtx = new Matrix(d, d);
    const xty = new Float64Array(d);
    for (let i = 0; i < n; i++)
      for (let a = 0; a < d; a++) {
        xty[a] = (xty[a] as number) + xc.get(i, a) * (yc[i] as number);
        for (let b = 0; b < d; b++)
          xtx.data[a * d + b] = (xtx.data[a * d + b] as number) + xc.get(i, a) * xc.get(i, b);
      }
    const eig = symmetricEigen(xtx).values;

    let variance = 0;
    for (const v of yc) variance += v * v;
    variance /= n;
    let alpha = 1 / (variance + Number.EPSILON);
    let lambda = 1;
    let coef = new Float64Array(d);
    let it = 0;
    for (; it < maxIterations; it++) {
      // posterior mean for the current precisions: (XᵀX + (λ/α) I)⁻¹ Xᵀy
      const a = new Matrix(d, d, Float64Array.from(xtx.data));
      for (let j = 0; j < d; j++) a.set(j, j, a.get(j, j) + lambda / alpha);
      const inv = inverse(a);
      const next = Float64Array.from({ length: d }, (_, r) => {
        let s = 0;
        for (let c = 0; c < d; c++) s += inv.get(r, c) * (xty[c] as number);
        return s;
      });
      let rss = 0;
      for (let i = 0; i < n; i++) {
        let p = 0;
        for (let j = 0; j < d; j++) p += xc.get(i, j) * (next[j] as number);
        rss += ((yc[i] as number) - p) ** 2;
      }
      let gamma = 0;
      for (const e of eig) gamma += (alpha * e) / (lambda + alpha * e);
      let wNorm = 0;
      for (const v of next) wNorm += v * v;
      lambda = (gamma + 2 * lambda1) / (wNorm + 2 * lambda2);
      alpha = (n - gamma + 2 * alpha1) / (rss + 2 * alpha2);
      const change = next.reduce((s, v, j) => s + Math.abs(v - (coef[j] as number)), 0);
      coef = next;
      if (it > 0 && change < tolerance) {
        it++;
        break;
      }
    }
    this.iterations = it;
    this.alpha = alpha;
    this.lambda = lambda;
    // final posterior with the converged precisions
    const a = new Matrix(d, d, Float64Array.from(xtx.data));
    for (let j = 0; j < d; j++) a.set(j, j, a.get(j, j) + lambda / alpha);
    this.covariance = inverse(a);
    for (let i = 0; i < d * d; i++)
      this.covariance.data[i] = (this.covariance.data[i] as number) / alpha;
    this.coef = Float64Array.from({ length: d }, (_, r) => {
      let s = 0;
      for (let c = 0; c < d; c++)
        s += (this.covariance.get(r, c) as number) * alpha * (xty[c] as number);
      return s;
    });
    this.intercept = intercept
      ? yMean - this.coef.reduce((s, w, j) => s + w * (this.xMean[j] as number), 0)
      : 0;
    return this;
  }

  predict(x: Matrix): Float64Array {
    return Float64Array.from({ length: x.rows }, (_, i) => {
      let s = this.intercept;
      for (let j = 0; j < x.cols; j++) s += x.get(i, j) * (this.coef[j] as number);
      return s;
    });
  }

  /** Predictive standard deviation: noise plus the uncertainty about the weights. */
  predictStd(x: Matrix): Float64Array {
    return Float64Array.from({ length: x.rows }, (_, i) => {
      let q = 0;
      for (let a = 0; a < x.cols; a++)
        for (let b = 0; b < x.cols; b++)
          q +=
            (x.get(i, a) - (this.xMean[a] as number)) *
            this.covariance.get(a, b) *
            (x.get(i, b) - (this.xMean[b] as number));
      return Math.sqrt(1 / this.alpha + q);
    });
  }
}
