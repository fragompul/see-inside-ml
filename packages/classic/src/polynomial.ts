import { Matrix, Standardizer, type Regressor } from '@sim/core';
import { ElasticNet } from './regularized';
import { LinearRegression } from './linear';

/** Columns `x, x², …, x^degree` of one feature (no constant column). */
export function polynomialFeatures(x: ArrayLike<number> | Matrix, degree: number): Matrix {
  const n = x instanceof Matrix ? x.rows : x.length;
  const value = (i: number) => (x instanceof Matrix ? x.get(i, 0) : (x[i] as number));
  const out = new Matrix(n, degree);
  for (let i = 0; i < n; i++) {
    let p = 1;
    const v = value(i);
    for (let k = 0; k < degree; k++) {
      p *= v;
      out.set(i, k, p);
    }
  }
  return out;
}

export type PolynomialPenalty = 'none' | 'ridge' | 'lasso';

export interface PolynomialOptions {
  degree: number;
  penalty?: PolynomialPenalty;
  /** Strength of the penalty (scikit-learn's α). */
  alpha?: number;
}

/**
 * Polynomial regression on one feature: expand `x` into its powers, standardise them (so the
 * penalty treats every power alike and the normal equations stay well conditioned), then fit a
 * linear model. Equivalent to scikit-learn's
 * `make_pipeline(PolynomialFeatures(d, include_bias=False), StandardScaler(), Ridge(alpha))`.
 */
export class PolynomialRegression implements Regressor {
  private scaler = new Standardizer();
  private inner: LinearRegression | ElasticNet = new LinearRegression();
  /** Largest |x| seen in training; powers are taken of `x / scale` so the expansion stays tame. */
  private scale = 1;

  constructor(readonly options: PolynomialOptions) {}

  private expand(x: Matrix): Matrix {
    const scaled = Float64Array.from({ length: x.rows }, (_, i) => x.get(i, 0) / this.scale);
    return polynomialFeatures(scaled, this.options.degree);
  }

  fit(x: Matrix, y: ArrayLike<number>): this {
    const { penalty = 'none', alpha = 0 } = this.options;
    let m = 0;
    for (let i = 0; i < x.rows; i++) m = Math.max(m, Math.abs(x.get(i, 0)));
    this.scale = m > 0 ? m : 1;
    const z = this.scaler.fit(this.expand(x)).transform(this.expand(x));
    this.inner =
      penalty === 'lasso'
        ? new ElasticNet({ alpha, l1Ratio: 1 })
        : new LinearRegression({ l2: penalty === 'ridge' ? alpha : 0 });
    this.inner.fit(z, y);
    return this;
  }

  predict(x: Matrix): Float64Array {
    return this.inner.predict(this.scaler.transform(this.expand(x)));
  }

  /** Coefficients of the standardised powers x/s, (x/s)², … */
  get coef(): Float64Array {
    return this.inner.coef;
  }

  get intercept(): number {
    return this.inner.intercept;
  }
}
