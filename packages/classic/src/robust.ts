import { Matrix, median, type Regressor } from '@sim/core';
import { LinearRegression } from './linear';

/**
 * Regression that resists outliers. Least squares squares the residuals, so one wild point can
 * dominate; these models limit or replace that influence.
 *
 * - `HuberRegression`: quadratic for small residuals, linear for large ones (M-estimation, IRLS).
 * - `QuantileRegression`: minimises the pinball loss, so it fits a chosen quantile (0.5 = median).
 * - `theilSen`: median of all pairwise slopes (one feature).
 */

/** Weighted least squares through `LinearRegression` on rows rescaled by √w. */
function weightedFit(
  x: Matrix,
  y: ArrayLike<number>,
  w: ArrayLike<number>,
): { intercept: number; coef: Float64Array } {
  // Minimising Σ wᵢ (yᵢ − b − xᵢ·β)² is OLS on [√wᵢ, √wᵢ·xᵢ] without a separate intercept column.
  const d = x.cols;
  const design = new Matrix(x.rows, d + 1);
  const target = new Float64Array(x.rows);
  for (let i = 0; i < x.rows; i++) {
    const s = Math.sqrt(w[i] as number);
    design.set(i, 0, s);
    for (let j = 0; j < d; j++) design.set(i, j + 1, s * x.get(i, j));
    target[i] = s * (y[i] as number);
  }
  const fit = new LinearRegression({ intercept: false }).fit(design, target);
  return { intercept: fit.coef[0] as number, coef: fit.coef.slice(1) };
}

export interface RobustOptions {
  maxIterations?: number;
  tolerance?: number;
}

/** Huber's M-estimator solved by iteratively reweighted least squares (statsmodels' `RLM(HuberT)`). */
export class HuberRegression implements Regressor {
  intercept = 0;
  coef: Float64Array = new Float64Array(0);
  /** Final robust scale of the residuals. */
  scale = 1;
  /** Weight each training point ended with (1 = fully trusted, small = treated as an outlier). */
  weights: Float64Array = new Float64Array(0);
  iterations = 0;

  /** `t` is the threshold in robust standard deviations (1.345 gives 95% efficiency under normal noise). */
  constructor(readonly options: RobustOptions & { t?: number } = {}) {}

  fit(x: Matrix, y: ArrayLike<number>): this {
    const { t = 1.345, maxIterations = 200, tolerance = 1e-10 } = this.options;
    const n = x.rows;
    let w = new Float64Array(n).fill(1);
    let model = weightedFit(x, y, w);
    let beta = [model.intercept, ...model.coef];
    let it = 0;
    for (; it < maxIterations; it++) {
      const pred = predictWith(beta, x);
      const resid = Float64Array.from(
        { length: n },
        (_, i) => (y[i] as number) - (pred[i] as number),
      );
      // MAD scale of the residuals about zero (statsmodels' choice), calibrated to the normal
      this.scale = median(resid.map((r) => Math.abs(r))) / 0.6744897501960817;
      if (this.scale < 1e-12) break;
      w = Float64Array.from(resid, (r) => {
        const u = Math.abs(r / this.scale);
        return u <= t ? 1 : t / u;
      });
      model = weightedFit(x, y, w);
      const next = [model.intercept, ...model.coef];
      const change = Math.max(...next.map((v, k) => Math.abs(v - (beta[k] as number))));
      beta = next;
      if (change < tolerance) {
        it++;
        break;
      }
    }
    this.iterations = it;
    this.weights = w;
    this.intercept = beta[0] as number;
    this.coef = Float64Array.from(beta.slice(1));
    return this;
  }

  predict(x: Matrix): Float64Array {
    return predictWith([this.intercept, ...this.coef], x);
  }
}

function predictWith(beta: readonly number[], x: Matrix): Float64Array {
  return Float64Array.from({ length: x.rows }, (_, i) => {
    let s = beta[0] as number;
    for (let j = 0; j < x.cols; j++) s += x.get(i, j) * (beta[j + 1] as number);
    return s;
  });
}

/** Mean pinball (quantile) loss of predictions `pred` for the quantile `q`. */
export function pinballLoss(y: ArrayLike<number>, pred: ArrayLike<number>, q: number): number {
  let s = 0;
  for (let i = 0; i < y.length; i++) {
    const e = (y[i] as number) - (pred[i] as number);
    s += e >= 0 ? q * e : (q - 1) * e;
  }
  return s / y.length;
}

/**
 * Linear quantile regression: the line below which a fraction `quantile` of the targets fall.
 * The pinball loss is piecewise linear; IRLS with weights 1/max(|r|, ε) (scaled by q or 1 − q)
 * converges to its minimiser, the same scheme statsmodels' `QuantReg` uses.
 */
export class QuantileRegression implements Regressor {
  intercept = 0;
  coef: Float64Array = new Float64Array(0);
  iterations = 0;

  constructor(readonly options: RobustOptions & { quantile?: number } = {}) {}

  fit(x: Matrix, y: ArrayLike<number>): this {
    const { quantile = 0.5, maxIterations = 2000, tolerance = 1e-9 } = this.options;
    const n = x.rows;
    let beta: number[] = [];
    let w = new Float64Array(n).fill(1);
    const eps = 1e-6;
    let it = 0;
    for (; it < maxIterations; it++) {
      const m = weightedFit(x, y, w);
      const next = [m.intercept, ...m.coef];
      const pred = predictWith(next, x);
      w = Float64Array.from({ length: n }, (_, i) => {
        const r = (y[i] as number) - (pred[i] as number);
        return (r >= 0 ? quantile : 1 - quantile) / Math.max(Math.abs(r), eps);
      });
      const change = beta.length
        ? Math.max(...next.map((v, k) => Math.abs(v - (beta[k] as number))))
        : Infinity;
      beta = next;
      if (change < tolerance) {
        it++;
        break;
      }
    }
    this.iterations = it;
    this.intercept = beta[0] as number;
    this.coef = Float64Array.from(beta.slice(1));
    return this;
  }

  predict(x: Matrix): Float64Array {
    return predictWith([this.intercept, ...this.coef], x);
  }
}

/** Theil–Sen estimator for one feature: the median pairwise slope and the median intercept. */
export class TheilSen implements Regressor {
  slope = 0;
  intercept = 0;

  fit(x: Matrix, y: ArrayLike<number>): this {
    if (x.cols !== 1) throw new RangeError('TheilSen supports one feature');
    const slopes: number[] = [];
    for (let i = 0; i < x.rows; i++)
      for (let j = i + 1; j < x.rows; j++) {
        const dx = x.get(j, 0) - x.get(i, 0);
        if (dx !== 0) slopes.push(((y[j] as number) - (y[i] as number)) / dx);
      }
    this.slope = median(slopes);
    this.intercept =
      median(Array.from(y)) -
      this.slope * median(Array.from({ length: x.rows }, (_, i) => x.get(i, 0)));
    return this;
  }

  get coef(): Float64Array {
    return Float64Array.of(this.slope);
  }

  predict(x: Matrix): Float64Array {
    return Float64Array.from(
      { length: x.rows },
      (_, i) => this.intercept + this.slope * x.get(i, 0),
    );
  }
}
