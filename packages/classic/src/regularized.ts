import type { Matrix, Regressor } from '@sim/core';
import { LinearRegression } from './linear';

/**
 * Penalised least squares. The objective follows scikit-learn:
 *
 *   Ridge:        ‖y − Xw − b‖² + α‖w‖²
 *   ElasticNet:   (1/2n)‖y − Xw − b‖² + α·ρ‖w‖₁ + ½·α·(1 − ρ)‖w‖²      (ρ = l1Ratio; ρ = 1 is the lasso)
 *
 * The intercept is never penalised. The elastic net is solved by cyclic coordinate descent, which
 * visits one coefficient at a time and sets it to its exact minimiser (a soft-thresholded value).
 */

export interface ElasticNetOptions {
  alpha: number;
  /** Mix between the lasso (1) and ridge (0) penalties. */
  l1Ratio?: number;
  intercept?: boolean;
  maxIterations?: number;
  /** Stop when no coefficient moves more than `tolerance` times the largest. */
  tolerance?: number;
}

export const softThreshold = (z: number, t: number): number => (z > t ? z - t : z < -t ? z + t : 0);

interface Centered {
  xc: Float64Array[]; // columns
  yc: Float64Array;
  xMean: Float64Array;
  yMean: number;
}

function center(x: Matrix, y: ArrayLike<number>, intercept: boolean): Centered {
  const n = x.rows;
  const xMean = new Float64Array(x.cols);
  let yMean = 0;
  if (intercept) {
    for (let j = 0; j < x.cols; j++) {
      let s = 0;
      for (let i = 0; i < n; i++) s += x.get(i, j);
      xMean[j] = s / n;
    }
    for (let i = 0; i < n; i++) yMean += y[i] as number;
    yMean /= n;
  }
  const xc = Array.from({ length: x.cols }, (_, j) =>
    Float64Array.from({ length: n }, (_, i) => x.get(i, j) - (xMean[j] as number)),
  );
  const yc = Float64Array.from({ length: n }, (_, i) => (y[i] as number) - yMean);
  return { xc, yc, xMean, yMean };
}

/** Coordinate descent on centred data, starting from `start` (a warm start along a path). */
function coordinateDescent(
  c: Centered,
  alpha: number,
  l1Ratio: number,
  start: Float64Array,
  maxIterations: number,
  tolerance: number,
): { coef: Float64Array; iterations: number } {
  const n = c.yc.length;
  const d = c.xc.length;
  const w = Float64Array.from(start);
  const colNorm = c.xc.map((col) => col.reduce((s, v) => s + v * v, 0));
  const resid = Float64Array.from(c.yc);
  for (let j = 0; j < d; j++) {
    const wj = w[j] as number;
    if (wj !== 0)
      for (let i = 0; i < n; i++) resid[i] = (resid[i] as number) - wj * (c.xc[j]![i] as number);
  }
  const l1 = n * alpha * l1Ratio;
  const l2 = n * alpha * (1 - l1Ratio);
  let iterations = 0;
  for (; iterations < maxIterations; iterations++) {
    let maxChange = 0;
    let maxCoef = 0;
    for (let j = 0; j < d; j++) {
      const col = c.xc[j] as Float64Array;
      const old = w[j] as number;
      // correlation of the column with the partial residual (residual with feature j added back)
      let rho = 0;
      for (let i = 0; i < n; i++) rho += (col[i] as number) * (resid[i] as number);
      rho += old * (colNorm[j] as number);
      const denom = (colNorm[j] as number) + l2;
      const updated = denom > 0 ? softThreshold(rho, l1) / denom : 0;
      if (updated !== old) {
        const delta = updated - old;
        for (let i = 0; i < n; i++) resid[i] = (resid[i] as number) - delta * (col[i] as number);
        w[j] = updated;
      }
      maxChange = Math.max(maxChange, Math.abs(updated - old));
      maxCoef = Math.max(maxCoef, Math.abs(updated));
    }
    if (maxChange <= tolerance * Math.max(maxCoef, 1e-12) || maxChange < 1e-14) {
      iterations++;
      break;
    }
  }
  return { coef: w, iterations };
}

export class ElasticNet implements Regressor {
  intercept = 0;
  coef: Float64Array = new Float64Array(0);
  iterations = 0;

  constructor(readonly options: ElasticNetOptions) {}

  fit(x: Matrix, y: ArrayLike<number>): this {
    const {
      alpha,
      l1Ratio = 1,
      intercept = true,
      maxIterations = 50000,
      tolerance = 1e-12,
    } = this.options;
    const c = center(x, y, intercept);
    const { coef, iterations } = coordinateDescent(
      c,
      alpha,
      l1Ratio,
      new Float64Array(x.cols),
      maxIterations,
      tolerance,
    );
    this.coef = coef;
    this.iterations = iterations;
    this.intercept = intercept
      ? c.yMean - coef.reduce((s, w, j) => s + w * (c.xMean[j] as number), 0)
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

  /** Number of coefficients that are not exactly zero. */
  get active(): number {
    return this.coef.reduce((n, w) => n + (w !== 0 ? 1 : 0), 0);
  }
}

/** The lasso is the elastic net with the whole penalty on the L1 norm. */
export class Lasso extends ElasticNet {
  constructor(options: Omit<ElasticNetOptions, 'l1Ratio'>) {
    super({ ...options, l1Ratio: 1 });
  }
}

/** Ridge regression: the L2-penalised normal equations (see `LinearRegression`). */
export class Ridge extends LinearRegression {
  constructor(options: { alpha: number; intercept?: boolean }) {
    super({ l2: options.alpha, intercept: options.intercept !== false });
  }
}

/** The smallest α for which every lasso / elastic-net coefficient is zero. */
export function alphaMax(x: Matrix, y: ArrayLike<number>, l1Ratio = 1, intercept = true): number {
  const c = center(x, y, intercept);
  const n = c.yc.length;
  let best = 0;
  for (const col of c.xc) {
    let dot = 0;
    for (let i = 0; i < n; i++) dot += (col[i] as number) * (c.yc[i] as number);
    best = Math.max(best, Math.abs(dot));
  }
  return best / (n * Math.max(l1Ratio, 1e-12));
}

export interface PathResult {
  alphas: number[];
  /** `coefs[k][j]` is the coefficient of feature j at `alphas[k]`. */
  coefs: Float64Array[];
  intercepts: number[];
}

/** Coefficients along a decreasing sequence of penalties, each warm-started from the previous. */
export function regularizationPath(
  x: Matrix,
  y: ArrayLike<number>,
  options: {
    l1Ratio?: number;
    alphas?: number[];
    count?: number;
    ratio?: number;
    intercept?: boolean;
  } = {},
): PathResult {
  const { l1Ratio = 1, intercept = true } = options;
  const top = alphaMax(x, y, l1Ratio, intercept);
  const alphas =
    options.alphas ??
    Array.from({ length: options.count ?? 60 }, (_, k) => {
      const count = options.count ?? 60;
      const ratio = options.ratio ?? 1e-3;
      return top * ratio ** (k / (count - 1));
    });
  const c = center(x, y, intercept);
  let warm: Float64Array = new Float64Array(x.cols);
  const coefs: Float64Array[] = [];
  const intercepts: number[] = [];
  for (const alpha of alphas) {
    warm = coordinateDescent(c, alpha, l1Ratio, warm, 50000, 1e-12).coef;
    coefs.push(Float64Array.from(warm));
    intercepts.push(
      intercept ? c.yMean - warm.reduce((s, w, j) => s + w * (c.xMean[j] as number), 0) : 0,
    );
  }
  return { alphas, coefs, intercepts };
}

/** Ridge coefficients along a list of penalties (closed form at each one). */
export function ridgePath(x: Matrix, y: ArrayLike<number>, alphas: number[]): PathResult {
  const coefs: Float64Array[] = [];
  const intercepts: number[] = [];
  for (const alpha of alphas) {
    const m = new Ridge({ alpha }).fit(x, y);
    coefs.push(Float64Array.from(m.coef));
    intercepts.push(m.intercept);
  }
  return { alphas, coefs, intercepts };
}
