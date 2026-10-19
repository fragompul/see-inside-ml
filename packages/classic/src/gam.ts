import { Matrix, matMul, solve, transpose, type Regressor } from '@sim/core';

/**
 * Smoothers for generalised additive models. A P-spline (Eilers and Marx) writes a smooth function of one input as a
 * combination of cubic B-spline bumps on equally spaced knots, and keeps it from wiggling by penalising the second
 * differences of the coefficients: minimise |y - B c|^2 + lambda |D c|^2. lambda = 0 is an ordinary regression on the
 * bumps; a huge lambda leaves only what the penalty cannot see, a straight line. An additive model is a sum of such
 * smooth functions, one per input, fitted by backfitting: smooth the residual left by all the others, in turn, until
 * nothing changes.
 */
export const SPLINE_DEGREE = 3;

/** Cubic B-spline basis on `count` bumps covering [lo, hi]: returns the n x count matrix of the bumps at each x. */
export function bsplineBasis(x: ArrayLike<number>, lo: number, hi: number, count: number): Matrix {
  const degree = SPLINE_DEGREE;
  if (count <= degree) throw new RangeError('Need more bumps than the degree of the spline');
  const inner = count - degree; // intervals between the knots that cover [lo, hi]
  const h = (hi - lo) / inner;
  const knots = Array.from({ length: count + degree + 1 }, (_, i) => lo + (i - degree) * h);
  const out = new Matrix(x.length, count);
  for (let r = 0; r < x.length; r++) {
    const v = Math.min(hi - 1e-12, Math.max(lo, x[r] as number));
    // Cox-de Boor recursion for the degree + 1 bumps that are not zero at v
    let span = Math.floor((v - lo) / h) + degree;
    span = Math.min(span, count - 1);
    const left = new Float64Array(degree + 1);
    const right = new Float64Array(degree + 1);
    const nz = new Float64Array(degree + 1);
    nz[0] = 1;
    for (let j = 1; j <= degree; j++) {
      left[j] = v - (knots[span + 1 - j] as number);
      right[j] = (knots[span + j] as number) - v;
      let saved = 0;
      for (let k = 0; k < j; k++) {
        const t = (nz[k] as number) / ((right[k + 1] as number) + (left[j - k] as number));
        nz[k] = saved + (right[k + 1] as number) * t;
        saved = (left[j - k] as number) * t;
      }
      nz[j] = saved;
    }
    for (let k = 0; k <= degree; k++) out.data[r * count + span - degree + k] = nz[k] as number;
  }
  return out;
}

/** Second-difference penalty matrix D^T D for `count` coefficients. */
export function differencePenalty(count: number): Matrix {
  const p = new Matrix(count, count);
  for (let i = 0; i + 2 < count; i++) {
    const row = [1, -2, 1];
    for (let a = 0; a < 3; a++)
      for (let b = 0; b < 3; b++)
        p.data[(i + a) * count + i + b] =
          (p.data[(i + a) * count + i + b] as number) + (row[a] as number) * (row[b] as number);
  }
  return p;
}

export interface PSplineOptions {
  bumps: number;
  lambda: number;
}

export class PSpline implements Regressor {
  coef: Float64Array = new Float64Array(0);
  lo = 0;
  hi = 1;
  /** Effective degrees of freedom: trace of the smoother matrix. */
  df = 0;
  /** Generalised cross-validation score. */
  gcv = 0;

  constructor(readonly options: PSplineOptions = { bumps: 12, lambda: 1 }) {}

  fit(x: Matrix, y: ArrayLike<number>): this {
    if (x.cols !== 1) throw new RangeError('A P-spline takes a single input');
    const xs = Array.from(x.data);
    this.lo = Math.min(...xs);
    this.hi = Math.max(...xs);
    if (this.hi === this.lo) this.hi = this.lo + 1;
    const b = bsplineBasis(xs, this.lo, this.hi, this.options.bumps);
    const bt = transpose(b);
    const a = matMul(bt, b);
    const pen = differencePenalty(this.options.bumps);
    for (let i = 0; i < a.data.length; i++)
      a.data[i] =
        (a.data[i] as number) +
        this.options.lambda * (pen.data[i] as number) +
        (i % (this.options.bumps + 1) === 0 ? 1e-8 : 0);
    const rhs = matMul(bt, new Matrix(x.rows, 1, Float64Array.from(y)));
    this.coef = solve(a, rhs).data;
    // trace of the smoother matrix B (B'B + lambda P)^-1 B' = trace((B'B + lambda P)^-1 B'B)
    const inv = solve(a, matMul(bt, b));
    this.df = 0;
    for (let i = 0; i < this.options.bumps; i++)
      this.df += inv.data[i * this.options.bumps + i] as number;
    const fit = this.predict(x);
    let sse = 0;
    for (let i = 0; i < x.rows; i++) sse += ((y[i] as number) - (fit[i] as number)) ** 2;
    this.gcv = (x.rows * sse) / (x.rows - this.df) ** 2;
    return this;
  }

  predict(x: Matrix): Float64Array {
    const b = bsplineBasis(Array.from(x.data), this.lo, this.hi, this.options.bumps);
    return Float64Array.from({ length: x.rows }, (_, i) => {
      let s = 0;
      for (let k = 0; k < this.options.bumps; k++)
        s += (b.data[i * this.options.bumps + k] as number) * (this.coef[k] as number);
      return s;
    });
  }
}

export interface AdditiveOptions {
  bumps: number;
  lambda: number;
  iterations: number;
}

/** y = intercept + f_1(x_1) + ... + f_d(x_d), each f_j a centred P-spline, fitted by backfitting. */
export class AdditiveModel implements Regressor {
  intercept = 0;
  smoothers: PSpline[] = [];
  /** Largest change in any fitted shape between the last two sweeps. */
  change = Infinity;
  sweeps = 0;

  constructor(readonly options: AdditiveOptions = { bumps: 10, lambda: 3, iterations: 40 }) {}

  fit(x: Matrix, y: ArrayLike<number>): this {
    const n = x.rows;
    const d = x.cols;
    this.intercept = Array.from(y).reduce((a, v) => a + v, 0) / n;
    const shapes = Array.from({ length: d }, () => new Float64Array(n));
    this.smoothers = Array.from(
      { length: d },
      () => new PSpline({ bumps: this.options.bumps, lambda: this.options.lambda }),
    );
    this.change = Infinity;
    for (
      this.sweeps = 0;
      this.sweeps < this.options.iterations && this.change > 1e-8;
      this.sweeps++
    ) {
      let worst = 0;
      for (let j = 0; j < d; j++) {
        const partial = Float64Array.from({ length: n }, (_, i) => {
          let r = (y[i] as number) - this.intercept;
          for (let k = 0; k < d; k++) if (k !== j) r -= (shapes[k] as Float64Array)[i] as number;
          return r;
        });
        const column = new Matrix(
          n,
          1,
          Float64Array.from({ length: n }, (_, i) => x.data[i * d + j] as number),
        );
        const smoother = (this.smoothers[j] as PSpline).fit(column, partial);
        const fitted = smoother.predict(column);
        const mean = fitted.reduce((a, v) => a + v, 0) / n;
        const shape = shapes[j] as Float64Array;
        for (let i = 0; i < n; i++) {
          const next = (fitted[i] as number) - mean;
          worst = Math.max(worst, Math.abs(next - (shape[i] as number)));
          shape[i] = next;
        }
        this.centre[j] = mean;
      }
      this.change = worst;
    }
    return this;
  }

  /** The mean each fitted smooth had before it was centred, per input. */
  private centre: number[] = [];

  /** The shape function of input j at the given values (centred to average zero over the training data). */
  shape(j: number, values: ArrayLike<number>): Float64Array {
    const smoother = this.smoothers[j] as PSpline;
    const f = smoother.predict(new Matrix(values.length, 1, Float64Array.from(values)));
    return f.map((v) => v - (this.centre[j] as number));
  }

  predict(x: Matrix): Float64Array {
    const out = new Float64Array(x.rows).fill(this.intercept);
    for (let j = 0; j < x.cols; j++) {
      const column = Float64Array.from(
        { length: x.rows },
        (_, i) => x.data[i * x.cols + j] as number,
      );
      const f = this.shape(j, column);
      for (let i = 0; i < x.rows; i++) out[i] = (out[i] as number) + (f[i] as number);
    }
    return out;
  }
}
