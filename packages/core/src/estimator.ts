import { Matrix } from './matrix';

/**
 * The shapes every classical model implements. A model is one file; the labs, the generic tests
 * and the "model showdown" only talk to these interfaces.
 */
export interface Classifier {
  /** Number of classes (labels are the integers 0 … classes − 1). */
  readonly classes: number;
  fit(x: Matrix, y: ArrayLike<number>): this;
  predict(x: Matrix): Int32Array;
  /** Class probabilities `n × classes`, for models that have them. */
  predictProba?(x: Matrix): Matrix;
}

export interface Regressor {
  fit(x: Matrix, y: ArrayLike<number>): this;
  predict(x: Matrix): Float64Array;
}

export interface Clusterer {
  fit(x: Matrix): this;
  /** Cluster of every training point (−1 = noise, for density methods). */
  readonly labels: Int32Array;
}

export interface Projector {
  fit(x: Matrix): this;
  transform(x: Matrix): Matrix;
}

/** An algorithm that can be watched step by step (tree growth, boosting rounds, EM iterations…). */
export interface Iterative<S> {
  states(): Generator<S, void, undefined>;
}

/** Row-major `n × 2` points covering `[x0, x1] × [y0, y1]` on an `n × n` grid, from the top row down. */
export function gridPoints(x0: number, x1: number, y0: number, y1: number, n: number): Matrix {
  const g = new Matrix(n * n, 2);
  for (let r = 0; r < n; r++)
    for (let c = 0; c < n; c++) {
      g.data[(r * n + c) * 2] = x0 + ((c + 0.5) / n) * (x1 - x0);
      g.data[(r * n + c) * 2 + 1] = y1 - ((r + 0.5) / n) * (y1 - y0);
    }
  return g;
}
