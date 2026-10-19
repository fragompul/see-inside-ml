import { Matrix, createRng } from '@sim/core';

/**
 * Model-agnostic importance tools. They only need a function that predicts, so they work for any
 * model in the atlas.
 *   - Permutation importance: shuffle one column and see how much the score drops.
 *   - Individual conditional expectation (ICE): sweep one feature across a grid for one row at a time.
 *   - Partial dependence (PDP): the average of the ICE curves.
 */
export type Predictor = (x: Matrix) => ArrayLike<number>;

/** Score where higher is better (accuracy, R², negative error...). */
export type Scorer = (yTrue: ArrayLike<number>, yPred: ArrayLike<number>) => number;

export interface PermutationResult {
  /** Average drop of the score when the column is shuffled. */
  importance: number[];
  /** Standard deviation of that drop over the repeats. */
  spread: number[];
  baseline: number;
}

export function permutationImportance(
  predict: Predictor,
  x: Matrix,
  y: ArrayLike<number>,
  score: Scorer,
  repeats = 10,
  seed = 1,
): PermutationResult {
  const rng = createRng(seed);
  const baseline = score(y, predict(x));
  const importance: number[] = [];
  const spread: number[] = [];
  for (let j = 0; j < x.cols; j++) {
    const drops: number[] = [];
    for (let r = 0; r < repeats; r++) {
      const shuffled = new Matrix(x.rows, x.cols, Float64Array.from(x.data));
      const order = Array.from({ length: x.rows }, (_, i) => i);
      for (let i = order.length - 1; i > 0; i--) {
        const k = Math.floor(rng.next() * (i + 1));
        [order[i], order[k]] = [order[k] as number, order[i] as number];
      }
      for (let i = 0; i < x.rows; i++) shuffled.set(i, j, x.get(order[i] as number, j));
      drops.push(baseline - score(y, predict(shuffled)));
    }
    const m = drops.reduce((a, b) => a + b, 0) / repeats;
    importance.push(m);
    spread.push(Math.sqrt(drops.reduce((a, b) => a + (b - m) ** 2, 0) / Math.max(repeats - 1, 1)));
  }
  return { importance, spread, baseline };
}

export interface IceResult {
  grid: number[];
  /** `curves[i][g]`: the prediction for row i when feature j is set to `grid[g]`. */
  curves: number[][];
  /** Average over rows: the partial dependence. */
  average: number[];
}

/** ICE curves and their average for feature `j`, over an evenly spaced grid across its observed range. */
export function iceCurves(
  predict: Predictor,
  x: Matrix,
  j: number,
  gridSize = 20,
  maxRows = 200,
): IceResult {
  let lo = Infinity;
  let hi = -Infinity;
  for (let i = 0; i < x.rows; i++) {
    lo = Math.min(lo, x.get(i, j));
    hi = Math.max(hi, x.get(i, j));
  }
  const grid = Array.from(
    { length: gridSize },
    (_, g) => lo + ((hi - lo) * g) / Math.max(gridSize - 1, 1),
  );
  const rows = Math.min(maxRows, x.rows);
  const stride = x.rows / rows;
  const picked = Array.from({ length: rows }, (_, r) => Math.floor(r * stride));
  const curves: number[][] = picked.map(() => []);
  for (const value of grid) {
    const sweep = new Matrix(rows, x.cols);
    picked.forEach((src, r) => {
      sweep.data.set(x.data.subarray(src * x.cols, (src + 1) * x.cols), r * x.cols);
      sweep.set(r, j, value);
    });
    const pred = predict(sweep);
    for (let r = 0; r < rows; r++) (curves[r] as number[]).push(pred[r] as number);
  }
  const average = grid.map((_, g) => curves.reduce((s, c) => s + (c[g] as number), 0) / rows);
  return { grid, curves, average };
}
