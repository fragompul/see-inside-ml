import { Matrix, createRng, type Classifier, type Regressor } from '@sim/core';

/**
 * k-nearest neighbours: no training, the data *is* the model. A query is answered by looking at
 * its `k` closest training points (brute force: the labs have at most a few hundred points).
 */
export type Metric = 'euclidean' | 'manhattan' | 'chebyshev';
export type NeighborWeights = 'uniform' | 'distance';

export interface KnnOptions {
  k?: number;
  metric?: Metric;
  weights?: NeighborWeights;
}

export function distance(a: ArrayLike<number>, b: ArrayLike<number>, metric: Metric): number {
  let s = 0;
  for (let j = 0; j < a.length; j++) {
    const d = Math.abs((a[j] as number) - (b[j] as number));
    if (metric === 'euclidean') s += d * d;
    else if (metric === 'manhattan') s += d;
    else s = Math.max(s, d);
  }
  return metric === 'euclidean' ? Math.sqrt(s) : s;
}

abstract class Neighbors {
  protected train: Matrix = new Matrix(0, 0);
  protected target: Float64Array = new Float64Array(0);

  constructor(readonly options: KnnOptions = {}) {}

  protected get k(): number {
    return this.options.k ?? 5;
  }

  /** Indices and distances of the `k` training points closest to row `i` of `x`, nearest first. */
  neighbors(
    x: Matrix,
    i: number,
    k = this.k,
    skipSelf = false,
  ): { index: number; distance: number }[] {
    const metric = this.options.metric ?? 'euclidean';
    const q = x.data.subarray(i * x.cols, (i + 1) * x.cols);
    const all: { index: number; distance: number }[] = [];
    for (let t = 0; t < this.train.rows; t++) {
      if (skipSelf && t === i) continue;
      all.push({
        index: t,
        distance: distance(
          q,
          this.train.data.subarray(t * this.train.cols, (t + 1) * this.train.cols),
          metric,
        ),
      });
    }
    // stable on ties: earlier training rows win
    all.sort((a, b) => a.distance - b.distance || a.index - b.index);
    return all.slice(0, Math.min(k, all.length));
  }

  /** Weight of each neighbour in the vote or the average. */
  protected weightsOf(near: { distance: number }[]): number[] {
    if ((this.options.weights ?? 'uniform') === 'uniform') return near.map(() => 1);
    const zero = near.some((n) => n.distance === 0);
    // a query sitting on training points listens only to them
    return near.map((n) => (zero ? (n.distance === 0 ? 1 : 0) : 1 / n.distance));
  }
}

export class KnnClassifier extends Neighbors implements Classifier {
  classes = 2;

  fit(x: Matrix, y: ArrayLike<number>): this {
    this.train = x;
    this.target = Float64Array.from(y);
    let k = 0;
    for (const v of this.target) k = Math.max(k, v + 1);
    this.classes = Math.max(2, k);
    return this;
  }

  predictProba(x: Matrix): Matrix {
    const out = new Matrix(x.rows, this.classes);
    for (let i = 0; i < x.rows; i++) {
      const near = this.neighbors(x, i);
      const w = this.weightsOf(near);
      const total = w.reduce((s, v) => s + v, 0);
      near.forEach((n, r) => {
        const c = this.target[n.index] as number;
        out.set(i, c, out.get(i, c) + (w[r] as number) / total);
      });
    }
    return out;
  }

  predict(x: Matrix): Int32Array {
    const p = this.predictProba(x);
    return Int32Array.from({ length: x.rows }, (_, i) => {
      let best = 0;
      for (let c = 1; c < this.classes; c++) if (p.get(i, c) > p.get(i, best)) best = c;
      return best;
    });
  }
}

export class KnnRegressor extends Neighbors implements Regressor {
  fit(x: Matrix, y: ArrayLike<number>): this {
    this.train = x;
    this.target = Float64Array.from(y);
    return this;
  }

  predict(x: Matrix): Float64Array {
    return Float64Array.from({ length: x.rows }, (_, i) => {
      const near = this.neighbors(x, i);
      const w = this.weightsOf(near);
      const total = w.reduce((s, v) => s + v, 0);
      return near.reduce(
        (s, n, r) => s + ((w[r] as number) * (this.target[n.index] as number)) / total,
        0,
      );
    });
  }
}

/**
 * Leave-one-out accuracy of k-NN for k = 1 … `maxK`: every training point is classified by its
 * neighbours *excluding itself*, which is the honest estimate (with itself included k = 1 is 100%).
 */
export function looAccuracyByK(
  x: Matrix,
  y: ArrayLike<number>,
  maxK: number,
  options: Omit<KnnOptions, 'k'> = {},
): number[] {
  const model = new KnnClassifier({ ...options, k: maxK }).fit(x, y);
  const out = new Array<number>(maxK).fill(0);
  for (let i = 0; i < x.rows; i++) {
    const near = model.neighbors(x, i, maxK, true);
    for (let k = 1; k <= maxK; k++) {
      const first = near.slice(0, k);
      const w =
        options.weights === 'distance'
          ? first.map((n) => 1 / Math.max(n.distance, 1e-12))
          : first.map(() => 1);
      const votes = new Map<number, number>();
      first.forEach((n, r) =>
        votes.set(y[n.index] as number, (votes.get(y[n.index] as number) ?? 0) + (w[r] as number)),
      );
      let best = -1;
      let bestVotes = -Infinity;
      for (const [c, v] of [...votes].sort((a, b) => a[0] - b[0])) {
        if (v > bestVotes) {
          best = c;
          bestVotes = v;
        }
      }
      if (best === y[i]) out[k - 1] = (out[k - 1] as number) + 1;
    }
  }
  return out.map((v) => v / x.rows);
}

/**
 * The curse of dimensionality for neighbours: in `d` dimensions, how close is the nearest of `n`
 * uniform random points compared with the farthest? Returns, per dimension, the mean ratio
 * nearest / farthest over random queries. It creeps towards 1: everything is "equally far".
 */
export function distanceConcentration(
  dims: readonly number[],
  options: { n?: number; queries?: number; seed?: number } = {},
): number[] {
  const { n = 200, queries = 20, seed = 1 } = options;
  const rng = createRng(seed);
  return dims.map((d) => {
    const points = Array.from({ length: n }, () =>
      Float64Array.from({ length: d }, () => rng.next()),
    );
    let total = 0;
    for (let q = 0; q < queries; q++) {
      const query = Float64Array.from({ length: d }, () => rng.next());
      let near = Infinity;
      let far = 0;
      for (const p of points) {
        const dist = distance(query, p, 'euclidean');
        near = Math.min(near, dist);
        far = Math.max(far, dist);
      }
      total += near / far;
    }
    return total / queries;
  });
}
