import { Matrix, cholesky, choleskySolve, createRng, type Rng } from '@sim/core';
import { pairwiseDistances } from './hierarchical';

/**
 * Anomaly detection: score every point by how unusual it is. Three ideas: *density* (Local Outlier
 * Factor: is it in a much sparser spot than its neighbours?), *isolation* (isolation forest: how few
 * random cuts does it take to cut the point off from the rest?) and *shape* (Mahalanobis distance:
 * how far is it, in standard deviations, once the correlations of the data are accounted for?).
 * Higher score = more anomalous in every case.
 */

// ---- Local outlier factor ---------------------------------------------------------------------

export interface LofResult {
  /** LOF per point: about 1 = like its neighbours, clearly above 1 = sparser than them. */
  scores: Float64Array;
  /** Local reachability density of every point. */
  density: Float64Array;
  kDistance: Float64Array;
}

/** Local outlier factor, following scikit-learn's `LocalOutlierFactor` (training scores, `novelty=False`). */
export function localOutlierFactor(x: Matrix, neighbors: number): LofResult {
  const n = x.rows;
  const k = Math.max(1, Math.min(neighbors, n - 1));
  const d = pairwiseDistances(x);
  const near: number[][] = [];
  const kDistance = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    const order = Array.from({ length: n }, (_, j) => j)
      .filter((j) => j !== i)
      .sort((a, b) => (d[i * n + a] as number) - (d[i * n + b] as number) || a - b)
      .slice(0, k);
    near.push(order);
    kDistance[i] = d[i * n + (order[k - 1] as number)] as number;
  }
  const density = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    let s = 0;
    for (const j of near[i] as number[])
      s += Math.max(kDistance[j] as number, d[i * n + j] as number);
    density[i] = 1 / (s / k + 1e-10);
  }
  const scores = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    let s = 0;
    for (const j of near[i] as number[]) s += (density[j] as number) / (density[i] as number);
    scores[i] = s / k;
  }
  return { scores, density, kDistance };
}

// ---- isolation forest --------------------------------------------------------------------------

/** Average path length of an unsuccessful search in a binary search tree of `n` points (the normaliser). */
export function averagePathLength(n: number): number {
  if (n <= 1) return 0;
  if (n === 2) return 1;
  return 2 * (Math.log(n - 1) + 0.5772156649015329) - (2 * (n - 1)) / n;
}

interface IsoNode {
  feature: number;
  split: number;
  left: IsoNode | null;
  right: IsoNode | null;
  size: number;
}

export interface IsolationOptions {
  trees?: number;
  /** Rows sampled for each tree (default min(256, n)). */
  subsample?: number;
  seed?: number;
}

export class IsolationForest {
  private trees: IsoNode[] = [];
  private psi = 1;

  constructor(readonly options: IsolationOptions = {}) {}

  fit(x: Matrix): this {
    const { trees = 100, seed = 1 } = this.options;
    const rng = createRng(seed);
    this.psi = Math.min(this.options.subsample ?? 256, x.rows);
    const limit = Math.ceil(Math.log2(Math.max(this.psi, 2)));
    this.trees = Array.from({ length: trees }, () => {
      const idx = sampleWithoutReplacement(x.rows, this.psi, rng);
      return this.grow(x, idx, 0, limit, rng);
    });
    return this;
  }

  private grow(x: Matrix, idx: number[], depth: number, limit: number, rng: Rng): IsoNode {
    if (depth >= limit || idx.length <= 1)
      return { feature: -1, split: 0, left: null, right: null, size: idx.length };
    const feature = Math.floor(rng.next() * x.cols);
    let lo = Infinity;
    let hi = -Infinity;
    for (const i of idx) {
      lo = Math.min(lo, x.get(i, feature));
      hi = Math.max(hi, x.get(i, feature));
    }
    if (hi <= lo) return { feature: -1, split: 0, left: null, right: null, size: idx.length };
    const split = lo + rng.next() * (hi - lo);
    return {
      feature,
      split,
      size: idx.length,
      left: this.grow(
        x,
        idx.filter((i) => x.get(i, feature) < split),
        depth + 1,
        limit,
        rng,
      ),
      right: this.grow(
        x,
        idx.filter((i) => x.get(i, feature) >= split),
        depth + 1,
        limit,
        rng,
      ),
    };
  }

  /** Path length of row `i` of `x` in one tree, with the usual correction for leaves that still hold several points. */
  private pathLength(tree: IsoNode, x: Matrix, i: number): number {
    let node = tree;
    let depth = 0;
    while (node.left && node.right) {
      node = x.get(i, node.feature) < node.split ? node.left : node.right;
      depth++;
    }
    return depth + averagePathLength(node.size);
  }

  /** Anomaly score in (0, 1): near 1 = isolated quickly, below 0.5 = deep in the crowd. */
  scores(x: Matrix): Float64Array {
    const c = averagePathLength(this.psi);
    return Float64Array.from({ length: x.rows }, (_, i) => {
      let mean = 0;
      for (const t of this.trees) mean += this.pathLength(t, x, i);
      mean /= this.trees.length;
      return 2 ** (-mean / (c || 1));
    });
  }
}

function sampleWithoutReplacement(n: number, k: number, rng: Rng): number[] {
  const all = Array.from({ length: n }, (_, i) => i);
  for (let i = n - 1; i > n - 1 - k; i--) {
    const j = Math.floor(rng.next() * (i + 1));
    [all[i], all[j]] = [all[j] as number, all[i] as number];
  }
  return all.slice(n - k);
}

// ---- Mahalanobis ---------------------------------------------------------------------------------

export class MahalanobisDetector {
  mean: Float64Array = new Float64Array(0);
  covariance: Matrix = new Matrix(0, 0);
  private factor: Matrix = new Matrix(0, 0);

  fit(x: Matrix): this {
    const n = x.rows;
    const d = x.cols;
    this.mean = new Float64Array(d);
    for (let j = 0; j < d; j++) {
      for (let i = 0; i < n; i++) this.mean[j] = (this.mean[j] as number) + x.get(i, j) / n;
    }
    this.covariance = new Matrix(d, d);
    for (let a = 0; a < d; a++)
      for (let b = a; b < d; b++) {
        let s = 0;
        for (let i = 0; i < n; i++)
          s += (x.get(i, a) - (this.mean[a] as number)) * (x.get(i, b) - (this.mean[b] as number));
        this.covariance.set(a, b, s / (n - 1));
        this.covariance.set(b, a, s / (n - 1));
      }
    const reg = new Matrix(d, d, Float64Array.from(this.covariance.data));
    for (let j = 0; j < d; j++) reg.set(j, j, reg.get(j, j) + 1e-9);
    this.factor = cholesky(reg);
    return this;
  }

  /** Mahalanobis distance of every row from the centre of the data, in "standard deviations". */
  scores(x: Matrix): Float64Array {
    const d = x.cols;
    return Float64Array.from({ length: x.rows }, (_, i) => {
      const diff = new Matrix(
        d,
        1,
        Float64Array.from({ length: d }, (_, j) => x.get(i, j) - (this.mean[j] as number)),
      );
      const sol = choleskySolve(this.factor, diff);
      let q = 0;
      for (let j = 0; j < d; j++) q += (diff.data[j] as number) * (sol.data[j] as number);
      return Math.sqrt(Math.max(q, 0));
    });
  }
}

// ---- evaluating a detector -------------------------------------------------------------------------

/** Area under the ROC curve of scores against binary anomaly labels (1 = anomaly): the chance a random anomaly outscores a random normal point. */
export function rocAuc(scores: ArrayLike<number>, anomalous: ArrayLike<number>): number {
  const idx = Array.from({ length: scores.length }, (_, i) => i).sort(
    (a, b) => (scores[a] as number) - (scores[b] as number),
  );
  let rankSum = 0;
  let pos = 0;
  // average ranks for ties
  let i = 0;
  while (i < idx.length) {
    let j = i;
    while (j + 1 < idx.length && scores[idx[j + 1] as number] === scores[idx[i] as number]) j++;
    const rank = (i + j) / 2 + 1;
    for (let t = i; t <= j; t++)
      if (anomalous[idx[t] as number] === 1) {
        rankSum += rank;
        pos++;
      }
    i = j + 1;
  }
  const neg = scores.length - pos;
  return pos === 0 || neg === 0 ? NaN : (rankSum - (pos * (pos + 1)) / 2) / (pos * neg);
}

/** The score above which a fraction `contamination` of the points lie. */
export function thresholdFor(scores: ArrayLike<number>, contamination: number): number {
  const sorted = Array.from(scores).sort((a, b) => b - a);
  const k = Math.max(1, Math.round(contamination * sorted.length));
  return sorted[Math.min(k, sorted.length) - 1] as number;
}
