import { Matrix, type Clusterer } from '@sim/core';
import { pairwiseDistances } from './hierarchical';

/**
 * Density-based clustering. DBSCAN grows clusters from "core" points (those with at least
 * `minSamples` neighbours within `eps`), so clusters can have any shape and sparse points are
 * labelled noise (−1). OPTICS orders the points so that one pass shows the clusters at *every*
 * density (the reachability plot). Mean shift climbs the density estimate from every point to the
 * nearest peak. All three follow scikit-learn.
 */

// ---- DBSCAN ---------------------------------------------------------------------------------

export interface DbscanOptions {
  eps: number;
  /** Neighbours (counting the point itself) a core point needs. */
  minSamples: number;
}

export class Dbscan implements Clusterer {
  labels: Int32Array = new Int32Array(0);
  /** Indices of the core points. */
  coreIndices: number[] = [];

  constructor(readonly options: DbscanOptions) {}

  fit(x: Matrix): this {
    const n = x.rows;
    const d = pairwiseDistances(x);
    const { eps, minSamples } = this.options;
    const neighbours = Array.from({ length: n }, (_, i) => {
      const out: number[] = [];
      for (let j = 0; j < n; j++) if ((d[i * n + j] as number) <= eps) out.push(j);
      return out;
    });
    const core = neighbours.map((nb) => nb.length >= minSamples);
    this.coreIndices = core.map((c, i) => (c ? i : -1)).filter((i) => i >= 0);
    const labels = new Int32Array(n).fill(-1);
    let cluster = 0;
    for (let i = 0; i < n; i++) {
      if (labels[i] !== -1 || !core[i]) continue;
      // breadth-first growth from an unlabelled core point
      const stack = [i];
      labels[i] = cluster;
      while (stack.length) {
        const p = stack.pop() as number;
        if (!core[p]) continue;
        for (const q of neighbours[p] as number[]) {
          if (labels[q] === -1) {
            labels[q] = cluster;
            stack.push(q);
          }
        }
      }
      cluster++;
    }
    this.labels = labels;
    return this;
  }

  get clusters(): number {
    return this.labels.reduce((m, l) => Math.max(m, l + 1), 0);
  }

  get noise(): number {
    return this.labels.reduce((n, l) => n + (l === -1 ? 1 : 0), 0);
  }
}

/** Distance to the k-th nearest neighbour (counting the point itself as the first), sorted: the "k-distance plot" used to pick `eps`. */
export function kDistances(x: Matrix, k: number): number[] {
  const n = x.rows;
  const d = pairwiseDistances(x);
  return Array.from({ length: n }, (_, i) => {
    const row = Array.from({ length: n }, (_, j) => d[i * n + j] as number).sort((a, b) => a - b);
    return row[Math.min(k, n) - 1] as number;
  }).sort((a, b) => a - b);
}

// ---- OPTICS ---------------------------------------------------------------------------------

export interface OpticsResult {
  /** Visiting order of the points. */
  ordering: number[];
  /** Reachability distance of each point (Infinity for the first of every density-connected group). */
  reachability: Float64Array;
  coreDistances: Float64Array;
  predecessor: Int32Array;
}

export function optics(x: Matrix, minSamples: number, maxEps = Infinity): OpticsResult {
  const n = x.rows;
  const d = pairwiseDistances(x);
  const coreDistances = new Float64Array(n).fill(Infinity);
  for (let i = 0; i < n; i++) {
    const row = Array.from({ length: n }, (_, j) => d[i * n + j] as number).sort((a, b) => a - b);
    const v = row[Math.min(minSamples, n) - 1] as number;
    coreDistances[i] = v <= maxEps ? v : Infinity;
  }
  const reachability = new Float64Array(n).fill(Infinity);
  const predecessor = new Int32Array(n).fill(-1);
  const processed = new Uint8Array(n);
  const ordering: number[] = [];
  for (let start = 0; start < n; start++) {
    if (processed[start]) continue;
    let point = start;
    for (;;) {
      processed[point] = 1;
      ordering.push(point);
      if (Number.isFinite(coreDistances[point])) {
        for (let j = 0; j < n; j++) {
          if (processed[j]) continue;
          const dist = d[point * n + j] as number;
          if (dist > maxEps) continue;
          const reach = Math.max(coreDistances[point] as number, dist);
          if (reach < (reachability[j] as number)) {
            reachability[j] = reach;
            predecessor[j] = point;
          }
        }
      }
      // next: the unprocessed point with the smallest reachability (the first one if all are infinite)
      let next = -1;
      let best = Infinity;
      for (let j = 0; j < n; j++) {
        if (processed[j]) continue;
        if (next === -1 || (reachability[j] as number) < best) {
          next = j;
          best = reachability[j] as number;
        }
      }
      if (next === -1 || !Number.isFinite(best)) break;
      point = next;
    }
  }
  return { ordering, reachability, coreDistances, predecessor };
}

/** Flat clusters from an OPTICS ordering as DBSCAN would find them at `eps` (scikit-learn's `cluster_optics_dbscan`). */
export function opticsClusters(result: OpticsResult, eps: number): Int32Array {
  const n = result.ordering.length;
  const labels = new Int32Array(n);
  let count = 0;
  const far = (i: number) => (result.reachability[i] as number) > eps;
  const near = (i: number) => (result.coreDistances[i] as number) <= eps;
  for (const i of result.ordering) {
    if (far(i) && near(i)) count++;
    labels[i] = count - 1;
  }
  for (let i = 0; i < n; i++) if (far(i) && !near(i)) labels[i] = -1;
  return labels;
}

// ---- mean shift -----------------------------------------------------------------------------

export interface MeanShiftOptions {
  bandwidth: number;
  maxIterations?: number;
}

export class MeanShift implements Clusterer {
  labels: Int32Array = new Int32Array(0);
  centers: Matrix = new Matrix(0, 0);
  /** Path each seed took (first and last few positions) for drawing arrows. */
  paths: [number, number][][] = [];

  constructor(readonly options: MeanShiftOptions) {}

  /**
   * The trajectory of one seed under repeated shifts (flat kernel), for animation. `count` is the
   * number of points inside the last window that was averaged (scikit-learn's "intensity").
   */
  static climb(
    x: Matrix,
    start: ArrayLike<number>,
    bandwidth: number,
    maxIterations = 300,
  ): { path: number[][]; count: number } {
    const path: number[][] = [Array.from(start)];
    let c = Array.from(start);
    let count = 0;
    for (let it = 0; it < maxIterations; it++) {
      const next = new Array<number>(x.cols).fill(0);
      count = 0;
      for (let i = 0; i < x.rows; i++) {
        let s = 0;
        for (let k = 0; k < x.cols; k++) s += (x.get(i, k) - (c[k] as number)) ** 2;
        if (Math.sqrt(s) <= bandwidth) {
          count++;
          for (let k = 0; k < x.cols; k++) next[k] = (next[k] as number) + x.get(i, k);
        }
      }
      if (count === 0) break;
      const moved = next.map((v) => v / count);
      const shift = Math.hypot(...moved.map((v, k) => v - (c[k] as number)));
      c = moved;
      path.push(c);
      if (shift <= 1e-3 * bandwidth) break;
    }
    return { path, count };
  }

  fit(x: Matrix): this {
    const { bandwidth } = this.options;
    const maxIterations = this.options.maxIterations ?? 300;
    const n = x.rows;
    // every point is a seed; remember where each one ends up and how many points it saw there
    const peaks = Array.from({ length: n }, (_, i) => {
      const { path, count } = MeanShift.climb(
        x,
        x.data.subarray(i * x.cols, (i + 1) * x.cols),
        bandwidth,
        maxIterations,
      );
      return { end: path[path.length - 1] as number[], count, path };
    });
    this.paths = peaks.map((p) => p.path.map((q) => [q[0] as number, q[1] as number]));
    // merge near-duplicate peaks, keeping the busiest (ties: the larger coordinates first)
    const lexicographic = (a: number[], b: number[]) => {
      for (let k = 0; k < a.length; k++)
        if (a[k] !== b[k]) return (b[k] as number) - (a[k] as number);
      return 0;
    };
    const ordered = [...peaks].sort((a, b) => b.count - a.count || lexicographic(a.end, b.end));
    const kept: number[][] = [];
    for (const p of ordered)
      if (kept.every((c) => Math.hypot(...c.map((v, k) => v - (p.end[k] as number))) > bandwidth))
        kept.push(p.end);
    this.centers = Matrix.from(kept);
    this.labels = Int32Array.from({ length: n }, (_, i) => {
      let best = 0;
      let bestD = Infinity;
      kept.forEach((c, j) => {
        let s = 0;
        for (let k = 0; k < x.cols; k++) s += (x.get(i, k) - (c[k] as number)) ** 2;
        if (s < bestD) {
          bestD = s;
          best = j;
        }
      });
      return best;
    });
    return this;
  }

  get clusters(): number {
    return this.centers.rows;
  }
}

/** A rule-of-thumb bandwidth: the mean distance to the `quantile` share of nearest neighbours (scikit-learn's `estimate_bandwidth`). */
export function estimateBandwidth(x: Matrix, quantile = 0.3): number {
  const n = x.rows;
  const d = pairwiseDistances(x);
  const k = Math.max(1, Math.floor(n * quantile));
  let total = 0;
  for (let i = 0; i < n; i++) {
    const row = Array.from({ length: n }, (_, j) => d[i * n + j] as number).sort((a, b) => a - b);
    total += row[Math.min(k, n) - 1] as number;
  }
  return total / n;
}
