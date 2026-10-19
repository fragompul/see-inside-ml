import type { Matrix } from '@sim/core';

/**
 * Agglomerative (bottom-up) hierarchical clustering: start with every point alone and repeatedly
 * merge the two closest clusters. How "closest" is measured for clusters (the linkage) shapes the
 * result: single (nearest pair) chains, complete (farthest pair) makes compact blobs, average is in
 * between, Ward merges whichever pair increases the within-cluster variance least. The output is a
 * linkage matrix in SciPy's convention, from which a dendrogram and any flat clustering follow.
 */
export type Linkage = 'single' | 'complete' | 'average' | 'ward';

export interface Merge {
  /** Ids of the two clusters joined; ids 0 … n−1 are the points, n + i is the cluster made by merge i. */
  a: number;
  b: number;
  /** Distance at which they were joined. */
  height: number;
  /** Points in the new cluster. */
  size: number;
}

export function pairwiseDistances(x: Matrix): Float64Array {
  const n = x.rows;
  const d = new Float64Array(n * n);
  for (let i = 0; i < n; i++)
    for (let j = i + 1; j < n; j++) {
      let s = 0;
      for (let k = 0; k < x.cols; k++) s += (x.get(i, k) - x.get(j, k)) ** 2;
      d[i * n + j] = d[j * n + i] = Math.sqrt(s);
    }
  return d;
}

/** Linkage matrix of `x` (Euclidean distances), the merges in the order they happen. */
export function linkage(x: Matrix, method: Linkage): Merge[] {
  return linkageFromDistances(pairwiseDistances(x), x.rows, method);
}

export function linkageFromDistances(distances: Float64Array, n: number, method: Linkage): Merge[] {
  const d = Float64Array.from(distances);
  const size = new Float64Array(2 * n - 1).fill(1);
  const id = Array.from({ length: n }, (_, i) => i); // cluster id living in each active slot
  const active = new Array<boolean>(n).fill(true);
  const merges: Merge[] = [];
  for (let step = 0; step < n - 1; step++) {
    let bi = -1;
    let bj = -1;
    let best = Infinity;
    for (let i = 0; i < n; i++) {
      if (!active[i]) continue;
      for (let j = i + 1; j < n; j++) {
        if (!active[j]) continue;
        const v = d[i * n + j] as number;
        if (v < best) {
          best = v;
          bi = i;
          bj = j;
        }
      }
    }
    const ia = id[bi] as number;
    const ib = id[bj] as number;
    const ni = size[ia] as number;
    const nj = size[ib] as number;
    // Lance–Williams update: the distance from the merged cluster to every other cluster k
    for (let k = 0; k < n; k++) {
      if (!active[k] || k === bi || k === bj) continue;
      const nk = size[id[k] as number] as number;
      const dik = d[bi * n + k] as number;
      const djk = d[bj * n + k] as number;
      let v: number;
      switch (method) {
        case 'single':
          v = Math.min(dik, djk);
          break;
        case 'complete':
          v = Math.max(dik, djk);
          break;
        case 'average':
          v = (ni * dik + nj * djk) / (ni + nj);
          break;
        default:
          v = Math.sqrt(
            ((ni + nk) * dik * dik + (nj + nk) * djk * djk - nk * best * best) / (ni + nj + nk),
          );
      }
      d[bi * n + k] = d[k * n + bi] = v;
    }
    active[bj] = false;
    const newId = n + step;
    size[newId] = ni + nj;
    id[bi] = newId;
    merges.push({ a: Math.min(ia, ib), b: Math.max(ia, ib), height: best, size: ni + nj });
  }
  return merges;
}

/** Flat clustering with exactly `k` clusters: undo the last `k − 1` merges. Labels are 0 … k−1. */
export function cutTree(merges: readonly Merge[], n: number, k: number): Int32Array {
  const parent = Array.from({ length: 2 * n - 1 }, (_, i) => i);
  const find = (i: number): number =>
    parent[i] === i ? i : (parent[i] = find(parent[i] as number));
  const keep = n - Math.max(1, Math.min(n, k));
  merges.slice(0, keep).forEach((m, i) => {
    parent[find(m.a)] = n + i;
    parent[find(m.b)] = n + i;
  });
  const labelOf = new Map<number, number>();
  return Int32Array.from({ length: n }, (_, i) => {
    const r = find(i);
    if (!labelOf.has(r)) labelOf.set(r, labelOf.size);
    return labelOf.get(r) as number;
  });
}

/** Flat clustering where no merge above `height` is performed. */
export function cutAtHeight(merges: readonly Merge[], n: number, height: number): Int32Array {
  const done = merges.filter((m) => m.height <= height).length;
  return cutTree(merges, n, n - done);
}

/** Left-to-right order of the leaves in a dendrogram (SciPy's convention: the first child goes left). */
export function dendrogramOrder(merges: readonly Merge[], n: number): number[] {
  const out: number[] = [];
  const walk = (node: number) => {
    if (node < n) out.push(node);
    else {
      const m = merges[node - n] as Merge;
      walk(m.a);
      walk(m.b);
    }
  };
  walk(n + merges.length - 1);
  return out;
}

/** Cophenetic distances: the height at which every pair of points first lands in the same cluster (condensed order i < j). */
export function cophenetic(merges: readonly Merge[], n: number): Float64Array {
  const members: number[][] = Array.from({ length: 2 * n - 1 }, (_, i) => (i < n ? [i] : []));
  const out = new Float64Array((n * (n - 1)) / 2);
  const at = (i: number, j: number) => i * n - (i * (i + 1)) / 2 + (j - i - 1);
  merges.forEach((m, s) => {
    const left = members[m.a] as number[];
    const right = members[m.b] as number[];
    for (const p of left) for (const q of right) out[at(Math.min(p, q), Math.max(p, q))] = m.height;
    members[n + s] = [...left, ...right];
  });
  return out;
}

/** Pearson correlation between the original and the cophenetic distances: how faithfully the tree keeps the geometry. */
export function copheneticCorrelation(
  distances: Float64Array,
  n: number,
  merges: readonly Merge[],
): number {
  const c = cophenetic(merges, n);
  const original: number[] = [];
  for (let i = 0; i < n; i++)
    for (let j = i + 1; j < n; j++) original.push(distances[i * n + j] as number);
  const mean = (v: ArrayLike<number>) => Array.from(v).reduce((a, b) => a + b, 0) / v.length;
  const ma = mean(original);
  const mb = mean(c);
  let num = 0;
  let da = 0;
  let db = 0;
  original.forEach((v, i) => {
    num += (v - ma) * ((c[i] as number) - mb);
    da += (v - ma) ** 2;
    db += ((c[i] as number) - mb) ** 2;
  });
  return num / Math.sqrt(da * db);
}

/** Agglomerative clustering as a `Clusterer`: build the tree, then cut it into `k` clusters. */
export class Agglomerative {
  merges: Merge[] = [];
  labels: Int32Array = new Int32Array(0);
  distances: Float64Array = new Float64Array(0);

  constructor(readonly options: { k: number; linkage: Linkage }) {}

  fit(x: Matrix): this {
    this.distances = pairwiseDistances(x);
    this.merges = linkageFromDistances(this.distances, x.rows, this.options.linkage);
    this.labels = cutTree(this.merges, x.rows, this.options.k);
    return this;
  }

  /** The cluster each point would get if the tree were cut into `k` clusters instead. */
  cut(k: number): Int32Array {
    return cutTree(this.merges, this.labels.length, k);
  }
}
