import { Matrix, createRng, solve, symmetricEigen } from '@sim/core';
import { pairwiseDistances } from './hierarchical';

/**
 * Embedding methods that squeeze high-dimensional points into a few dimensions while keeping what
 * matters: classical and metric MDS (keep all distances), Isomap (keep distances along the data's
 * surface), LLE (keep each point's local linear recipe), plus the standard ways to judge an
 * embedding: do neighbours stay neighbours?
 */

/** Squared Euclidean distances (flat `n × n`). */
export function squaredDistances(x: Matrix): Float64Array {
  const d = pairwiseDistances(x);
  return d.map((v) => v * v);
}

function takeColumns(
  vectors: Matrix,
  values: ArrayLike<number>,
  k: number,
  scale: boolean,
): Matrix {
  const n = vectors.rows;
  const out = new Matrix(n, k);
  for (let c = 0; c < k; c++) {
    // sign convention of scikit-learn's svd_flip: the entry of largest magnitude is positive
    let best = 0;
    for (let i = 1; i < n; i++)
      if (Math.abs(vectors.get(i, c)) > Math.abs(vectors.get(best, c))) best = i;
    const sign = vectors.get(best, c) < 0 ? -1 : 1;
    const s = scale ? Math.sqrt(Math.max(values[c] as number, 0)) : 1;
    for (let i = 0; i < n; i++) out.set(i, c, sign * s * vectors.get(i, c));
  }
  return out;
}

/** Double-centres a matrix of squared distances: `B = −½ J D² J` (the Gram matrix of the centred points). */
export function gramFromSquaredDistances(d2: Float64Array, n: number): Matrix {
  const rowMean = new Float64Array(n);
  let total = 0;
  for (let i = 0; i < n; i++) {
    let s = 0;
    for (let j = 0; j < n; j++) s += d2[i * n + j] as number;
    rowMean[i] = s / n;
    total += s;
  }
  total /= n * n;
  const b = new Matrix(n, n);
  for (let i = 0; i < n; i++)
    for (let j = 0; j < n; j++)
      b.set(
        i,
        j,
        -0.5 *
          ((d2[i * n + j] as number) - (rowMean[i] as number) - (rowMean[j] as number) + total),
      );
  return b;
}

/** Classical (Torgerson) MDS: the coordinates whose distances best match `distances` (flat `n × n`). */
export function classicalMds(
  distances: Float64Array,
  n: number,
  dims = 2,
): { embedding: Matrix; eigenvalues: Float64Array } {
  const d2 = distances.map((v) => v * v);
  const { values, vectors } = symmetricEigen(gramFromSquaredDistances(d2, n));
  return { embedding: takeColumns(vectors, values, dims, true), eigenvalues: values };
}

/** Raw stress of an embedding: Σ (dᵢⱼ − δᵢⱼ)² over pairs, with δ the original distances. */
export function stress(distances: Float64Array, embedding: Matrix): number {
  const n = embedding.rows;
  const dz = pairwiseDistances(embedding);
  let s = 0;
  for (let i = 0; i < n; i++)
    for (let j = i + 1; j < n; j++)
      s += ((dz[i * n + j] as number) - (distances[i * n + j] as number)) ** 2;
  return s;
}

export interface SmacofState {
  iteration: number;
  embedding: Matrix;
  stress: number;
}

/**
 * Metric MDS by SMACOF: repeatedly move every point to the average of where its distances say it
 * should be (the Guttman transform). Mirrors scikit-learn's `smacof`: state 0 is the starting
 * configuration, then one state per update with the stress measured *after* it, and the run stops
 * when the improvement relative to the total squared distance falls below `tolerance`.
 */
export function* smacofSteps(
  distances: Float64Array,
  n: number,
  init: Matrix,
  options: { maxIterations?: number; tolerance?: number } = {},
): Generator<SmacofState, void, undefined> {
  const { maxIterations = 300, tolerance = 1e-3 } = options;
  const dims = init.cols;
  let x = new Matrix(n, dims, Float64Array.from(init.data));
  const rawStress = (dis: Float64Array) => {
    let s = 0;
    for (let i = 0; i < n; i++)
      for (let j = i + 1; j < n; j++)
        s += ((dis[i * n + j] as number) - (distances[i * n + j] as number)) ** 2;
    return s;
  };
  let dis = pairwiseDistances(x);
  yield { iteration: 0, embedding: x, stress: rawStress(dis) };
  let old: number | null = null;
  for (let it = 0; it < maxIterations; it++) {
    // Guttman transform with unit weights: X <- (1/n) B(X) X, where B = -ratio off the diagonal
    const next = new Matrix(n, dims);
    for (let i = 0; i < n; i++) {
      let diag = 0;
      for (let j = 0; j < n; j++) {
        if (i === j) continue;
        const dij = Math.max(dis[i * n + j] as number, 1e-5);
        const ratio = (distances[i * n + j] as number) / dij;
        diag += ratio;
        for (let c = 0; c < dims; c++)
          next.data[i * dims + c] = (next.data[i * dims + c] as number) - ratio * x.get(j, c);
      }
      for (let c = 0; c < dims; c++)
        next.data[i * dims + c] = ((next.data[i * dims + c] as number) + diag * x.get(i, c)) / n;
    }
    x = next;
    dis = pairwiseDistances(x);
    const stress = rawStress(dis);
    yield { iteration: it + 1, embedding: x, stress };
    if (old !== null) {
      let sumSquared = 0;
      for (let i = 0; i < n; i++)
        for (let j = i + 1; j < n; j++) sumSquared += (dis[i * n + j] as number) ** 2;
      if ((old - stress) / sumSquared < tolerance) return;
    }
    old = stress;
  }
}

// ---- Isomap ----------------------------------------------------------------------------------

/** Shortest-path (geodesic) distances over the k-nearest-neighbour graph, treated as undirected. */
export function geodesicDistances(x: Matrix, neighbors: number): Float64Array {
  const n = x.rows;
  const d = pairwiseDistances(x);
  const g = new Float64Array(n * n).fill(Infinity);
  for (let i = 0; i < n; i++) {
    g[i * n + i] = 0;
    const order = Array.from({ length: n }, (_, j) => j)
      .filter((j) => j !== i)
      .sort((a, b) => (d[i * n + a] as number) - (d[i * n + b] as number) || a - b)
      .slice(0, neighbors);
    for (const j of order) {
      const w = d[i * n + j] as number;
      g[i * n + j] = Math.min(g[i * n + j] as number, w);
      g[j * n + i] = Math.min(g[j * n + i] as number, w);
    }
  }
  // Floyd–Warshall: fine for the few hundred points a lab uses
  for (let k = 0; k < n; k++)
    for (let i = 0; i < n; i++) {
      const dik = g[i * n + k] as number;
      if (dik === Infinity) continue;
      for (let j = 0; j < n; j++) {
        const v = dik + (g[k * n + j] as number);
        if (v < (g[i * n + j] as number)) g[i * n + j] = v;
      }
    }
  return g;
}

/** How many separate pieces the neighbour graph falls into (Isomap needs one). */
export function graphComponents(x: Matrix, neighbors: number): number {
  const g = geodesicDistances(x, neighbors);
  const n = x.rows;
  const seen = new Array<boolean>(n).fill(false);
  let count = 0;
  for (let i = 0; i < n; i++) {
    if (seen[i]) continue;
    count++;
    for (let j = 0; j < n; j++) if (g[i * n + j] !== Infinity) seen[j] = true;
  }
  return count;
}

export function isomap(
  x: Matrix,
  neighbors: number,
  dims = 2,
): { embedding: Matrix; geodesic: Float64Array; eigenvalues: Float64Array } {
  const n = x.rows;
  const geodesic = geodesicDistances(x, neighbors);
  if (geodesic.some((v) => v === Infinity))
    throw new Error('The neighbour graph is disconnected: raise the number of neighbours');
  const d2 = geodesic.map((v) => v * v);
  const { values, vectors } = symmetricEigen(gramFromSquaredDistances(d2, n));
  return { embedding: takeColumns(vectors, values, dims, true), geodesic, eigenvalues: values };
}

// ---- locally linear embedding -------------------------------------------------------------------

/** Weights that reconstruct each point from its neighbours (rows sum to one), regularised like scikit-learn. */
export function lleWeights(
  x: Matrix,
  neighbors: number,
  reg = 1e-3,
): { weights: Matrix; neighborIndex: number[][] } {
  const n = x.rows;
  const d = pairwiseDistances(x);
  const weights = new Matrix(n, n);
  const neighborIndex: number[][] = [];
  for (let i = 0; i < n; i++) {
    const nb = Array.from({ length: n }, (_, j) => j)
      .filter((j) => j !== i)
      .sort((a, b) => (d[i * n + a] as number) - (d[i * n + b] as number) || a - b)
      .slice(0, neighbors);
    neighborIndex.push(nb);
    const k = nb.length;
    const z = nb.map((j) => Array.from({ length: x.cols }, (_, c) => x.get(j, c) - x.get(i, c)));
    const gram = new Matrix(k, k);
    for (let a = 0; a < k; a++)
      for (let b = 0; b < k; b++) {
        let s = 0;
        for (let c = 0; c < x.cols; c++) s += (z[a] as number[])[c]! * (z[b] as number[])[c]!;
        gram.set(a, b, s);
      }
    let trace = 0;
    for (let a = 0; a < k; a++) trace += gram.get(a, a);
    const r = trace > 0 ? reg * trace : reg;
    for (let a = 0; a < k; a++) gram.set(a, a, gram.get(a, a) + r);
    const w = solve(gram, new Matrix(k, 1, new Float64Array(k).fill(1))).data;
    const sum = w.reduce((a, b) => a + b, 0);
    nb.forEach((j, a) => weights.set(i, j, (w[a] as number) / sum));
  }
  return { weights, neighborIndex };
}

export function lle(
  x: Matrix,
  neighbors: number,
  dims = 2,
  reg = 1e-3,
): { embedding: Matrix; eigenvalues: Float64Array; reconstructionError: number } {
  const n = x.rows;
  const { weights } = lleWeights(x, neighbors, reg);
  // M = (I − W)ᵀ (I − W): the embedding minimises Σ‖yᵢ − Σⱼ wᵢⱼ yⱼ‖²
  const iw = new Matrix(n, n);
  for (let i = 0; i < n; i++)
    for (let j = 0; j < n; j++) iw.set(i, j, (i === j ? 1 : 0) - weights.get(i, j));
  const m = new Matrix(n, n);
  for (let a = 0; a < n; a++)
    for (let b = a; b < n; b++) {
      let s = 0;
      for (let i = 0; i < n; i++) s += iw.get(i, a) * iw.get(i, b);
      m.set(a, b, s);
      m.set(b, a, s);
    }
  const { values, vectors } = symmetricEigen(m); // largest first: the smallest are at the end
  // skip the smallest (the constant vector) and take the next `dims`
  const out = new Matrix(n, dims);
  const used = new Float64Array(dims);
  for (let c = 0; c < dims; c++) {
    const idx = n - 2 - c;
    used[c] = values[idx] as number;
    let best = 0;
    for (let i = 1; i < n; i++)
      if (Math.abs(vectors.get(i, idx)) > Math.abs(vectors.get(best, idx))) best = i;
    const sign = vectors.get(best, idx) < 0 ? -1 : 1;
    for (let i = 0; i < n; i++) out.set(i, c, sign * vectors.get(i, idx));
  }
  let err = 0;
  for (let c = 0; c < dims; c++) err += used[c] as number;
  return { embedding: out, eigenvalues: used, reconstructionError: err };
}

// ---- judging an embedding ------------------------------------------------------------------------

function ranks(d: Float64Array, n: number): Int32Array[] {
  // rank of every point as seen from i: 0 = i itself, 1 = nearest, …
  return Array.from({ length: n }, (_, i) => {
    const order = Array.from({ length: n }, (_, j) => j).sort(
      (a, b) => (d[i * n + a] as number) - (d[i * n + b] as number) || a - b,
    );
    const r = new Int32Array(n);
    order.forEach((j, pos) => (r[j] = pos));
    return r;
  });
}

/**
 * Trustworthiness (scikit-learn's definition): of the k nearest neighbours in the embedding, how
 * many are genuinely near in the original space? 1 = no false neighbours.
 */
export function trustworthiness(original: Matrix, embedded: Matrix, k = 5): number {
  const n = original.rows;
  const rankOriginal = ranks(pairwiseDistances(original), n);
  const dz = pairwiseDistances(embedded);
  let t = 0;
  for (let i = 0; i < n; i++) {
    const near = Array.from({ length: n }, (_, j) => j)
      .filter((j) => j !== i)
      .sort((a, b) => (dz[i * n + a] as number) - (dz[i * n + b] as number) || a - b)
      .slice(0, k);
    for (const j of near) t += Math.max(0, (rankOriginal[i] as Int32Array)[j]! - k);
  }
  return 1 - (2 / (n * k * (2 * n - 3 * k - 1))) * t;
}

/** Continuity: the mirror image of trustworthiness (are original neighbours still neighbours in the embedding?). */
export function continuity(original: Matrix, embedded: Matrix, k = 5): number {
  return trustworthiness(embedded, original, k);
}

/** Average share of each point's k nearest original neighbours that are among its k nearest in the embedding. */
export function neighbourPreservation(original: Matrix, embedded: Matrix, k = 5): number {
  const n = original.rows;
  const d1 = pairwiseDistances(original);
  const d2 = pairwiseDistances(embedded);
  const nearest = (d: Float64Array, i: number) =>
    new Set(
      Array.from({ length: n }, (_, j) => j)
        .filter((j) => j !== i)
        .sort((a, b) => (d[i * n + a] as number) - (d[i * n + b] as number) || a - b)
        .slice(0, k),
    );
  let total = 0;
  for (let i = 0; i < n; i++) {
    const a = nearest(d1, i);
    const b = nearest(d2, i);
    let shared = 0;
    for (const j of a) if (b.has(j)) shared++;
    total += shared / k;
  }
  return total / n;
}

/** A tiny seeded random start for iterative embeddings (small Gaussian blobs around the origin). */
export function randomEmbedding(n: number, dims: number, seed: number, scale = 1e-2): Matrix {
  const rng = createRng(seed);
  return new Matrix(
    n,
    dims,
    Float64Array.from({ length: n * dims }, () => rng.normal() * scale),
  );
}
