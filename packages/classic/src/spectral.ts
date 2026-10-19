import { Matrix, symmetricEigen, type Clusterer } from '@sim/core';
import { KMeans } from './kmeans';
import { pairwiseDistances } from './hierarchical';

/**
 * Spectral clustering. Build a similarity graph over the points, take the first k eigenvectors of its
 * normalised Laplacian (they turn "connected through a chain of neighbours" into "close together"),
 * then run k-means on those coordinates. This is what lets it separate shapes such as crescents or
 * rings that defeat k-means on the raw coordinates.
 */
export type Affinity = { kind: 'rbf'; gamma: number } | { kind: 'knn'; neighbors: number };

export interface SpectralOptions {
  k: number;
  affinity: Affinity;
  seed?: number;
}

/** Symmetric similarity matrix with a zero diagonal (flat `n × n`). */
export function affinityMatrix(x: Matrix, affinity: Affinity): Float64Array {
  const n = x.rows;
  const d = pairwiseDistances(x);
  const w = new Float64Array(n * n);
  if (affinity.kind === 'rbf') {
    for (let i = 0; i < n; i++)
      for (let j = 0; j < n; j++)
        if (i !== j) w[i * n + j] = Math.exp(-affinity.gamma * (d[i * n + j] as number) ** 2);
    return w;
  }
  const k = Math.min(affinity.neighbors, n - 1);
  const near = new Uint8Array(n * n);
  for (let i = 0; i < n; i++) {
    const order = Array.from({ length: n }, (_, j) => j)
      .filter((j) => j !== i)
      .sort((a, b) => (d[i * n + a] as number) - (d[i * n + b] as number) || a - b)
      .slice(0, k);
    for (const j of order) near[i * n + j] = 1;
  }
  // connected if either is among the other's nearest neighbours: average with the transpose
  for (let i = 0; i < n; i++)
    for (let j = 0; j < n; j++)
      w[i * n + j] = 0.5 * ((near[i * n + j] as number) + (near[j * n + i] as number));
  return w;
}

export interface SpectralEmbedding {
  /** Eigenvalues of the normalised Laplacian `I − D^(−1/2) W D^(−1/2)`, smallest first. */
  eigenvalues: Float64Array;
  /** The first `k` eigenvectors as columns (`n × k`), before row normalisation. */
  vectors: Matrix;
}

/** Eigen-decomposition of the normalised Laplacian of an affinity matrix. */
export function spectralEmbedding(w: Float64Array, n: number, k: number): SpectralEmbedding {
  const deg = new Float64Array(n);
  for (let i = 0; i < n; i++)
    for (let j = 0; j < n; j++) deg[i] = (deg[i] as number) + (w[i * n + j] as number);
  const m = new Matrix(n, n);
  for (let i = 0; i < n; i++)
    for (let j = 0; j < n; j++) {
      const s = Math.sqrt((deg[i] as number) * (deg[j] as number));
      m.set(i, j, s > 0 ? (w[i * n + j] as number) / s : 0);
    }
  const { values, vectors } = symmetricEigen(m); // largest first
  const eigenvalues = Float64Array.from(values, (v) => 1 - v);
  const take = Math.min(k, n);
  const out = new Matrix(n, take);
  for (let i = 0; i < n; i++) for (let c = 0; c < take; c++) out.set(i, c, vectors.get(i, c));
  return { eigenvalues, vectors: out };
}

export class SpectralClustering implements Clusterer {
  labels: Int32Array = new Int32Array(0);
  /** The row-normalised spectral coordinates the final k-means ran on. */
  embedding: Matrix = new Matrix(0, 0);
  eigenvalues: Float64Array = new Float64Array(0);

  constructor(readonly options: SpectralOptions) {}

  fit(x: Matrix): this {
    const { k, affinity, seed = 1 } = this.options;
    const n = x.rows;
    const w = affinityMatrix(x, affinity);
    const { eigenvalues, vectors } = spectralEmbedding(w, n, k);
    this.eigenvalues = eigenvalues;
    const rows = new Matrix(n, vectors.cols);
    for (let i = 0; i < n; i++) {
      let norm = 0;
      for (let c = 0; c < vectors.cols; c++) norm += vectors.get(i, c) ** 2;
      norm = Math.sqrt(norm) || 1;
      for (let c = 0; c < vectors.cols; c++) rows.set(i, c, vectors.get(i, c) / norm);
    }
    this.embedding = rows;
    this.labels = new KMeans({ k, restarts: 8, seed }).fit(rows).labels;
    return this;
  }
}

/** The number of eigenvalues below `tolerance`: how many (almost) disconnected pieces the graph has. */
export function nearZeroEigenvalues(eigenvalues: ArrayLike<number>, tolerance = 1e-3): number {
  let n = 0;
  for (let i = 0; i < eigenvalues.length; i++) if ((eigenvalues[i] as number) < tolerance) n++;
  return n;
}
