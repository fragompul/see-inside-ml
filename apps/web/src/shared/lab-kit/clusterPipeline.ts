import {
  Matrix,
  adjustedRandIndex,
  createRng,
  gridPoints,
  makeClassification,
  plotBounds,
  silhouette,
  type ClassificationSet,
  type Clusterer,
  type Table,
} from '@sim/core';
import type { Regions } from './pipeline';

/**
 * The pure part of the unsupervised labs: generate points, hide the labels from the algorithm, run
 * it, and score what it found against the groups that were really there (which a real analyst would
 * not have). Tests of every lab's experiments call the same functions as the pages.
 */
export interface ClusterSetup {
  dataset: ClassificationSet;
  n: number;
  noise: number;
  seed: number;
  /** Extra points scattered uniformly over the plane, labelled -1 (the anomaly-detection lab). */
  outliers: number;
}

export const DEFAULT_CLUSTER_SETUP: ClusterSetup = {
  dataset: 'blobs',
  n: 240,
  noise: 0.1,
  seed: 4,
  outliers: 0,
};

export interface ClusterScores {
  /** Number of clusters found (noise excluded). */
  clusters: number;
  /** Points labelled −1 (density-based methods). */
  noise: number;
  /** Mean silhouette of the clustered points, or null with fewer than two clusters. */
  silhouette: number | null;
  /** Adjusted Rand index against the true groups: 1 = identical partitions, ~0 = no better than chance. */
  ari: number;
}

export interface ClusterRun {
  data: Table;
  model: Clusterer;
  labels: Int32Array;
  scores: ClusterScores;
}

export function buildClusterData(setup: ClusterSetup, custom?: Table | null): Table {
  if (custom) return custom;
  const base = makeClassification(setup.dataset, setup.n, setup.noise, createRng(setup.seed));
  if (setup.outliers <= 0) return base;
  // the strays come from their own stream so the base sample does not change when their number does
  const rng = createRng(setup.seed + 7919);
  const { x0, x1, y0, y1 } = plotBounds(base.x, 0.3);
  const n = base.x.rows;
  const total = n + setup.outliers;
  const x = new Matrix(total, 2);
  x.data.set(base.x.data);
  const y = new Float64Array(total).fill(-1);
  y.set(base.y);
  for (let i = n; i < total; i++) {
    x.set(i, 0, rng.uniform(x0, x1));
    x.set(i, 1, rng.uniform(y0, y1));
  }
  return { ...base, x, y };
}

export function scoreClusters(data: Table, labels: Int32Array): ClusterScores {
  let noise = 0;
  const kept: number[] = [];
  const seen = new Set<number>();
  labels.forEach((l, i) => {
    if (l < 0) noise++;
    else {
      kept.push(i);
      seen.add(l);
    }
  });
  let sil: number | null = null;
  if (seen.size >= 2 && kept.length > seen.size) {
    const sub = new Matrix(kept.length, data.x.cols);
    kept.forEach((r, i) =>
      sub.data.set(data.x.data.subarray(r * data.x.cols, (r + 1) * data.x.cols), i * data.x.cols),
    );
    sil = silhouette(
      sub,
      kept.map((r) => labels[r] as number),
    );
  }
  return {
    clusters: seen.size,
    noise,
    silhouette: sil,
    // noise points count as one extra group, which is how a partition metric must treat them
    ari: adjustedRandIndex(data.y, labels),
  };
}

export function runClusterer(model: Clusterer, data: Table): ClusterRun {
  model.fit(data.x);
  return { data, model, labels: model.labels, scores: scoreClusters(data, model.labels) };
}

/** Which cluster a new point would join, over a grid, for models that can predict. */
export function clusterRegions(model: Clusterer, data: Table, n = 70): Regions | null {
  const predictor = model as Clusterer & { predict?: (x: Matrix) => Int32Array };
  if (!predictor.predict) return null;
  const bounds = plotBounds(data.x, 0.2);
  const grid = gridPoints(bounds.x0, bounds.x1, bounds.y0, bounds.y1, n);
  return {
    n,
    labels: predictor.predict(grid),
    confidence: new Float64Array(n * n).fill(1),
    bounds,
  };
}
