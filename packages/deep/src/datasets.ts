import { Matrix, selectRows, type Problem, permutation, type Rng } from '@sim/core';

export const CLASSIFICATION_DATASETS = [
  'circle',
  'xor',
  'checker',
  'gaussian',
  'spiral',
  'moons',
] as const;
export const REGRESSION_DATASETS = ['plane', 'saddle', 'ripple'] as const;
export type ClassificationDataset = (typeof CLASSIFICATION_DATASETS)[number];
export type RegressionDataset = (typeof REGRESSION_DATASETS)[number];
export type DatasetName = ClassificationDataset | RegressionDataset;

/** Half-width of the square the datasets live in; the UI draws [-DOMAIN, DOMAIN]². */
export const DOMAIN = 6;

export const MIN_CLASSES = 2;
export const MAX_CLASSES = 5;

export interface DatasetOptions {
  samples: number;
  /** 0..1 */
  noise: number;
  /** Ignored by regression and by `moons` (always 2). */
  classes: number;
}

export interface Dataset {
  problem: Problem;
  /** n×2 raw coordinates. */
  X: Matrix;
  /** Class index per sample (classification). Empty for regression. */
  labels: Int32Array;
  /** Target value per sample (regression). Empty for classification. */
  values: Float64Array;
  numClasses: number;
}

export const datasetInfo: Record<
  DatasetName,
  { label: string; problem: Problem; classes: readonly [number, number] }
> = {
  circle: { label: 'Circles', problem: 'classification', classes: [2, 5] },
  xor: { label: 'XOR', problem: 'classification', classes: [2, 2] },
  checker: { label: 'Checkerboard', problem: 'classification', classes: [2, 5] },
  gaussian: { label: 'Gaussians', problem: 'classification', classes: [2, 5] },
  spiral: { label: 'Spiral', problem: 'classification', classes: [2, 5] },
  moons: { label: 'Moons', problem: 'classification', classes: [2, 2] },
  plane: { label: 'Plane', problem: 'regression', classes: [1, 1] },
  saddle: { label: 'Waves', problem: 'regression', classes: [1, 1] },
  ripple: { label: 'Ripple', problem: 'regression', classes: [1, 1] },
};

export function isClassificationDataset(name: DatasetName): name is ClassificationDataset {
  return (CLASSIFICATION_DATASETS as readonly string[]).includes(name);
}

function clampClasses(name: ClassificationDataset, requested: number): number {
  const [lo, hi] = datasetInfo[name].classes;
  return Math.min(hi, Math.max(lo, Math.round(requested)));
}

function gaussianNoise(rng: Rng, scale: number): number {
  return scale === 0 ? 0 : rng.normal() * scale;
}

/** Position jitter plus a small fraction of flipped labels, scaled by `noise`. */
function noisy(xs: number[], ys: number[], labels: number[], k: number, noise: number, rng: Rng) {
  for (let i = 0; i < xs.length; i++) {
    xs[i] = (xs[i] as number) + gaussianNoise(rng, noise * 0.5);
    ys[i] = (ys[i] as number) + gaussianNoise(rng, noise * 0.5);
  }
  if (noise > 0) {
    for (let i = 0; i < labels.length; i++) {
      if (rng.next() < noise * 0.1) labels[i] = Math.floor(rng.next() * k);
    }
  }
}

/** Generates a 2-D classification dataset. Deterministic for a given `rng` state. */
export function generateClassification(
  name: ClassificationDataset,
  options: DatasetOptions,
  rng: Rng,
): Dataset {
  const { samples, noise } = options;
  const k = clampClasses(name, options.classes);
  const xs: number[] = [];
  const ys: number[] = [];
  const labels: number[] = [];
  const push = (x: number, y: number, label: number) => {
    xs.push(x);
    ys.push(y);
    labels.push(label);
  };

  switch (name) {
    case 'circle': {
      const per = Math.floor(samples / k);
      for (let c = 0; c < k; c++) {
        const radius = (c + 1) * (5 / k);
        for (let i = 0; i < per; i++) {
          const t = per === 1 ? 0 : (i / (per - 1)) * 2 * Math.PI;
          push(
            radius * Math.sin(t) + gaussianNoise(rng, noise * 0.5),
            radius * Math.cos(t) + gaussianNoise(rng, noise * 0.5),
            c,
          );
        }
      }
      break;
    }
    case 'gaussian': {
      const std = 0.5 + noise * 1.5;
      const centers = Array.from({ length: k }, () => [rng.uniform(-10, 10), rng.uniform(-10, 10)]);
      for (let i = 0; i < samples; i++) {
        const c = i % k;
        const center = centers[c] as number[];
        push(
          (center[0] as number) + rng.normal() * std,
          (center[1] as number) + rng.normal() * std,
          c,
        );
      }
      // Standardise, then scale so the clusters fill roughly [-5, 5].
      for (const arr of [xs, ys]) {
        const mean = arr.reduce((s, v) => s + v, 0) / arr.length;
        const sd = Math.sqrt(arr.reduce((s, v) => s + (v - mean) ** 2, 0) / arr.length) || 1;
        for (let i = 0; i < arr.length; i++) arr[i] = (((arr[i] as number) - mean) / sd) * 2.5;
      }
      break;
    }
    case 'xor': {
      // Classic quadrants: opposite corners share a class. Points keep a small margin from the
      // axes so the problem is solvable but still impossible for a purely linear model.
      for (let i = 0; i < samples; i++) {
        const x = rng.uniform(-5, 5);
        const y = rng.uniform(-5, 5);
        push(x + Math.sign(x) * 0.3, y + Math.sign(y) * 0.3, x > 0 !== y > 0 ? 1 : 0);
      }
      noisy(xs, ys, labels, k, noise, rng);
      break;
    }
    case 'checker': {
      // Multi-class checkerboard with big cells (a 4×4 grid over the plane).
      const cell = 2.5;
      for (let i = 0; i < samples; i++) {
        const x = rng.uniform(-5, 5);
        const y = rng.uniform(-5, 5);
        const raw = Math.floor(x / cell) + Math.floor(y / cell);
        push(x, y, ((raw % k) + k) % k);
      }
      noisy(xs, ys, labels, k, noise, rng);
      break;
    }
    case 'spiral': {
      const per = Math.floor(samples / k);
      for (let c = 0; c < k; c++) {
        for (let i = 0; i < per; i++) {
          const r = (i / per) * 5;
          const t =
            ((1.75 * i) / per) * 2 * Math.PI +
            (c * 2 * Math.PI) / k +
            gaussianNoise(rng, noise * 0.25);
          push(r * Math.sin(t), r * Math.cos(t), c);
        }
      }
      break;
    }
    case 'moons': {
      const half = Math.floor(samples / 2);
      for (let c = 0; c < 2; c++) {
        for (let i = 0; i < half; i++) {
          const t = half === 1 ? 0 : (i / (half - 1)) * Math.PI;
          const x = c === 0 ? Math.cos(t) : 1 - Math.cos(t);
          const y = c === 0 ? Math.sin(t) : 0.5 - Math.sin(t);
          push(
            (x - 0.5) * 3.4 + gaussianNoise(rng, noise * 0.5),
            (y - 0.25) * 3.4 + gaussianNoise(rng, noise * 0.5),
            c,
          );
        }
      }
      break;
    }
  }

  // Shuffle so that class blocks are not contiguous (matters for mini-batches and splits).
  const order = permutation(labels.length, rng);
  const X = new Matrix(labels.length, 2);
  const out = new Int32Array(labels.length);
  order.forEach((src, dst) => {
    X.set(dst, 0, xs[src] as number);
    X.set(dst, 1, ys[src] as number);
    out[dst] = labels[src] as number;
  });
  return { problem: 'classification', X, labels: out, values: new Float64Array(0), numClasses: k };
}

/** Generates a 2-D regression dataset (z = f(x, y) + noise). */
export function generateRegression(
  name: RegressionDataset,
  options: Pick<DatasetOptions, 'samples' | 'noise'>,
  rng: Rng,
): Dataset {
  const { samples, noise } = options;
  const X = new Matrix(samples, 2);
  const values = new Float64Array(samples);
  for (let i = 0; i < samples; i++) {
    const x = rng.uniform(-5, 5);
    const y = rng.uniform(-5, 5);
    X.set(i, 0, x);
    X.set(i, 1, y);
    let z: number;
    switch (name) {
      case 'plane':
        z = 0.5 * x + 0.5 * y;
        break;
      case 'saddle':
        z = Math.sin(x * 0.6) * Math.cos(y * 0.6) * 3;
        break;
      case 'ripple': {
        const r = Math.hypot(x, y);
        z = 3 * Math.sin(1.2 * r) * Math.exp(-r / 5);
        break;
      }
    }
    values[i] = z + gaussianNoise(rng, noise * 0.5);
  }
  return { problem: 'regression', X, labels: new Int32Array(0), values, numClasses: 1 };
}

export function generateDataset(name: DatasetName, options: DatasetOptions, rng: Rng): Dataset {
  return isClassificationDataset(name)
    ? generateClassification(name, options, rng)
    : generateRegression(name, options, rng);
}

/** One-hot (classification) or n×1 (regression) target matrix for a dataset subset. */
export function toTargets(dataset: Dataset, indices?: ArrayLike<number>): Matrix {
  const n = indices ? indices.length : dataset.X.rows;
  if (dataset.problem === 'regression') {
    const y = new Matrix(n, 1);
    for (let i = 0; i < n; i++)
      y.data[i] = dataset.values[indices ? (indices[i] as number) : i] as number;
    return y;
  }
  const y = new Matrix(n, dataset.numClasses);
  for (let i = 0; i < n; i++) {
    y.set(i, dataset.labels[indices ? (indices[i] as number) : i] as number, 1);
  }
  return y;
}

export interface Split {
  trainIndices: Uint32Array;
  testIndices: Uint32Array;
}

/**
 * Random train/test split. Both sides always keep at least one sample so that
 * test metrics are defined.
 */
export function splitIndices(n: number, trainPercent: number, rng: Rng): Split {
  const order = permutation(n, rng);
  const cut = Math.min(n - 1, Math.max(1, Math.floor((n * trainPercent) / 100)));
  return { trainIndices: order.slice(0, cut), testIndices: order.slice(cut) };
}

export function subsetX(dataset: Dataset, indices: ArrayLike<number>): Matrix {
  return selectRows(dataset.X, indices);
}
