import { Matrix } from './matrix';
import { permutation, type Rng } from './rng';
import { columnMeans, std } from './stats';

/**
 * Tabular data for the classical-model labs: a feature matrix and a target, plus generators of
 * small, reproducible datasets (2-D classification problems and regression curves with the
 * troubles real data has: noise, outliers, unequal variance, correlated features).
 */
export interface Table {
  /** `n × d` features. */
  x: Matrix;
  /** Target: class labels 0 … k − 1 for classification, real numbers for regression. */
  y: Float64Array;
  names: string[];
  /** Class names for classification tables. */
  classes?: string[];
}

export const tableSize = (t: Table): number => t.x.rows;

export function makeTable(
  rows: number[][],
  y: ArrayLike<number>,
  names: string[],
  classes?: string[],
): Table {
  const d = rows[0]?.length ?? 0;
  const x = new Matrix(rows.length, d);
  rows.forEach((r, i) => x.data.set(r, i * d));
  return { x, y: Float64Array.from(y), names, ...(classes ? { classes } : {}) };
}

// ---- 2-D classification ---------------------------------------------------------------------

export const CLASSIFICATION_SETS = [
  'blobs',
  'moons',
  'circles',
  'spirals',
  'xor',
  'imbalanced',
  'overlap',
  'tilted',
  'cigars',
] as const;
export type ClassificationSet = (typeof CLASSIFICATION_SETS)[number];

export const classificationSetInfo: Record<ClassificationSet, { label: string; note: string }> = {
  blobs: {
    label: 'Blobs',
    note: 'Three well-separated Gaussian clusters: every model should manage.',
  },
  moons: {
    label: 'Moons',
    note: 'Two interleaved crescents: a straight line cannot separate them.',
  },
  circles: {
    label: 'Circles',
    note: 'One class inside a ring of the other: the boundary must be closed.',
  },
  spirals: {
    label: 'Spirals',
    note: 'Two intertwined spirals: the hardest of the set for simple models.',
  },
  xor: { label: 'XOR', note: 'Classes on opposite corners: no single line works.' },
  imbalanced: {
    label: 'Imbalanced',
    note: 'Ten times fewer points in one class: accuracy can mislead.',
  },
  overlap: {
    label: 'Overlapping',
    note: 'Two clusters that overlap: even the best model must make mistakes.',
  },
  tilted: {
    label: 'Correlated',
    note: 'Two classes shifted along x₁ inside a strongly correlated cloud: the telling direction is a diagonal, not an axis.',
  },
  cigars: {
    label: 'Cigars',
    note: 'Same centre, different orientation: only a model with a covariance per class can tell them apart.',
  },
};

const gauss = (rng: Rng, mu: number, sigma: number) => mu + sigma * rng.normal();

/** Generates a 2-D classification dataset with about `n` points. */
export function makeClassification(
  kind: ClassificationSet,
  n: number,
  noise: number,
  rng: Rng,
): Table {
  const rows: number[][] = [];
  const y: number[] = [];
  const push = (a: number, b: number, c: number) => {
    rows.push([a, b]);
    y.push(c);
  };
  switch (kind) {
    case 'blobs': {
      const centers = [
        [-1.6, -1.2],
        [1.7, -1],
        [0, 1.9],
      ] as const;
      for (let i = 0; i < n; i++) {
        const c = i % 3;
        push(gauss(rng, centers[c]![0], 0.55 + noise), gauss(rng, centers[c]![1], 0.55 + noise), c);
      }
      break;
    }
    case 'moons':
      for (let i = 0; i < n; i++) {
        const c = i % 2;
        const t = rng.next() * Math.PI;
        const a = c === 0 ? Math.cos(t) : 1 - Math.cos(t);
        const b = c === 0 ? Math.sin(t) : 0.5 - Math.sin(t);
        push((a - 0.5) * 1.8 + gauss(rng, 0, noise), (b - 0.25) * 1.8 + gauss(rng, 0, noise), c);
      }
      break;
    case 'circles':
      for (let i = 0; i < n; i++) {
        const c = i % 2;
        const t = rng.next() * 2 * Math.PI;
        const r = c === 0 ? 2 : 0.9;
        push(r * Math.cos(t) + gauss(rng, 0, noise), r * Math.sin(t) + gauss(rng, 0, noise), c);
      }
      break;
    case 'spirals':
      for (let i = 0; i < n; i++) {
        const c = i % 2;
        const t = 0.25 + (rng.next() * 3.2 * Math.PI) / 1;
        const r = t / 4.2;
        const phase = c * Math.PI;
        push(
          r * Math.cos(t + phase) * 1.6 + gauss(rng, 0, noise),
          r * Math.sin(t + phase) * 1.6 + gauss(rng, 0, noise),
          c,
        );
      }
      break;
    case 'xor':
      for (let i = 0; i < n; i++) {
        const a = rng.next() * 4 - 2;
        const b = rng.next() * 4 - 2;
        push(a + gauss(rng, 0, noise), b + gauss(rng, 0, noise), a * b > 0 ? 1 : 0);
      }
      break;
    case 'imbalanced':
      for (let i = 0; i < n; i++) {
        const c = i % 11 === 0 ? 1 : 0;
        push(
          gauss(rng, c === 1 ? 1.3 : -0.8, 0.9 + noise),
          gauss(rng, c === 1 ? 0.8 : -0.4, 0.9 + noise),
          c,
        );
      }
      break;
    case 'overlap':
      for (let i = 0; i < n; i++) {
        const c = i % 2;
        push(
          gauss(rng, c === 0 ? -0.7 : 0.7, 1 + noise),
          gauss(rng, c === 0 ? -0.5 : 0.5, 1 + noise),
          c,
        );
      }
      break;
    case 'tilted':
      for (let i = 0; i < n; i++) {
        const c = i % 2;
        const a = rng.normal();
        const b = 0.92 * a + Math.sqrt(1 - 0.92 * 0.92) * rng.normal();
        const s = 1.2 + noise;
        push(s * a + (c === 0 ? -0.8 : 0.8), s * b, c);
      }
      break;
    case 'cigars':
      for (let i = 0; i < n; i++) {
        const c = i % 2;
        const u = gauss(rng, 0, 1.7);
        const v = gauss(rng, 0, 0.3 + noise / 2);
        const theta = c === 0 ? Math.PI / 4 : -Math.PI / 4;
        push(
          u * Math.cos(theta) - v * Math.sin(theta),
          u * Math.sin(theta) + v * Math.cos(theta),
          c,
        );
      }
      break;
  }
  const classes =
    kind === 'blobs' ? ['A', 'B', 'C'] : kind === 'imbalanced' ? ['common', 'rare'] : ['A', 'B'];
  return makeTable(rows, y, ['x₁', 'x₂'], classes);
}

// ---- regression -----------------------------------------------------------------------------

export const REGRESSION_SETS = [
  'line',
  'curve',
  'sine',
  'heteroscedastic',
  'outliers',
  'step',
  'counts',
] as const;
export type RegressionSet = (typeof REGRESSION_SETS)[number];

export const regressionSetInfo: Record<RegressionSet, { label: string; note: string }> = {
  line: {
    label: 'Straight line',
    note: 'A linear trend plus noise: where a line is exactly right.',
  },
  curve: { label: 'Parabola', note: 'A quadratic trend: a line underfits it.' },
  sine: {
    label: 'Sine wave',
    note: 'A wiggly signal: needs a flexible model, and invites overfitting.',
  },
  heteroscedastic: {
    label: 'Growing noise',
    note: 'The scatter grows with x: the usual assumptions fail.',
  },
  outliers: {
    label: 'With outliers',
    note: 'A clean trend with a few wild points that tug on a least-squares fit.',
  },
  step: { label: 'Step', note: 'A jump at x = 0: smooth models blur it, trees capture it.' },
  counts: {
    label: 'Counts',
    note: 'Whole-number counts whose average grows exponentially with x (the noise slider is ignored).',
  },
};

/** 1-D regression data, `x` in [−3, 3], with `n` points. */
export function makeRegression(kind: RegressionSet, n: number, noise: number, rng: Rng): Table {
  const rows: number[][] = [];
  const y: number[] = [];
  for (let i = 0; i < n; i++) {
    const x = rng.next() * 6 - 3;
    let t: number;
    switch (kind) {
      case 'line':
        t = 0.9 * x + 0.5 + gauss(rng, 0, noise);
        break;
      case 'curve':
        t = 0.45 * x * x - 0.6 * x - 1 + gauss(rng, 0, noise);
        break;
      case 'sine':
        t = 1.6 * Math.sin(1.7 * x) + gauss(rng, 0, noise);
        break;
      case 'heteroscedastic':
        t = 0.8 * x + gauss(rng, 0, noise * (0.25 + Math.abs(x + 3) / 3));
        break;
      case 'outliers':
        t =
          0.9 * x +
          0.5 +
          gauss(rng, 0, noise) +
          (i % 13 === 0 ? (rng.next() < 0.5 ? -1 : 1) * (4 + rng.next() * 3) : 0);
        break;
      case 'step':
        t = (x < 0 ? -1.5 : 1.5) + gauss(rng, 0, noise);
        break;
      case 'counts': {
        // Poisson(exp(0.45·x + 0.6)) by Knuth's multiplication method
        const limit = Math.exp(-Math.exp(0.45 * x + 0.6));
        let k = 0;
        let p = 1;
        do {
          k++;
          p *= rng.next();
        } while (p > limit);
        t = k - 1;
        break;
      }
    }
    rows.push([x]);
    y.push(t);
  }
  return makeTable(rows, y, ['x']);
}

/** Anscombe's quartet: four datasets with the same mean, variance, correlation and regression line. */
export function anscombe(which: 1 | 2 | 3 | 4): Table {
  const x123 = [10, 8, 13, 9, 11, 14, 6, 4, 12, 7, 5];
  const sets: Record<number, { x: number[]; y: number[] }> = {
    1: { x: x123, y: [8.04, 6.95, 7.58, 8.81, 8.33, 9.96, 7.24, 4.26, 10.84, 4.82, 5.68] },
    2: { x: x123, y: [9.14, 8.14, 8.74, 8.77, 9.26, 8.1, 6.13, 3.1, 9.13, 7.26, 4.74] },
    3: { x: x123, y: [7.46, 6.77, 12.74, 7.11, 7.81, 8.84, 6.08, 5.39, 8.15, 6.42, 5.73] },
    4: {
      x: [8, 8, 8, 8, 8, 8, 8, 19, 8, 8, 8],
      y: [6.58, 5.76, 7.71, 8.84, 8.47, 7.04, 5.25, 12.5, 5.56, 7.91, 6.89],
    },
  };
  const s = sets[which] as { x: number[]; y: number[] };
  return makeTable(
    s.x.map((v) => [v]),
    s.y,
    ['x'],
  );
}

/**
 * A multi-feature regression table with correlated features: `x₂ ≈ ρ·x₁ + noise`, and a response
 * that depends on the first, second and (weakly) third feature.
 */
export function makeMultiple(n: number, noise: number, correlation: number, rng: Rng): Table {
  const rows: number[][] = [];
  const y: number[] = [];
  for (let i = 0; i < n; i++) {
    const a = rng.normal();
    const b =
      correlation * a + Math.sqrt(Math.max(0, 1 - correlation * correlation)) * rng.normal();
    const c = rng.normal();
    rows.push([a, b, c]);
    y.push(1.5 + 2 * a - 1 * b + 0.4 * c + gauss(rng, 0, noise));
  }
  return makeTable(rows, y, ['x₁', 'x₂', 'x₃']);
}

// ---- manifolds ------------------------------------------------------------------------------

export const MANIFOLD_SETS = ['swiss', 'scurve', 'helix', 'sphere', 'clusters'] as const;
export type ManifoldSet = (typeof MANIFOLD_SETS)[number];

export const manifoldSetInfo: Record<
  ManifoldSet,
  { label: string; note: string; dimensions: number; colour: 'continuous' | 'classes' }
> = {
  swiss: {
    label: 'Swiss roll',
    note: 'A flat sheet rolled up in 3-D: two numbers describe every point, but straight-line distances shortcut through the roll.',
    dimensions: 3,
    colour: 'continuous',
  },
  scurve: {
    label: 'S-curve',
    note: 'A sheet bent into an S: gentler than the roll, so linear methods do less badly.',
    dimensions: 3,
    colour: 'continuous',
  },
  helix: {
    label: 'Helix',
    note: 'A spring in 3-D: really a one-dimensional curve.',
    dimensions: 3,
    colour: 'continuous',
  },
  sphere: {
    label: 'Sphere',
    note: 'The surface of a ball: it cannot be flattened without tearing or distorting something.',
    dimensions: 3,
    colour: 'continuous',
  },
  clusters: {
    label: '10-D clusters',
    note: 'Five groups in ten dimensions, each more spread along a few directions than others.',
    dimensions: 10,
    colour: 'classes',
  },
};

/**
 * Points on a curved surface in 3-D (or five clusters in 10-D), for the dimensionality-reduction labs.
 * `y` is the position along the surface scaled to [0, 1] (for colouring) or the cluster index.
 */
export function makeManifold(kind: ManifoldSet, n: number, noise: number, rng: Rng): Table {
  const rows: number[][] = [];
  const y: number[] = [];
  if (kind === 'clusters') {
    const centers = Array.from({ length: 5 }, () =>
      Array.from({ length: 10 }, () => rng.normal() * 3),
    );
    const scales = Array.from({ length: 5 }, () =>
      Array.from({ length: 10 }, () => 0.3 + rng.next() * 1.2),
    );
    for (let i = 0; i < n; i++) {
      const c = i % 5;
      rows.push(
        centers[c]!.map((m, j) => m + rng.normal() * (scales[c]![j] as number) * (0.6 + noise)),
      );
      y.push(c);
    }
    return makeTable(
      rows,
      y,
      Array.from({ length: 10 }, (_, j) => `x${j + 1}`),
      ['A', 'B', 'C', 'D', 'E'],
    );
  }
  for (let i = 0; i < n; i++) {
    const u = rng.next();
    const v = rng.next();
    let p: number[];
    switch (kind) {
      case 'swiss': {
        const t = 1.5 * Math.PI * (1 + 2 * u);
        p = [t * Math.cos(t), 21 * v, t * Math.sin(t)];
        break;
      }
      case 'scurve': {
        const t = 3 * Math.PI * (u - 0.5);
        p = [Math.sin(t), 2 * v, Math.sign(t) * (Math.cos(t) - 1)];
        break;
      }
      case 'helix': {
        const t = 4 * Math.PI * u;
        p = [Math.cos(t), Math.sin(t), t / 4];
        break;
      }
      default: {
        const phi = Math.acos(1 - 2 * u);
        const theta = 2 * Math.PI * v;
        p = [Math.sin(phi) * Math.cos(theta), Math.sin(phi) * Math.sin(theta), Math.cos(phi)];
      }
    }
    const scale = kind === 'swiss' ? 6 : 1;
    rows.push(p.map((q) => q + rng.normal() * noise * scale * 0.2));
    y.push(kind === 'sphere' ? (1 - Math.cos(Math.acos(1 - 2 * u))) / 2 : u);
  }
  return makeTable(rows, y, ['x', 'y', 'z']);
}

// ---- splitting and scaling ------------------------------------------------------------------

export function subset(t: Table, indices: ArrayLike<number>): Table {
  const d = t.x.cols;
  const x = new Matrix(indices.length, d);
  const y = new Float64Array(indices.length);
  for (let k = 0; k < indices.length; k++) {
    const i = indices[k] as number;
    x.data.set(t.x.data.subarray(i * d, (i + 1) * d), k * d);
    y[k] = t.y[i] as number;
  }
  return { x, y, names: t.names, ...(t.classes ? { classes: t.classes } : {}) };
}

/**
 * Chooses which rows go to the test set: a boolean mask over the rows. With `stratify` each class
 * keeps its share in both halves.
 */
export function splitMask(t: Table, testFraction: number, rng: Rng, stratify = false): Uint8Array {
  const n = tableSize(t);
  const mask = new Uint8Array(n);
  if (stratify) {
    const byClass = new Map<number, number[]>();
    for (let i = 0; i < n; i++)
      byClass.set(t.y[i] as number, [...(byClass.get(t.y[i] as number) ?? []), i]);
    for (const group of byClass.values()) {
      const order = permutation(group.length, rng);
      const k = Math.round(group.length * testFraction);
      order.forEach((o, pos) => {
        if (pos < k) mask[group[o] as number] = 1;
      });
    }
  } else {
    const order = permutation(n, rng);
    const k = Math.round(n * testFraction);
    order.forEach((o, pos) => {
      if (pos < k) mask[o] = 1;
    });
  }
  return mask;
}

/** The rows of `t` whose mask entry equals `value` (1 = test rows, 0 = train rows). */
export function tableRows(t: Table, mask: ArrayLike<number>, value: 0 | 1): Table {
  const idx: number[] = [];
  for (let i = 0; i < tableSize(t); i++) if (mask[i] === value) idx.push(i);
  return subset(t, idx);
}

/** Random split; with `stratify` each class keeps its share in both halves. */
export function trainTestSplit(
  t: Table,
  testFraction: number,
  rng: Rng,
  stratify = false,
): { train: Table; test: Table } {
  const mask = splitMask(t, testFraction, rng, stratify);
  return { train: tableRows(t, mask, 0), test: tableRows(t, mask, 1) };
}

/** Index sets for k-fold cross-validation: `folds[i]` is the validation part of fold i. */
export function kFold(n: number, k: number, rng: Rng): { train: number[]; validation: number[] }[] {
  if (k < 2 || k > n) throw new RangeError('k must be between 2 and n');
  const order = Array.from(permutation(n, rng));
  return Array.from({ length: k }, (_, f) => {
    const validation = order.filter((_, pos) => pos % k === f);
    const train = order.filter((_, pos) => pos % k !== f);
    return { train, validation };
  });
}

/** Zero-mean, unit-variance scaling learned on one table and applied to any other. */
export class Standardizer {
  mean: Float64Array = new Float64Array(0);
  scale: Float64Array = new Float64Array(0);

  fit(x: Matrix): this {
    this.mean = columnMeans(x);
    this.scale = Float64Array.from({ length: x.cols }, (_, j) => {
      const col = Array.from({ length: x.rows }, (_, i) => x.get(i, j));
      const s = std(col);
      return s > 1e-12 ? s : 1;
    });
    return this;
  }

  transform(x: Matrix): Matrix {
    const out = new Matrix(x.rows, x.cols);
    for (let i = 0; i < x.rows; i++)
      for (let j = 0; j < x.cols; j++)
        out.set(i, j, (x.get(i, j) - (this.mean[j] as number)) / (this.scale[j] as number));
    return out;
  }

  inverse(x: Matrix): Matrix {
    const out = new Matrix(x.rows, x.cols);
    for (let i = 0; i < x.rows; i++)
      for (let j = 0; j < x.cols; j++)
        out.set(i, j, x.get(i, j) * (this.scale[j] as number) + (this.mean[j] as number));
    return out;
  }
}

/** Bounding box of a table's first two features, padded: the window a plane plot should show. */
export function plotBounds(
  x: Matrix,
  pad = 0.15,
): { x0: number; x1: number; y0: number; y1: number } {
  let x0 = Infinity;
  let x1 = -Infinity;
  let y0 = Infinity;
  let y1 = -Infinity;
  for (let i = 0; i < x.rows; i++) {
    x0 = Math.min(x0, x.get(i, 0));
    x1 = Math.max(x1, x.get(i, 0));
    if (x.cols > 1) {
      y0 = Math.min(y0, x.get(i, 1));
      y1 = Math.max(y1, x.get(i, 1));
    }
  }
  if (x.cols < 2) {
    y0 = -1;
    y1 = 1;
  }
  const px = (x1 - x0 || 1) * pad;
  const py = (y1 - y0 || 1) * pad;
  return { x0: x0 - px, x1: x1 + px, y0: y0 - py, y1: y1 + py };
}
