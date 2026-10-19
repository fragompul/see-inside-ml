import { Matrix, createRng, solve } from '@sim/core';

/**
 * Bias, variance and generalisation, measured by brute force: draw many training sets from a known
 * world, fit the same model to each, and compare the fits with the truth. Because the truth is known,
 * the expected test error splits exactly into bias squared, variance and irreducible noise.
 */
export type WorldId = 'wave' | 'steps' | 'line';

export const WORLDS: Record<WorldId, { label: string; note: string; f: (x: number) => number }> = {
  wave: {
    label: 'A smooth wave',
    note: 'y = sin(2πx): one full smooth oscillation.',
    f: (x) => Math.sin(2 * Math.PI * x),
  },
  steps: {
    label: 'Sharp steps',
    note: 'A staircase: flat stretches with abrupt jumps, hard for smooth curves.',
    f: (x) => (x < 0.3 ? -0.8 : x < 0.55 ? 0.7 : x < 0.8 ? -0.2 : 0.9),
  },
  line: {
    label: 'A straight line',
    note: 'y = 1.6x - 0.8: the simplest possible truth.',
    f: (x) => 1.6 * x - 0.8,
  },
};

/** Legendre polynomials on [0, 1] (well conditioned even at high degree). */
export function legendreFeatures(xs: ArrayLike<number>, count: number): Matrix {
  const n = xs.length;
  const out = new Matrix(n, count);
  for (let i = 0; i < n; i++) {
    const t = 2 * (xs[i] as number) - 1;
    let prev = 1;
    let cur = t;
    out.set(i, 0, 1);
    if (count > 1) out.set(i, 1, t);
    for (let k = 2; k < count; k++) {
      const next = ((2 * k - 1) * t * cur - (k - 1) * prev) / k;
      out.set(i, k, next);
      prev = cur;
      cur = next;
    }
  }
  return out;
}

/**
 * Least squares with `p` Legendre features. For p <= n it is ordinary (ridge) regression; once there
 * are more features than points it returns the minimum-norm solution that interpolates the data,
 * which is what gradient descent from zero converges to.
 */
export function fitFeatures(x: ArrayLike<number>, y: ArrayLike<number>, p: number, ridge = 1e-8) {
  const phi = legendreFeatures(x, p);
  const n = phi.rows;
  let w: Float64Array;
  if (p <= n) {
    const a = new Matrix(p, p);
    const b = new Matrix(p, 1);
    for (let i = 0; i < p; i++) {
      for (let j = 0; j < p; j++) {
        let s = 0;
        for (let r = 0; r < n; r++) s += phi.get(r, i) * phi.get(r, j);
        a.set(i, j, s + (i === j ? ridge : 0));
      }
      let s = 0;
      for (let r = 0; r < n; r++) s += phi.get(r, i) * (y[r] as number);
      b.set(i, 0, s);
    }
    w = solve(a, b).data;
  } else {
    // w = Phi^T (Phi Phi^T + eps I)^-1 y
    const k = new Matrix(n, n);
    for (let i = 0; i < n; i++)
      for (let j = 0; j < n; j++) {
        let s = 0;
        for (let c = 0; c < p; c++) s += phi.get(i, c) * phi.get(j, c);
        k.set(i, j, s + (i === j ? ridge : 0));
      }
    const alpha = solve(k, new Matrix(n, 1, Float64Array.from(y)));
    w = new Float64Array(p);
    for (let c = 0; c < p; c++) {
      let s = 0;
      for (let r = 0; r < n; r++) s += phi.get(r, c) * alpha.get(r, 0);
      w[c] = s;
    }
  }
  return {
    predict: (xs: ArrayLike<number>): number[] => {
      const f = legendreFeatures(xs, p);
      return Array.from({ length: xs.length }, (_, i) => {
        let s = 0;
        for (let c = 0; c < p; c++) s += f.get(i, c) * (w[c] as number);
        return s;
      });
    },
    coefficients: w,
  };
}

export interface BiasVarianceScenario {
  world: WorldId;
  /** Number of features (degree + 1). */
  features: number;
  /** Training points per dataset. */
  n: number;
  noise: number;
  /** Penalty on the coefficients (0 = essentially none). */
  ridge: number;
  datasets: number;
  seed: number;
}

export const DEFAULT_BV: BiasVarianceScenario = {
  world: 'wave',
  features: 4,
  n: 30,
  noise: 0.3,
  ridge: 0,
  datasets: 60,
  seed: 7,
};

export const GRID = Array.from({ length: 101 }, (_, i) => i / 100);

export interface BiasVarianceRun {
  scenario: BiasVarianceScenario;
  truth: number[];
  /** A few of the fitted curves (for the spaghetti plot) and the average over all datasets. */
  curves: number[][];
  meanCurve: number[];
  /** First training set (to draw the points). */
  sample: { x: number[]; y: number[] };
  bias2: number;
  variance: number;
  noiseVariance: number;
  /** Average test error on fresh noisy data, and average training error. */
  testError: number;
  trainError: number;
}

export function simulate(s: BiasVarianceScenario): BiasVarianceRun {
  const world = WORLDS[s.world].f;
  const rng = createRng(s.seed);
  const truth = GRID.map(world);
  const preds: number[][] = [];
  let train = 0;
  let sample: { x: number[]; y: number[] } = { x: [], y: [] };
  for (let r = 0; r < s.datasets; r++) {
    // evenly spread inputs with a random jitter inside each slot: no huge gaps, so the fits stay on the page
    const x = Array.from({ length: s.n }, (_, i) => (i + rng.next()) / s.n);
    const y = x.map((v) => world(v) + rng.normal() * s.noise);
    if (r === 0) sample = { x, y };
    const fit = fitFeatures(x, y, s.features, Math.max(s.ridge, 1e-8));
    preds.push(fit.predict(GRID));
    const fitted = fit.predict(x);
    train += fitted.reduce((a, f, i) => a + (f - (y[i] as number)) ** 2, 0) / s.n;
  }
  const R = preds.length;
  const meanCurve = GRID.map((_, g) => preds.reduce((a, p) => a + (p[g] as number), 0) / R);
  const bias2 =
    GRID.reduce((a, _, g) => a + ((meanCurve[g] as number) - (truth[g] as number)) ** 2, 0) /
    GRID.length;
  const variance =
    GRID.reduce(
      (a, _, g) =>
        a + preds.reduce((b, p) => b + ((p[g] as number) - (meanCurve[g] as number)) ** 2, 0) / R,
      0,
    ) / GRID.length;
  const noiseVariance = s.noise * s.noise;
  // expected error on a fresh noisy observation at a random x: exact decomposition
  const testError =
    GRID.reduce(
      (a, _, g) =>
        a + preds.reduce((b, p) => b + ((p[g] as number) - (truth[g] as number)) ** 2, 0) / R,
      0,
    ) /
      GRID.length +
    noiseVariance;
  return {
    scenario: s,
    truth,
    curves: preds.slice(0, 12),
    meanCurve,
    sample,
    bias2,
    variance,
    noiseVariance,
    testError,
    trainError: train / R,
  };
}

/** Error of the model family as the number of features grows: the classic U shape, and double descent beyond n. */
export function complexityCurve(
  base: BiasVarianceScenario,
  maxFeatures: number,
): { features: number; train: number; test: number }[] {
  const out: { features: number; train: number; test: number }[] = [];
  for (let p = 1; p <= maxFeatures; p++) {
    const r = simulate({ ...base, features: p, datasets: Math.min(base.datasets, 30) });
    out.push({ features: p, train: r.trainError, test: r.testError });
  }
  return out;
}

/** Test error as the training set grows, for a fixed model. */
export function learningCurve(
  base: BiasVarianceScenario,
  sizes: readonly number[],
): { n: number; train: number; test: number }[] {
  return sizes.map((n) => {
    const r = simulate({ ...base, n, datasets: Math.min(base.datasets, 30) });
    return { n, train: r.trainError, test: r.testError };
  });
}
