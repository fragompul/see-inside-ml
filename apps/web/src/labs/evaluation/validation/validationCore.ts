import { KnnClassifier } from '@sim/classic';
import { Matrix, createRng, makeClassification, type Rng } from '@sim/core';

/**
 * Three ways validation goes right or wrong, each simulated many times so that the *distribution* of
 * what you would have measured is visible:
 *   estimators  - how well do hold-out and k-fold estimate the real accuracy, and how noisy are they?
 *   selection   - what happens to the score of the best of M candidates picked on the same data?
 *   leakage     - what happens if a data-driven step (feature selection) sees the validation rows?
 */
export type ValidationKind = 'estimators' | 'selection' | 'leakage';

export interface ValidationScenario {
  kind: ValidationKind;
  seed: number;
  // estimators
  n: number;
  neighbors: number;
  // selection
  candidates: number;
  signal: number;
  // leakage
  samples: number;
  features: number;
  keep: number;
  repeats: number;
}

export const DEFAULT_VALIDATION: ValidationScenario = {
  kind: 'estimators',
  seed: 5,
  n: 100,
  neighbors: 1,
  candidates: 50,
  signal: 0,
  samples: 50,
  features: 500,
  keep: 10,
  repeats: 40,
};

// ---- helpers -----------------------------------------------------------------------------------

export const mean = (v: readonly number[]): number => v.reduce((a, b) => a + b, 0) / v.length;
export const std = (v: readonly number[]): number => {
  const m = mean(v);
  return Math.sqrt(v.reduce((a, b) => a + (b - m) ** 2, 0) / Math.max(v.length - 1, 1));
};

/** Random permutation of 0 .. n-1. */
export function shuffled(n: number, rng: Rng): number[] {
  const idx = Array.from({ length: n }, (_, i) => i);
  for (let i = n - 1; i > 0; i--) {
    const j = Math.floor(rng.next() * (i + 1));
    [idx[i], idx[j]] = [idx[j] as number, idx[i] as number];
  }
  return idx;
}

/** The `k` folds of a shuffled index list (as even as possible). */
export function kFolds(n: number, k: number, rng: Rng): number[][] {
  const idx = shuffled(n, rng);
  const folds: number[][] = Array.from({ length: k }, () => []);
  idx.forEach((v, i) => (folds[i % k] as number[]).push(v));
  return folds;
}

const rows = (x: Matrix, idx: readonly number[]): Matrix => {
  const out = new Matrix(idx.length, x.cols);
  idx.forEach((r, i) => out.data.set(x.data.subarray(r * x.cols, (r + 1) * x.cols), i * x.cols));
  return out;
};

const accuracyOf = (
  model: { predict(x: Matrix): Int32Array },
  x: Matrix,
  y: ArrayLike<number>,
): number => {
  const p = model.predict(x);
  let ok = 0;
  for (let i = 0; i < p.length; i++) if (p[i] === y[i]) ok++;
  return ok / p.length;
};

// ---- 1. estimators ------------------------------------------------------------------------------

export type EstimatorName = 'holdout 20%' | 'holdout 50%' | '5-fold' | '10-fold';
export const ESTIMATORS: readonly EstimatorName[] = [
  'holdout 20%',
  'holdout 50%',
  '5-fold',
  '10-fold',
];

export interface EstimatorStats {
  name: EstimatorName;
  /** One estimate per repetition. */
  estimates: number[];
  mean: number;
  std: number;
  /** Mean of (estimate - the real accuracy of the model trained on all of that repetition's data). */
  bias: number;
}

export interface EstimatorsRun {
  kind: 'estimators';
  scenario: ValidationScenario;
  stats: EstimatorStats[];
  /** The real accuracy (on 1500 fresh points) of the model trained on all n points, averaged over repetitions. */
  truth: number;
}

export function runEstimators(s: ValidationScenario): EstimatorsRun {
  const rng = createRng(s.seed);
  const reps = 60;
  const store = new Map<EstimatorName, number[]>(ESTIMATORS.map((e) => [e, []]));
  const truths: number[] = [];
  const fresh = makeClassification('moons', 1500, 0.35, createRng(s.seed + 1000));
  const fit = (x: Matrix, y: ArrayLike<number>) => new KnnClassifier({ k: s.neighbors }).fit(x, y);
  for (let r = 0; r < reps; r++) {
    const d = makeClassification('moons', s.n, 0.35, rng);
    const full = fit(d.x, d.y);
    truths.push(accuracyOf(full, fresh.x, fresh.y));
    const holdout = (frac: number) => {
      const idx = shuffled(s.n, rng);
      const cut = Math.round(s.n * frac);
      const test = idx.slice(0, cut);
      const train = idx.slice(cut);
      const m = fit(
        rows(d.x, train),
        train.map((i) => d.y[i] as number),
      );
      return accuracyOf(
        m,
        rows(d.x, test),
        test.map((i) => d.y[i] as number),
      );
    };
    const cv = (k: number) => {
      const folds = kFolds(s.n, k, rng);
      let correct = 0;
      for (const f of folds) {
        const inFold = new Set(f);
        const train = Array.from({ length: s.n }, (_, i) => i).filter((i) => !inFold.has(i));
        const m = fit(
          rows(d.x, train),
          train.map((i) => d.y[i] as number),
        );
        correct +=
          accuracyOf(
            m,
            rows(d.x, f),
            f.map((i) => d.y[i] as number),
          ) * f.length;
      }
      return correct / s.n;
    };
    (store.get('holdout 20%') as number[]).push(holdout(0.2));
    (store.get('holdout 50%') as number[]).push(holdout(0.5));
    (store.get('5-fold') as number[]).push(cv(5));
    (store.get('10-fold') as number[]).push(cv(10));
  }
  const truth = mean(truths);
  return {
    kind: 'estimators',
    scenario: s,
    truth,
    stats: ESTIMATORS.map((name) => {
      const estimates = store.get(name) as number[];
      return {
        name,
        estimates,
        mean: mean(estimates),
        std: std(estimates),
        bias: mean(estimates.map((e, i) => e - (truths[i] as number))),
      };
    }),
  };
}

// ---- 2. selection -------------------------------------------------------------------------------

export interface SelectionRun {
  kind: 'selection';
  scenario: ValidationScenario;
  /** Accuracy of the winning candidate on the data it was picked on, per repetition. */
  winnerScores: number[];
  /** Accuracy of that same candidate on a separate, untouched test set. */
  testScores: number[];
  optimism: number;
  winnerMean: number;
  testMean: number;
  /** The same quantities for the best of one candidate (no selection at all). */
  baseline: number;
}

/**
 * Each candidate is a one-feature threshold rule on a table of `features` columns, of which `signal`
 * share are weakly informative. The winner on a validation set of 60 rows is scored on a separate test
 * set of 2000 rows.
 */
export function runSelection(s: ValidationScenario): SelectionRun {
  const rng = createRng(s.seed);
  const reps = 100;
  const nVal = 60;
  const nTest = 2000;
  const winner: number[] = [];
  const test: number[] = [];
  for (let r = 0; r < reps; r++) {
    // a candidate is (feature direction): score(x) = sign * x_j; the effect of feature j on the label is eff[j]
    const eff = Array.from(
      { length: s.candidates },
      () => (rng.next() < s.signal ? 0.5 : 0) * (rng.next() < 0.5 ? -1 : 1),
    );
    const draw = (n: number) => {
      const y = Array.from({ length: n }, () => (rng.next() < 0.5 ? 1 : 0));
      const x = y.map((label) => eff.map((e) => rng.normal() + e * (label ? 1 : -1)));
      return { x, y };
    };
    const val = draw(nVal);
    // each candidate classifies by whether its feature is above its median-ish threshold 0 (direction chosen by the sign of its mean difference)
    const scoreOf = (set: { x: number[][]; y: number[] }, j: number, dir: number) =>
      set.x.reduce(
        (a, row, i) => a + ((dir * (row[j] as number) > 0 ? 1 : 0) === set.y[i] ? 1 : 0),
        0,
      ) / set.y.length;
    let bestScore = -1;
    let best = { j: 0, dir: 1 };
    for (let j = 0; j < s.candidates; j++) {
      // the direction is fitted on the validation set too, as in practice
      const up = scoreOf(val, j, 1);
      const down = scoreOf(val, j, -1);
      const dir = up >= down ? 1 : -1;
      const sc = Math.max(up, down);
      if (sc > bestScore) {
        bestScore = sc;
        best = { j, dir };
      }
    }
    winner.push(bestScore);
    const fresh = draw(nTest);
    test.push(scoreOf(fresh, best.j, best.dir));
  }
  return {
    kind: 'selection',
    scenario: s,
    winnerScores: winner,
    testScores: test,
    optimism: mean(winner) - mean(test),
    winnerMean: mean(winner),
    testMean: mean(test),
    baseline: 0.5,
  };
}

// ---- 3. leakage ----------------------------------------------------------------------------------

export interface LeakageRun {
  kind: 'leakage';
  scenario: ValidationScenario;
  /** Cross-validated accuracy when features were chosen using all the rows first (the mistake). */
  leaky: number[];
  /** ... and when the choice is made again inside every training fold (the right way). */
  honest: number[];
  leakyMean: number;
  honestMean: number;
  /** Accuracy on an independent test set of the model that the honest procedure builds on all the data. */
  independent: number;
}

/** Indices of the `keep` columns of `x` most correlated (in absolute value) with the labels of the rows `idx`. */
function topFeatures(x: number[][], y: number[], idx: readonly number[], keep: number): number[] {
  const p = (x[0] as number[]).length;
  const n = idx.length;
  const my = idx.reduce((a, i) => a + (y[i] as number), 0) / n;
  const scores = Array.from({ length: p }, (_, j) => {
    let mx = 0;
    for (const i of idx) mx += (x[i] as number[])[j]! / n;
    let sxy = 0;
    let sxx = 0;
    for (const i of idx) {
      const dx = (x[i] as number[])[j]! - mx;
      sxy += dx * ((y[i] as number) - my);
      sxx += dx * dx;
    }
    return Math.abs(sxy) / Math.sqrt(sxx || 1);
  });
  return Array.from({ length: p }, (_, j) => j)
    .sort((a, b) => (scores[b] as number) - (scores[a] as number))
    .slice(0, keep);
}

/** Nearest-centroid classifier on the chosen columns, trained on `train`, scored on `test`. */
function centroidAccuracy(
  x: number[][],
  y: number[],
  train: readonly number[],
  test: readonly number[],
  cols: readonly number[],
): number {
  const c = [0, 1].map((cls) => {
    const members = train.filter((i) => y[i] === cls);
    return cols.map(
      (j) => members.reduce((a, i) => a + (x[i] as number[])[j]!, 0) / Math.max(members.length, 1),
    );
  });
  let ok = 0;
  for (const i of test) {
    const d = c.map((cent) =>
      cols.reduce((a, j, k) => a + ((x[i] as number[])[j]! - (cent[k] as number)) ** 2, 0),
    );
    if ((d[1]! < d[0]! ? 1 : 0) === y[i]) ok++;
  }
  return ok / test.length;
}

export function runLeakage(s: ValidationScenario): LeakageRun {
  const rng = createRng(s.seed);
  const leaky: number[] = [];
  const honest: number[] = [];
  let independent = 0;
  for (let r = 0; r < s.repeats; r++) {
    // pure noise features (a few weakly informative ones when signal > 0)
    const informative = Math.round(s.signal * 5);
    const make = (n: number) => {
      const y = Array.from({ length: n }, () => (rng.next() < 0.5 ? 1 : 0));
      const x = y.map((label) =>
        Array.from(
          { length: s.features },
          (_, j) => rng.normal() + (j < informative ? 0.6 * (label ? 1 : -1) : 0),
        ),
      );
      return { x, y };
    };
    const d = make(s.samples);
    const all = Array.from({ length: s.samples }, (_, i) => i);
    const folds = kFolds(s.samples, 5, rng);
    // the mistake: choose the columns once, looking at every row
    const colsAll = topFeatures(d.x, d.y, all, s.keep);
    let leakyOk = 0;
    let honestOk = 0;
    for (const f of folds) {
      const inFold = new Set(f);
      const train = all.filter((i) => !inFold.has(i));
      leakyOk += centroidAccuracy(d.x, d.y, train, f, colsAll) * f.length;
      const cols = topFeatures(d.x, d.y, train, s.keep);
      honestOk += centroidAccuracy(d.x, d.y, train, f, cols) * f.length;
    }
    leaky.push(leakyOk / s.samples);
    honest.push(honestOk / s.samples);
    if (r < 10) {
      const t = make(500);
      const cols = topFeatures(d.x, d.y, all, s.keep);
      const merged = { x: [...d.x, ...t.x], y: [...d.y, ...t.y] };
      independent +=
        centroidAccuracy(
          merged.x,
          merged.y,
          all,
          Array.from({ length: 500 }, (_, i) => s.samples + i),
          cols,
        ) / 10;
    }
  }
  return {
    kind: 'leakage',
    scenario: s,
    leaky,
    honest,
    leakyMean: mean(leaky),
    honestMean: mean(honest),
    independent,
  };
}

export type ValidationRun = EstimatorsRun | SelectionRun | LeakageRun;

export function runValidation(s: ValidationScenario): ValidationRun {
  if (s.kind === 'estimators') return runEstimators(s);
  if (s.kind === 'selection') return runSelection(s);
  return runLeakage(s);
}
