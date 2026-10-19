import {
  LinearRegression,
  RandomForestRegressor,
  alphaMax,
  permutationImportance,
  regularizationPath,
} from '@sim/classic';
import { Matrix, createRng, r2 } from '@sim/core';

export type SelectionMethod = 'filter' | 'forward' | 'lasso' | 'forest' | 'permutation';

export const METHOD_LABELS: Record<SelectionMethod, string> = {
  filter: 'Filter: correlation with the target',
  forward: 'Wrapper: forward selection',
  lasso: 'Embedded: lasso path',
  forest: 'Forest impurity importance',
  permutation: 'Permutation importance (forest)',
};

export const METHOD_NOTES: Record<SelectionMethod, string> = {
  filter:
    'Scores every feature on its own, before any model: fast, but blind to redundancy and interactions.',
  forward:
    'Adds, one at a time, the feature that most improves cross-validated R²: accounts for what is already chosen.',
  lasso:
    'An L1 penalty zeroes coefficients; features enter the model in order as the penalty is relaxed.',
  forest:
    'How much each feature reduced impurity across the trees of a random forest: cheap, but biased towards features with many split points.',
  permutation:
    'How much the forest score on held-out rows drops when the feature is shuffled: honest, and costs one prediction pass per feature.',
};

export interface FeaturesScenario {
  n: number;
  /** Noise added to the copies of the informative features (smaller = more redundant). */
  copyNoise: number;
  method: SelectionMethod;
  /** How many top-ranked features the model may use. */
  keep: number;
  seed: number;
}

export const DEFAULT_FEATURES: FeaturesScenario = {
  n: 150,
  copyNoise: 0.3,
  method: 'filter',
  keep: 4,
  seed: 9,
};

/** Layout of the 20 columns: 4 informative, 4 noisy copies of them, 12 pure noise. */
export const NAMES = [
  ...['signal A', 'signal B', 'signal C', 'signal D'],
  ...['copy of A', 'copy of B', 'copy of C', 'copy of D'],
  ...Array.from({ length: 12 }, (_, i) => `noise ${i + 1}`),
];
export const GROUP = NAMES.map((_, j) => (j < 4 ? j : j < 8 ? j - 4 : -1));
export const COEF = [3, 2, -2, 1];

export interface FeaturesRun {
  scenario: FeaturesScenario;
  /** A score per feature (higher = more important) and the resulting order, best first. */
  scores: number[];
  ranking: number[];
  chosen: number[];
  /** R² on held-out rows of a linear model that uses only the chosen features. */
  testR2: number;
  /** R² of a linear model with all 20 features and of an oracle that knows the four true signals. */
  allR2: number;
  oracleR2: number;
  /** Test R² when the top 1, 2, ... 12 features are used. */
  curve: number[];
  /** Chosen features that are pure noise. */
  noisePicked: number;
  /** Distinct true signals covered by the chosen features (a signal and its copy count once). */
  signalsCovered: number;
  /** Chosen features that duplicate another chosen feature's signal. */
  redundant: number;
}

function makeData(s: FeaturesScenario) {
  const rng = createRng(s.seed);
  const n = s.n * 2;
  const x = new Matrix(n, 20);
  const y: number[] = [];
  for (let i = 0; i < n; i++) {
    const signal = Array.from({ length: 4 }, () => rng.normal());
    for (let k = 0; k < 4; k++) {
      x.set(i, k, signal[k] as number);
      x.set(i, 4 + k, (signal[k] as number) + rng.normal() * s.copyNoise);
    }
    for (let k = 8; k < 20; k++) x.set(i, k, rng.normal());
    y.push(signal.reduce((a, v, k) => a + (COEF[k] as number) * v, 0) + rng.normal() * 0.5);
  }
  return { x, y };
}

const takeRows = (x: Matrix, from: number, to: number): Matrix =>
  new Matrix(to - from, x.cols, x.data.slice(from * x.cols, to * x.cols));

const takeCols = (x: Matrix, cols: readonly number[]): Matrix => {
  const out = new Matrix(x.rows, cols.length);
  for (let i = 0; i < x.rows; i++) cols.forEach((c, k) => out.set(i, k, x.get(i, c)));
  return out;
};

function corr(a: ArrayLike<number>, b: ArrayLike<number>): number {
  const n = a.length;
  const ma = Array.from(a).reduce((s, v) => s + v, 0) / n;
  const mb = Array.from(b).reduce((s, v) => s + v, 0) / n;
  let sab = 0;
  let saa = 0;
  let sbb = 0;
  for (let i = 0; i < n; i++) {
    sab += ((a[i] as number) - ma) * ((b[i] as number) - mb);
    saa += ((a[i] as number) - ma) ** 2;
    sbb += ((b[i] as number) - mb) ** 2;
  }
  return sab / Math.sqrt(saa * sbb || 1);
}

function cvR2(x: Matrix, y: number[], cols: readonly number[], folds = 5): number {
  const n = x.rows;
  const pred = new Array<number>(n).fill(0);
  const xs = takeCols(x, cols);
  const pick = (idx: number[]) => {
    const m = new Matrix(idx.length, xs.cols);
    idx.forEach((r, i) =>
      m.data.set(xs.data.subarray(r * xs.cols, (r + 1) * xs.cols), i * xs.cols),
    );
    return m;
  };
  for (let f = 0; f < folds; f++) {
    const test: number[] = [];
    const train: number[] = [];
    for (let i = 0; i < n; i++) (i % folds === f ? test : train).push(i);
    const model = new LinearRegression().fit(
      pick(train),
      train.map((i) => y[i] as number),
    );
    const p = model.predict(pick(test));
    test.forEach((r, i) => (pred[r] = p[i] as number));
  }
  return r2(y, pred);
}

export function runFeatures(s: FeaturesScenario): FeaturesRun {
  const { x: all, y: yAll } = makeData(s);
  const xTrain = takeRows(all, 0, s.n);
  const xTest = takeRows(all, s.n, 2 * s.n);
  const yTrain = yAll.slice(0, s.n);
  const yTest = yAll.slice(s.n);
  let scores: number[];
  if (s.method === 'filter') {
    scores = Array.from({ length: 20 }, (_, j) =>
      Math.abs(
        corr(
          Array.from({ length: s.n }, (_, i) => xTrain.get(i, j)),
          yTrain,
        ),
      ),
    );
  } else if (s.method === 'forward') {
    const chosen: number[] = [];
    scores = new Array<number>(20).fill(0);
    for (let step = 0; step < 20; step++) {
      let best = -1;
      let bestScore = -Infinity;
      for (let j = 0; j < 20; j++) {
        if (chosen.includes(j)) continue;
        const sc = cvR2(xTrain, yTrain, [...chosen, j]);
        if (sc > bestScore) {
          bestScore = sc;
          best = j;
        }
      }
      chosen.push(best);
      scores[best] = 20 - step;
    }
  } else if (s.method === 'lasso') {
    const top = alphaMax(xTrain, yTrain);
    const path = regularizationPath(xTrain, yTrain, { count: 80, ratio: 1e-3 });
    scores = new Array<number>(20).fill(0);
    for (let j = 0; j < 20; j++) {
      const entry = path.coefs.findIndex((c) => Math.abs(c[j] as number) > 1e-9);
      scores[j] =
        entry < 0
          ? 0
          : 80 - entry + Math.abs((path.coefs[79] as Float64Array)[j] as number) / (top + 1e-9);
    }
  } else {
    const forest = new RandomForestRegressor({
      trees: 60,
      seed: s.seed,
      maxFeatures: 7,
      minSamplesLeaf: 2,
    }).fit(xTrain, yTrain);
    if (s.method === 'forest') scores = [...forest.importances];
    else
      scores = permutationImportance(
        (m) => forest.predict(m),
        xTest,
        yTest,
        r2,
        5,
        s.seed,
      ).importance;
  }
  const ranking = Array.from({ length: 20 }, (_, j) => j).sort(
    (a, b) => (scores[b] as number) - (scores[a] as number) || a - b,
  );
  const chosen = ranking.slice(0, s.keep);
  const fitTest = (cols: readonly number[]) => {
    const m = new LinearRegression().fit(takeCols(xTrain, cols), yTrain);
    return r2(yTest, m.predict(takeCols(xTest, cols)));
  };
  const groups = new Set(chosen.map((j) => GROUP[j] as number).filter((g) => g >= 0));
  const used = new Set<number>();
  let redundant = 0;
  for (const j of chosen) {
    const g = GROUP[j] as number;
    if (g < 0) continue;
    if (used.has(g)) redundant++;
    used.add(g);
  }
  return {
    scenario: s,
    scores,
    ranking,
    chosen,
    testR2: fitTest(chosen),
    allR2: fitTest(Array.from({ length: 20 }, (_, j) => j)),
    oracleR2: fitTest([0, 1, 2, 3]),
    curve: Array.from({ length: 12 }, (_, k) => fitTest(ranking.slice(0, k + 1))),
    noisePicked: chosen.filter((j) => (GROUP[j] as number) < 0).length,
    signalsCovered: groups.size,
    redundant,
  };
}
