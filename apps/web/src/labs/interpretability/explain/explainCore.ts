import {
  GradientBoostingRegressor,
  LinearRegression,
  RandomForestRegressor,
  DecisionTreeRegressor,
  iceCurves,
  kernelShap,
  lime,
  permutationImportance,
  shapleyExact,
  type Attribution,
  type Model,
} from '@sim/classic';
import { Matrix, r2 } from '@sim/core';
import { CLASSIC_DATASETS } from '@sim/core/datasets';
import type { ClassicDataset } from '@sim/core/datasets';

export type ExplainModel = 'linear' | 'tree' | 'forest' | 'boosting';

export const MODEL_LABELS: Record<ExplainModel, string> = {
  linear: 'Linear regression',
  tree: 'One decision tree (depth 4)',
  forest: 'Random forest',
  boosting: 'Gradient boosting',
};

export interface ExplainScenario {
  model: ExplainModel;
  /** Index of the test patient explained in detail. */
  instance: number;
  /** Feature shown in the partial-dependence panel. */
  feature: number;
  limeSamples: number;
  limeSeed: number;
}

export const DEFAULT_EXPLAIN: ExplainScenario = {
  model: 'forest',
  instance: 4,
  feature: 2,
  limeSamples: 500,
  limeSeed: 1,
};

const ds = CLASSIC_DATASETS.diabetes as ClassicDataset;
export const FEATURES = ds.features;

/** First 300 rows train the model, the remaining 142 are the patients to explain. */
const split = () => {
  const all = Matrix.from(ds.rows);
  const take = (from: number, to: number) =>
    new Matrix(to - from, all.cols, all.data.slice(from * all.cols, to * all.cols));
  return {
    xTrain: take(0, 300),
    yTrain: ds.target.slice(0, 300),
    xTest: take(300, ds.rows.length),
    yTest: ds.target.slice(300),
  };
};

export interface ExplainRun {
  scenario: ExplainScenario;
  testR2: number;
  instanceValues: number[];
  actual: number;
  shap: Attribution;
  kernel: Attribution;
  lime: { coefficients: number[]; intercept: number; fidelity: number; contributions: number[] };
  /** Permutation importance on the test set, and mean |SHAP| over 12 test patients. */
  permutation: number[];
  meanAbsShap: number[];
  ice: { grid: number[]; curves: number[][]; average: number[] };
  /** For linear models, coefficient x (value - mean): what SHAP must equal. */
  linearContributions: number[] | null;
}

const modelCache = new Map<string, { predict: Model; r2: number; coef: Float64Array | null }>();

function fitModel(kind: ExplainModel) {
  const hit = modelCache.get(kind);
  if (hit) return hit;
  const { xTrain, yTrain, xTest, yTest } = split();
  let predict: Model;
  let coef: Float64Array | null = null;
  if (kind === 'linear') {
    const m = new LinearRegression().fit(xTrain, yTrain);
    coef = m.coef;
    predict = (x) => m.predict(x);
  } else if (kind === 'tree') {
    const m = new DecisionTreeRegressor({ maxDepth: 4, minSamplesLeaf: 5 }).fit(xTrain, yTrain);
    predict = (x) => m.predict(x);
  } else if (kind === 'forest') {
    const m = new RandomForestRegressor({
      trees: 60,
      maxDepth: 7,
      minSamplesLeaf: 4,
      maxFeatures: 5,
      seed: 2,
    }).fit(xTrain, yTrain);
    predict = (x) => m.predict(x);
  } else {
    const m = new GradientBoostingRegressor({ rounds: 80, learningRate: 0.08, depth: 2 }).fit(
      xTrain,
      yTrain,
    );
    predict = (x) => m.predict(x);
  }
  const out = { predict, r2: r2(yTest, predict(xTest)), coef };
  modelCache.set(kind, out);
  return out;
}

const background = (): Matrix => {
  const { xTrain } = split();
  const b = new Matrix(30, xTrain.cols);
  for (let i = 0; i < 30; i++)
    b.data.set(
      xTrain.data.subarray(i * 10 * xTrain.cols, (i * 10 + 1) * xTrain.cols),
      i * xTrain.cols,
    );
  return b;
};

const globalCache = new Map<ExplainModel, { permutation: number[]; meanAbsShap: number[] }>();
const shapCache = new Map<string, { shap: Attribution; kernel: Attribution }>();

function globals(kind: ExplainModel) {
  const hit = globalCache.get(kind);
  if (hit) return hit;
  const { xTest, yTest } = split();
  const { predict } = fitModel(kind);
  const bg = background();
  const permutation = permutationImportance(predict, xTest, yTest, r2, 5, 1).importance;
  const meanAbsShap = new Array<number>(xTest.cols).fill(0);
  const many = 10;
  for (let i = 0; i < many; i++) {
    const r = Array.from(xTest.data.subarray(i * xTest.cols, (i + 1) * xTest.cols));
    shapleyExact(predict, r, bg).values.forEach(
      (v, j) => (meanAbsShap[j] = (meanAbsShap[j] as number) + Math.abs(v) / many),
    );
  }
  const out = { permutation, meanAbsShap };
  globalCache.set(kind, out);
  return out;
}

/** Everything that depends on the instance, cached because exact Shapley values are the slow part. */
function attributions(kind: ExplainModel, instance: number) {
  const key = `${kind}|${instance}`;
  const hit = shapCache.get(key);
  if (hit) return hit;
  const { xTest } = split();
  const { predict } = fitModel(kind);
  const bg = background();
  const row = Array.from(xTest.data.subarray(instance * xTest.cols, (instance + 1) * xTest.cols));
  const out = { shap: shapleyExact(predict, row, bg), kernel: kernelShap(predict, row, bg, 40, 1) };
  shapCache.set(key, out);
  return out;
}

export function limeOf(kind: ExplainModel, instance: number, samples: number, seed: number) {
  const { xTest, xTrain } = split();
  const { predict } = fitModel(kind);
  const row = Array.from(xTest.data.subarray(instance * xTest.cols, (instance + 1) * xTest.cols));
  const fit = lime(predict, row, xTrain, { samples, width: 0.5, kernelWidth: 1.5, ridge: 1, seed });
  // LIME coefficients are per unit of the feature: multiply by the distance from the average to compare with SHAP
  const means = Array.from({ length: xTrain.cols }, (_, j) => {
    let m = 0;
    for (let i = 0; i < xTrain.rows; i++) m += xTrain.get(i, j) / xTrain.rows;
    return m;
  });
  return {
    ...fit,
    contributions: fit.coefficients.map((c, j) => c * ((row[j] as number) - (means[j] as number))),
  };
}

export function runExplain(s: ExplainScenario): ExplainRun {
  const { xTest, yTest } = split();
  const { predict, r2: testR2, coef } = fitModel(s.model);
  const bg = background();
  const row = Array.from(
    xTest.data.subarray(s.instance * xTest.cols, (s.instance + 1) * xTest.cols),
  );
  const { shap, kernel } = attributions(s.model, s.instance);
  const g = globals(s.model);
  const bgMean = Array.from({ length: bg.cols }, (_, j) => {
    let m = 0;
    for (let i = 0; i < bg.rows; i++) m += bg.get(i, j) / bg.rows;
    return m;
  });
  return {
    scenario: s,
    testR2,
    instanceValues: row,
    actual: yTest[s.instance] as number,
    shap,
    kernel,
    lime: limeOf(s.model, s.instance, s.limeSamples, s.limeSeed),
    permutation: g.permutation,
    meanAbsShap: g.meanAbsShap,
    ice: iceCurves(predict, xTest, s.feature, 20, 40),
    linearContributions: coef
      ? Array.from(coef, (c, j) => c * ((row[j] as number) - (bgMean[j] as number)))
      : null,
  };
}

export const topK = (v: readonly number[], k: number): number[] =>
  Array.from(v.keys())
    .sort((a, b) => Math.abs(v[b] as number) - Math.abs(v[a] as number))
    .slice(0, k);
