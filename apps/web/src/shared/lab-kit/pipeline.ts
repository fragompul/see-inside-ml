import {
  accuracy,
  confusionMatrix,
  createRng,
  logLoss,
  makeClassification,
  makeRegression,
  mae,
  mse,
  plotBounds,
  r2,
  rmse,
  gridPoints,
  Matrix,
  tableRows,
  splitMask,
  type Classifier,
  type ClassificationSet,
  type RegressionSet,
  type Regressor,
  type Table,
} from '@sim/core';

/**
 * The pure part of the classical-model labs: build a dataset, split it, fit a model, score it and
 * paint its decision regions. The React components only call these functions, and the tests of every
 * lab's experiments call the same ones, so what the page claims is exactly what is measured.
 */

export interface DataSetup {
  n: number;
  noise: number;
  seed: number;
  /** Share of the points held out for testing (0 – 0.6). */
  testFraction: number;
}

export interface ClassificationSetup extends DataSetup {
  dataset: ClassificationSet;
}

export interface RegressionSetup extends DataSetup {
  dataset: RegressionSet;
}

export const DEFAULT_DATA: DataSetup = { n: 200, noise: 0.25, seed: 4, testFraction: 0.3 };

export interface SplitData {
  /** All the points (train + test) as they were before splitting. */
  all: Table;
  /** 1 for the rows of `all` that are held out for testing. */
  testMask: Uint8Array;
  train: Table;
  test: Table;
}

/** Builds and splits a classification dataset, or splits `custom` points the user edited. */
export function buildClassification(setup: ClassificationSetup, custom?: Table | null): SplitData {
  const all =
    custom ?? makeClassification(setup.dataset, setup.n, setup.noise, createRng(setup.seed));
  return splitData(all, setup, true);
}

export function buildRegression(setup: RegressionSetup, custom?: Table | null): SplitData {
  const all = custom ?? makeRegression(setup.dataset, setup.n, setup.noise, createRng(setup.seed));
  return splitData(all, setup, false);
}

export function splitData(all: Table, setup: DataSetup, stratify: boolean): SplitData {
  // tiny tables (a user who erased most points) cannot be split: everything trains
  const tiny = all.x.rows < 6 || setup.testFraction <= 0;
  const testMask = tiny
    ? new Uint8Array(all.x.rows)
    : splitMask(all, setup.testFraction, createRng(setup.seed + 1), stratify);
  return { all, testMask, train: tableRows(all, testMask, 0), test: tableRows(all, testMask, 1) };
}

export interface ClassifierScores {
  trainAccuracy: number;
  testAccuracy: number | null;
  trainLoss: number | null;
  testLoss: number | null;
  /** Rows = true class, columns = predicted class, on the test set (or on the training set if there is none). */
  confusion: number[][];
}

export interface ClassifierRun {
  data: SplitData;
  model: Classifier;
  scores: ClassifierScores;
}

/** Fits a classifier and scores it. */
export function runClassifier(model: Classifier, data: SplitData): ClassifierRun {
  const { train, test } = data;
  model.fit(train.x, train.y);
  const trainPred = model.predict(train.x);
  const hasTest = test.x.rows > 0;
  const testPred = hasTest ? model.predict(test.x) : null;
  const probs = (m: Table) => (model.predictProba ? logLoss(model.predictProba(m.x), m.y) : null);
  return {
    data,
    model,
    scores: {
      trainAccuracy: accuracy(train.y, trainPred),
      testAccuracy: testPred ? accuracy(test.y, testPred) : null,
      trainLoss: probs(train),
      testLoss: hasTest ? probs(test) : null,
      confusion: confusionMatrix(hasTest ? test.y : train.y, testPred ?? trainPred, model.classes),
    },
  };
}

export interface RegressionScores {
  trainRmse: number;
  testRmse: number | null;
  trainR2: number;
  testR2: number | null;
  trainMae: number;
  testMae: number | null;
  trainMse: number;
  testMse: number | null;
}

export interface RegressorRun {
  data: SplitData;
  model: Regressor;
  scores: RegressionScores;
}

export function runRegressor(model: Regressor, data: SplitData): RegressorRun {
  const { train, test } = data;
  model.fit(train.x, train.y);
  const trainPred = model.predict(train.x);
  const hasTest = test.x.rows > 0;
  const testPred = hasTest ? model.predict(test.x) : null;
  return {
    data,
    model,
    scores: {
      trainRmse: rmse(train.y, trainPred),
      testRmse: testPred ? rmse(test.y, testPred) : null,
      trainR2: r2(train.y, trainPred),
      testR2: testPred ? r2(test.y, testPred) : null,
      trainMae: mae(train.y, trainPred),
      testMae: testPred ? mae(test.y, testPred) : null,
      trainMse: mse(train.y, trainPred),
      testMse: testPred ? mse(test.y, testPred) : null,
    },
  };
}

export interface Regions {
  /** Cells per side. */
  n: number;
  /** Predicted class of each cell, row by row from the top. */
  labels: Int32Array;
  /** Confidence of that prediction (0..1) where the model has probabilities, else 1. */
  confidence: Float64Array;
  bounds: { x0: number; x1: number; y0: number; y1: number };
}

/** Evaluates a fitted classifier over a regular grid covering the data. */
export function predictRegions(model: Classifier, data: SplitData, n = 70): Regions {
  const bounds = plotBounds(data.all.x, 0.2);
  const grid = gridPoints(bounds.x0, bounds.x1, bounds.y0, bounds.y1, n);
  const labels = model.predict(grid);
  const confidence = new Float64Array(n * n).fill(1);
  if (model.predictProba) {
    const p = model.predictProba(grid);
    for (let i = 0; i < n * n; i++) {
      let best = 0;
      for (let c = 0; c < model.classes; c++) best = Math.max(best, p.get(i, c));
      confidence[i] = best;
    }
  }
  return { n, labels, confidence, bounds };
}

/** Samples a fitted regressor along the x axis of a one-feature table. */
export function predictCurve(
  model: Regressor,
  data: SplitData,
  n = 200,
): { xs: Float64Array; ys: Float64Array; x0: number; x1: number } {
  const b = plotBounds(data.all.x, 0.05);
  const xs = Float64Array.from({ length: n }, (_, c) => b.x0 + ((c + 0.5) / n) * (b.x1 - b.x0));
  return { xs, ys: model.predict(new Matrix(n, 1, Float64Array.from(xs))), x0: b.x0, x1: b.x1 };
}
