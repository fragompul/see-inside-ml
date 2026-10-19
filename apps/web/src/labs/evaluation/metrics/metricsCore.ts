import {
  DecisionTree,
  KnnClassifier,
  LogisticRegression,
  bestThreshold,
  brierScore,
  calibrationBins,
  costAt,
  expectedCalibrationError,
  metricsAt,
  oversample,
  plattScaler,
  prCurve,
  smote,
  type CalibrationBin,
  type ThresholdMetrics,
} from '@sim/classic';
import { Matrix, auc, createRng, rocCurve, type Classifier } from '@sim/core';

/**
 * A rare-event detector, scored every way there is. Two Gaussian clouds (negatives and rarer
 * positives) are split into training, calibration and test parts; one model is trained and then
 * judged by accuracy, precision, recall, ROC and precision-recall curves, calibration and cost.
 */
export type ModelId = 'logistic' | 'tree' | 'knn';
export type Rebalance = 'none' | 'oversample' | 'smote';

export interface MetricsScenario {
  /** Share of positives in the data. */
  positiveShare: number;
  /** Distance between the two class means, in standard deviations. */
  separation: number;
  model: ModelId;
  rebalance: Rebalance;
  /** Recalibrate the probabilities with Platt scaling on a separate part of the data. */
  calibrate: boolean;
  threshold: number;
  /** How many times worse a missed positive is than a false alarm. */
  costRatio: number;
  seed: number;
}

export const DEFAULT_METRICS: MetricsScenario = {
  positiveShare: 0.1,
  separation: 2,
  model: 'logistic',
  rebalance: 'none',
  calibrate: false,
  threshold: 0.5,
  costRatio: 1,
  seed: 3,
};

export const MODEL_LABELS: Record<ModelId, string> = {
  logistic: 'Logistic regression',
  tree: 'Decision tree (depth 8)',
  knn: '15 nearest neighbours',
};

export interface MetricsRun {
  scenario: MetricsScenario;
  scores: number[];
  labels: number[];
  at: ThresholdMetrics;
  /** Accuracy of always predicting "negative". */
  baselineAccuracy: number;
  roc: { fpr: number[]; tpr: number[] };
  rocAuc: number;
  pr: { precision: number[]; recall: number[] };
  averagePrecision: number;
  calibration: CalibrationBin[];
  brier: number;
  ece: number;
  cost: number;
  best: { threshold: number; cost: number };
  /** Positives in the test set. */
  positives: number;
}

function sample(n: number, share: number, separation: number, seed: number) {
  const rng = createRng(seed);
  const x = new Matrix(n, 2);
  const y: number[] = [];
  const shift = separation / Math.SQRT2;
  for (let i = 0; i < n; i++) {
    const pos = rng.next() < share;
    y.push(pos ? 1 : 0);
    x.set(i, 0, rng.normal() + (pos ? shift : 0));
    x.set(i, 1, rng.normal() + (pos ? shift : 0));
  }
  return { x, y };
}

const pick = (x: Matrix, idx: readonly number[]): Matrix => {
  const out = new Matrix(idx.length, x.cols);
  idx.forEach((r, i) => out.data.set(x.data.subarray(r * x.cols, (r + 1) * x.cols), i * x.cols));
  return out;
};

function make(model: ModelId): Classifier {
  if (model === 'logistic') return new LogisticRegression({ C: 1 });
  if (model === 'tree') return new DecisionTree({ maxDepth: 8, minSamplesLeaf: 1 });
  return new KnnClassifier({ k: 15 });
}

function positiveProbability(m: Classifier, x: Matrix): number[] {
  const proba = m.predictProba?.(x);
  if (!proba) throw new Error('This model has no probabilities');
  return Array.from({ length: x.rows }, (_, i) => proba.get(i, 1));
}

export function runMetrics(s: MetricsScenario): MetricsRun {
  const total = sample(4000, s.positiveShare, s.separation, s.seed);
  const idx = Array.from({ length: 4000 }, (_, i) => i);
  const train = idx.slice(0, 2000);
  const cal = idx.slice(2000, 2800);
  const test = idx.slice(2800);
  let xTrain = pick(total.x, train);
  let yTrain = train.map((i) => total.y[i] as number);
  if (s.rebalance === 'oversample') ({ x: xTrain, y: yTrain } = oversample(xTrain, yTrain, s.seed));
  if (s.rebalance === 'smote') ({ x: xTrain, y: yTrain } = smote(xTrain, yTrain, 5, s.seed));
  const model = make(s.model).fit(xTrain, yTrain);
  let scores = positiveProbability(model, pick(total.x, test));
  if (s.calibrate) {
    const f = plattScaler(
      positiveProbability(model, pick(total.x, cal)),
      cal.map((i) => total.y[i] as number),
    );
    scores = scores.map(f);
  }
  const labels = test.map((i) => total.y[i] as number);
  const roc = rocCurve(scores, labels);
  const pr = prCurve(scores, labels);
  const positives = labels.filter((v) => v === 1).length;
  return {
    scenario: s,
    scores,
    labels,
    at: metricsAt(scores, labels, s.threshold),
    baselineAccuracy: 1 - positives / labels.length,
    roc: { fpr: roc.fpr, tpr: roc.tpr },
    rocAuc: auc(roc.fpr, roc.tpr),
    pr: { precision: pr.precision, recall: pr.recall },
    averagePrecision: pr.averagePrecision,
    calibration: calibrationBins(scores, labels, 10),
    brier: brierScore(scores, labels),
    ece: expectedCalibrationError(scores, labels, 10),
    cost: costAt(scores, labels, s.threshold, s.costRatio, 1),
    best: bestThreshold(scores, labels, s.costRatio, 1),
    positives,
  };
}
