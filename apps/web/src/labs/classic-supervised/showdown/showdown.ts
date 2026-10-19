import {
  AdaBoost,
  DecisionTree,
  GaussianNB,
  GradientBoostingClassifier,
  KnnClassifier,
  LDA,
  LogisticRegression,
  QDA,
  RandomForestClassifier,
  SVC,
} from '@sim/classic';
import type { Classifier, ClassificationSet } from '@sim/core';
import {
  DEFAULT_DATA,
  buildClassification,
  predictRegions,
  runClassifier,
  type ClassificationSetup,
  type Regions,
} from '@/shared/lab-kit';

export interface ShowdownModel {
  id: string;
  label: string;
  family: 'linear' | 'probabilistic' | 'instance' | 'tree' | 'ensemble' | 'kernel';
  make: () => Classifier;
}

/** Ten classifiers with sensible default settings (nothing tuned per dataset, on purpose). */
export const SHOWDOWN_MODELS: readonly ShowdownModel[] = [
  {
    id: 'logistic',
    label: 'Logistic regression',
    family: 'linear',
    make: () => new LogisticRegression({ C: 1 }),
  },
  { id: 'lda', label: 'LDA', family: 'probabilistic', make: () => new LDA() },
  { id: 'qda', label: 'QDA', family: 'probabilistic', make: () => new QDA({ regParam: 0.05 }) },
  { id: 'nb', label: 'Naive Bayes', family: 'probabilistic', make: () => new GaussianNB() },
  { id: 'knn', label: 'k-NN (k = 7)', family: 'instance', make: () => new KnnClassifier({ k: 7 }) },
  {
    id: 'tree',
    label: 'Decision tree (depth 6)',
    family: 'tree',
    make: () => new DecisionTree({ maxDepth: 6, minSamplesLeaf: 3 }),
  },
  {
    id: 'forest',
    label: 'Random forest',
    family: 'ensemble',
    make: () => new RandomForestClassifier({ trees: 60, maxDepth: 8, seed: 3 }),
  },
  {
    id: 'adaboost',
    label: 'AdaBoost',
    family: 'ensemble',
    make: () => new AdaBoost({ rounds: 40, depth: 2 }),
  },
  {
    id: 'gbm',
    label: 'Gradient boosting',
    family: 'ensemble',
    make: () =>
      new GradientBoostingClassifier({ rounds: 60, depth: 3, learningRate: 0.1, seed: 2 }),
  },
  {
    id: 'svm',
    label: 'SVM (RBF)',
    family: 'kernel',
    make: () => new SVC({ kernel: 'rbf', C: 1, gamma: 1 }),
  },
];

export interface ShowdownRow {
  model: ShowdownModel;
  trainAccuracy: number;
  testAccuracy: number | null;
  testLoss: number | null;
  fitMs: number;
  regions: Regions;
}

export interface ShowdownResult {
  rows: ShowdownRow[];
  /** Id of the model with the best test accuracy (ties: the earlier one). */
  winner: string;
}

export type ShowdownSetup = ClassificationSetup;

export const defaultShowdown = (dataset: ClassificationSet = 'moons'): ShowdownSetup => ({
  ...DEFAULT_DATA,
  dataset,
  n: 240,
  noise: 0.3,
});

export function runShowdown(
  setup: ShowdownSetup,
  models = SHOWDOWN_MODELS,
  gridSize = 48,
): ShowdownResult {
  const data = buildClassification(setup);
  const rows = models.map((model) => {
    const start = performance.now();
    const run = runClassifier(model.make(), data);
    const fitMs = performance.now() - start;
    return {
      model,
      trainAccuracy: run.scores.trainAccuracy,
      testAccuracy: run.scores.testAccuracy,
      testLoss: run.scores.testLoss,
      fitMs,
      regions: predictRegions(run.model, data, gridSize),
    };
  });
  const score = (r: ShowdownRow) => r.testAccuracy ?? r.trainAccuracy;
  const winner = rows.reduce((b, r) => (score(r) > score(b) ? r : b), rows[0] as ShowdownRow).model
    .id;
  return { rows, winner };
}

const acc = (r: ShowdownResult, id: string): number => {
  const row = r.rows.find((x) => x.model.id === id) as ShowdownRow;
  return row.testAccuracy ?? row.trainAccuracy;
};

export interface ShowdownExperiment {
  id: string;
  title: string;
  text: string;
  setup: Partial<ShowdownSetup>;
  claim: (r: ShowdownResult) => boolean;
}

export const showdownExperiments: readonly ShowdownExperiment[] = [
  {
    id: 'moons',
    title: 'Crescents: flexibility wins',
    text: 'Straight-boundary models (logistic, LDA) cannot follow the crescents; the flexible ones can.',
    setup: { dataset: 'moons', noise: 0.25 },
    claim: (r) =>
      Math.max(acc(r, 'knn'), acc(r, 'svm'), acc(r, 'forest')) > acc(r, 'logistic') + 0.04,
  },
  {
    id: 'xor',
    title: 'XOR: lines are lost',
    text: 'On XOR logistic regression and LDA are no better than a coin flip; trees, neighbours and kernels solve it.',
    setup: { dataset: 'xor', noise: 0.2 },
    claim: (r) => acc(r, 'logistic') < 0.65 && acc(r, 'lda') < 0.65 && acc(r, 'knn') > 0.85,
  },
  {
    id: 'cigars',
    title: 'Cigars: covariance is everything',
    text: 'Both classes share a centre and differ only in orientation, so only models with a covariance per class (QDA) or flexible ones see it.',
    setup: { dataset: 'cigars', noise: 0.15 },
    claim: (r) => acc(r, 'qda') > acc(r, 'lda') + 0.2,
  },
  {
    id: 'correlated',
    title: 'Correlated features: naive Bayes stumbles',
    text: 'Naive Bayes ignores correlations between features; LDA, which models them, does clearly better.',
    setup: { dataset: 'tilted', noise: 0.1 },
    claim: (r) => acc(r, 'lda') > acc(r, 'nb') + 0.05,
  },
  {
    id: 'overlap',
    title: 'Overlapping classes: a ceiling for everyone',
    text: 'When the classes genuinely overlap no model can beat the noise: the spread between the best and the simplest is small.',
    setup: { dataset: 'overlap', noise: 0.15, n: 400 },
    claim: (r) => Math.max(...r.rows.map((x) => x.testAccuracy ?? 0)) - acc(r, 'logistic') < 0.1,
  },
];

export const experimentSetup = (e: ShowdownExperiment): ShowdownSetup => ({
  ...defaultShowdown(),
  ...e.setup,
});
