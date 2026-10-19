import { GradientBoostingRegressor, XgbRegressor } from '@sim/classic';
import type { Matrix, Regressor } from '@sim/core';
import { num, str, type ParamSpec, type RegressorLabSpec } from '@/shared/lab-kit';

const params: ParamSpec[] = [
  {
    key: 'engine',
    label: 'Flavour',
    kind: 'segmented',
    options: [
      { value: 'gbm', label: 'Gradient boosting' },
      { value: 'xgb', label: 'XGBoost-style' },
    ],
    hint: 'Classic gradient boosting fits residuals; the XGBoost style also uses curvature and penalties.',
  },
  { key: 'rounds', label: 'Rounds (trees)', kind: 'slider', min: 1, max: 300, step: 1 },
  {
    key: 'lr',
    label: 'Learning rate',
    kind: 'ladder',
    values: [0.01, 0.03, 0.1, 0.3, 1],
    format: (v) => String(v),
  },
  { key: 'depth', label: 'Tree depth', kind: 'slider', min: 1, max: 8, step: 1 },
  {
    key: 'subsample',
    label: 'Row subsample',
    kind: 'slider',
    min: 0.3,
    max: 1,
    step: 0.1,
    format: (v) => v.toFixed(1),
    showIf: (v) => v.engine === 'gbm',
    hint: 'Below 1 = stochastic gradient boosting.',
  },
  {
    key: 'lambda',
    label: 'Leaf penalty λ',
    kind: 'ladder',
    values: [0, 1, 5, 20, 100],
    format: (v) => String(v),
    showIf: (v) => v.engine === 'xgb',
  },
  {
    key: 'gamma',
    label: 'Split price γ',
    kind: 'ladder',
    values: [0, 0.1, 0.5, 2, 10],
    format: (v) => String(v),
    showIf: (v) => v.engine === 'xgb',
  },
];

export type StagedRegressor = Regressor & {
  predict(x: Matrix, count?: number): Float64Array;
};

export const makeBooster = (p: Record<string, number | string | boolean>): StagedRegressor =>
  str(p, 'engine') === 'xgb'
    ? new XgbRegressor({
        rounds: num(p, 'rounds'),
        eta: num(p, 'lr'),
        maxDepth: num(p, 'depth'),
        lambda: num(p, 'lambda'),
        gamma: num(p, 'gamma'),
        baseScore: 0,
      })
    : new GradientBoostingRegressor({
        rounds: num(p, 'rounds'),
        learningRate: num(p, 'lr'),
        depth: num(p, 'depth'),
        subsample: num(p, 'subsample'),
        seed: 5,
      });

export const boostingSpec: RegressorLabSpec = {
  id: 'gradient-boosting',
  create: makeBooster,
  params,
  defaults: { engine: 'gbm', rounds: 100, lr: 0.1, depth: 3, subsample: 1, lambda: 1, gamma: 0 },
  dataset: 'sine',
  data: { n: 150, noise: 0.35 },
  experiments: [
    {
      id: 'one-tree',
      title: 'One stump',
      text: 'After a single one-split tree the model is a two-step staircase: boosting starts from a very weak fit.',
      params: { rounds: 1, lr: 1, depth: 1 },
      claim: (run) => run.scores.trainR2 < 0.6,
    },
    {
      id: 'shrunk',
      title: '100 small steps',
      text: 'A hundred trees, each allowed to move the prediction only 10% of the way, trace the wave smoothly.',
      claim: (run) => run.scores.trainR2 > 0.85 && (run.scores.testR2 ?? 0) > 0.75,
    },
    {
      id: 'too-slow',
      title: 'Learning rate too small',
      text: 'With a tiny step size 20 trees have barely begun: the model is still underfit.',
      params: { rounds: 20, lr: 0.01 },
      claim: (run) => run.scores.trainR2 < 0.4,
    },
    {
      id: 'overfit',
      title: 'Deep trees, big steps',
      text: 'Powerful trees with no shrinkage chase every noisy point: training error collapses, test error does not.',
      params: { rounds: 150, lr: 1, depth: 6 },
      claim: (run) =>
        run.scores.trainRmse < 0.2 && (run.scores.testRmse ?? 0) > 1.5 * run.scores.trainRmse,
    },
    {
      id: 'xgb-penalty',
      title: 'XGBoost: a heavy λ',
      text: 'The leaf penalty λ shrinks every leaf towards zero, so the same trees move the prediction far less.',
      params: { engine: 'xgb', lambda: 100, rounds: 30 },
      claim: (run) => run.scores.trainR2 < 0.7,
    },
    {
      id: 'xgb-balanced',
      title: 'XGBoost with modest penalties',
      text: 'With λ = 1 and a small γ the second-order booster fits the wave as well as the classic one.',
      params: { engine: 'xgb', lambda: 1, gamma: 0.5 },
      claim: (run) => (run.scores.testR2 ?? 0) > 0.7,
    },
  ],
};
