import { AdaBoost } from '@sim/classic';
import type { Classifier } from '@sim/core';
import { num, type ClassifierLabSpec, type ParamSpec } from '@/shared/lab-kit';

const params: ParamSpec[] = [
  { key: 'rounds', label: 'Boosting rounds', kind: 'slider', min: 1, max: 100, step: 1 },
  {
    key: 'depth',
    label: 'Weak learner depth',
    kind: 'slider',
    min: 1,
    max: 4,
    step: 1,
    hint: '1 = decision stumps.',
  },
  {
    key: 'lr',
    label: 'Learning rate',
    kind: 'ladder',
    values: [0.05, 0.1, 0.25, 0.5, 1],
    format: (v) => String(v),
  },
];

export const adaOf = (m: Classifier): AdaBoost | null => (m instanceof AdaBoost ? m : null);

export const adaboostSpec: ClassifierLabSpec = {
  id: 'adaboost',
  create: (p) =>
    new AdaBoost({ rounds: num(p, 'rounds'), depth: num(p, 'depth'), learningRate: num(p, 'lr') }),
  params,
  defaults: { rounds: 30, depth: 1, lr: 1 },
  dataset: 'moons',
  data: { n: 200, noise: 0.3 },
  experiments: [
    {
      id: 'one-stump',
      title: 'One stump',
      text: 'A single split of one feature: a weak learner, barely better than chance on crescents.',
      params: { rounds: 1 },
      claim: (run) => run.scores.trainAccuracy < 0.85,
    },
    {
      id: 'many-stumps',
      title: '50 stumps',
      text: 'Each round reweights the points the ensemble gets wrong and adds a stump for them: weak learners add up to a strong one.',
      params: { rounds: 50 },
      claim: (run) => run.scores.trainAccuracy > 0.9,
    },
    {
      id: 'xor-stumps',
      title: 'Stumps cannot do XOR',
      text: 'A sum of axis-aligned stumps is additive in the features, and XOR is not: no number of rounds fixes it.',
      setup: { dataset: 'xor' },
      params: { rounds: 100 },
      claim: (run) => run.scores.trainAccuracy < 0.75,
    },
    {
      id: 'xor-depth2',
      title: 'Depth-2 learners solve XOR',
      text: 'Allow each weak learner two questions in a row and the interaction between the features is captured.',
      setup: { dataset: 'xor' },
      params: { rounds: 30, depth: 2 },
      claim: (run) => run.scores.trainAccuracy > 0.9,
    },
  ],
};
