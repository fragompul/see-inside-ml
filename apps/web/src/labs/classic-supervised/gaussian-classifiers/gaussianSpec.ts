import { GaussianNB, LDA, QDA } from '@sim/classic';
import type { Classifier } from '@sim/core';
import { num, str, type ClassifierLabSpec, type ParamSpec } from '@/shared/lab-kit';

const params: ParamSpec[] = [
  {
    key: 'model',
    label: 'Model',
    kind: 'segmented',
    options: [
      { value: 'nb', label: 'Naive Bayes' },
      { value: 'lda', label: 'LDA' },
      { value: 'qda', label: 'QDA' },
    ],
    hint: 'Naive Bayes: independent features. LDA: one shared covariance. QDA: one covariance per class.',
  },
  {
    key: 'reg',
    label: 'Covariance shrinkage',
    kind: 'slider',
    min: 0,
    max: 0.9,
    step: 0.05,
    format: (v) => v.toFixed(2),
    hint: 'Pulls the covariance towards a round one: steadier with few points.',
    showIf: (v) => v.model !== 'nb',
  },
];

export type GaussianModel = GaussianNB | LDA | QDA;

export const gaussianOf = (model: Classifier): GaussianModel | null =>
  model instanceof GaussianNB || model instanceof LDA || model instanceof QDA ? model : null;

export const gaussianSpec: ClassifierLabSpec = {
  id: 'gaussian-classifiers',
  create: (p) => {
    const reg = num(p, 'reg');
    switch (str(p, 'model')) {
      case 'lda':
        return new LDA({ shrinkage: reg });
      case 'qda':
        return new QDA({ regParam: reg });
      default:
        return new GaussianNB();
    }
  },
  params,
  defaults: { model: 'lda', reg: 0 },
  dataset: 'overlap',
  data: { n: 240, noise: 0.15 },
  experiments: [
    {
      id: 'lda-blobs',
      title: 'LDA on round clusters',
      text: 'When every class is a round blob of the same size, a shared covariance is exactly right and the boundaries are straight lines between the centres.',
      setup: { dataset: 'blobs', noise: 0.1 },
      params: { model: 'lda' },
      claim: (run) => run.scores.trainAccuracy > 0.9,
    },
    {
      id: 'nb-correlated',
      title: 'Naive Bayes ignores correlation',
      text: 'The classes are shifted along x₁ inside a tilted, correlated cloud. Naive Bayes treats the features as independent, so its boundary ignores the tilt and misclassifies many points.',
      setup: { dataset: 'tilted', noise: 0.1 },
      params: { model: 'nb' },
      claim: (run) => run.scores.trainAccuracy < 0.85,
    },
    {
      id: 'lda-correlated',
      title: 'LDA uses the correlation',
      text: 'Same data, but a full covariance lets LDA lean its line against the cloud and separate the classes.',
      setup: { dataset: 'tilted', noise: 0.1 },
      params: { model: 'lda' },
      claim: (run) => run.scores.trainAccuracy > 0.9,
    },
    {
      id: 'lda-cigars',
      title: 'LDA on cigars: lost',
      text: 'Both classes share a centre and differ only in orientation. One shared covariance cannot express that: LDA is no better than chance.',
      setup: { dataset: 'cigars', noise: 0.1 },
      params: { model: 'lda' },
      claim: (run) => run.scores.trainAccuracy < 0.65,
    },
    {
      id: 'qda-cigars',
      title: 'QDA on cigars: solved',
      text: 'A covariance per class gives each its own ellipse and a curved boundary between them.',
      setup: { dataset: 'cigars', noise: 0.1 },
      params: { model: 'qda' },
      claim: (run) => run.scores.trainAccuracy > 0.8,
    },
    {
      id: 'qda-xor',
      title: 'QDA even solves XOR',
      text: 'The two XOR classes sit on opposite diagonals around the same centre, which is the cigars pattern in disguise.',
      setup: { dataset: 'xor' },
      params: { model: 'qda' },
      claim: (run) => run.scores.trainAccuracy > 0.8,
    },
  ],
};
