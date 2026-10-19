import { LogisticRegression, PolynomialFeatureClassifier, polynomialSize } from '@sim/classic';
import type { Classifier } from '@sim/core';
import { num, type ClassifierLabSpec, type ParamSpec } from '@/shared/lab-kit';

export const C_VALUES = [0.001, 0.01, 0.1, 1, 10, 100, 1000] as const;

const params: ParamSpec[] = [
  {
    key: 'C',
    label: 'Inverse regularisation C',
    kind: 'ladder',
    values: C_VALUES,
    format: (v) => String(v),
    hint: 'Small C = strong penalty on the weights; large C = almost none.',
  },
  {
    key: 'degree',
    label: 'Feature degree',
    kind: 'slider',
    min: 1,
    max: 5,
    step: 1,
    hint: '1 = the two inputs; higher adds their powers and products, so the boundary can bend.',
  },
];

/** The raw logistic model inside a (possibly polynomial) pipeline. */
export const logisticOf = (model: Classifier): LogisticRegression | null =>
  model instanceof LogisticRegression
    ? model
    : model instanceof PolynomialFeatureClassifier && model.inner instanceof LogisticRegression
      ? model.inner
      : null;

export const featureCount = (degree: number) => polynomialSize(2, degree);

export const logisticSpec: ClassifierLabSpec = {
  id: 'logistic-regression',
  create: (p) => {
    const lr = new LogisticRegression({ C: num(p, 'C') });
    const degree = num(p, 'degree');
    return degree > 1 ? new PolynomialFeatureClassifier(lr, degree) : lr;
  },
  params,
  defaults: { C: 1, degree: 1 },
  dataset: 'overlap',
  data: { n: 200, noise: 0.25 },
  experiments: [
    {
      id: 'linear-fits',
      title: 'A line is enough',
      text: 'Two overlapping clusters: the boundary is a straight line and the probabilities fade smoothly across it.',
      setup: { dataset: 'overlap' },
      claim: (run) => run.scores.trainAccuracy > 0.75,
    },
    {
      id: 'xor-fails',
      title: 'XOR defeats a line',
      text: 'Opposite corners share a class, so no straight boundary works: accuracy stays near a coin flip.',
      setup: { dataset: 'xor' },
      claim: (run) => run.scores.trainAccuracy < 0.7,
    },
    {
      id: 'xor-features',
      title: 'Add squared and product features',
      text: 'With the product x₁·x₂ among the features the same linear model separates XOR: logistic regression is linear in its features, not in the raw inputs.',
      setup: { dataset: 'xor' },
      params: { degree: 2, C: 100 },
      claim: (run) => run.scores.trainAccuracy > 0.9,
    },
    {
      id: 'circles-features',
      title: 'Circles with degree 2',
      text: 'A ring needs a closed boundary; squared features supply it.',
      setup: { dataset: 'circles' },
      params: { degree: 2, C: 100 },
      claim: (run) => run.scores.trainAccuracy > 0.9,
    },
    {
      id: 'strong-penalty',
      title: 'Crush the weights',
      text: 'A tiny C forces the weights to almost zero: every probability drifts towards the base rate and the log-loss rises.',
      setup: { dataset: 'overlap' },
      params: { C: 0.001 },
      claim: (run) => run.scores.trainLoss !== null && run.scores.trainLoss > 0.55,
    },
    {
      id: 'three-classes',
      title: 'Three classes: softmax',
      text: 'With more than two classes the model learns one score per class and the softmax turns them into probabilities.',
      setup: { dataset: 'blobs' },
      claim: (run) => run.model.classes === 3 && run.scores.trainAccuracy > 0.9,
    },
  ],
};
