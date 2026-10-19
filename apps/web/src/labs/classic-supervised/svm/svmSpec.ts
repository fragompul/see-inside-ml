import { SVC, type KernelName } from '@sim/classic';
import type { Classifier } from '@sim/core';
import { num, str, type ClassifierLabSpec, type ParamSpec } from '@/shared/lab-kit';

export const C_VALUES = [0.01, 0.1, 1, 10, 100, 1000] as const;
export const GAMMA_VALUES = [0.01, 0.05, 0.2, 1, 5, 20, 50] as const;

const params: ParamSpec[] = [
  {
    key: 'kernel',
    label: 'Kernel',
    kind: 'segmented',
    options: [
      { value: 'linear', label: 'Linear' },
      { value: 'rbf', label: 'RBF' },
      { value: 'poly', label: 'Polynomial' },
      { value: 'sigmoid', label: 'Sigmoid' },
    ],
  },
  {
    key: 'C',
    label: 'Penalty C',
    kind: 'ladder',
    values: C_VALUES,
    format: (v) => String(v),
    hint: 'Small C = a wide, tolerant margin; large C = few violations allowed.',
  },
  {
    key: 'gamma',
    label: 'Kernel width γ',
    kind: 'ladder',
    values: GAMMA_VALUES,
    format: (v) => String(v),
    hint: 'Large γ = each point only influences its close neighbours: a bumpy boundary.',
    showIf: (v) => v.kernel !== 'linear',
  },
  {
    key: 'degree',
    label: 'Polynomial degree',
    kind: 'slider',
    min: 2,
    max: 6,
    step: 1,
    showIf: (v) => v.kernel === 'poly',
  },
];

export const svcOf = (m: Classifier): SVC | null => (m instanceof SVC ? m : null);

export const svmSpec: ClassifierLabSpec = {
  id: 'svm',
  create: (p) =>
    new SVC({
      kernel: str(p, 'kernel') as KernelName,
      C: num(p, 'C'),
      gamma: num(p, 'gamma'),
      degree: num(p, 'degree'),
      coef0: 1,
      tol: 1e-4,
    }),
  params,
  defaults: { kernel: 'rbf', C: 1, gamma: 1, degree: 3 },
  dataset: 'moons',
  data: { n: 200, noise: 0.3 },
  experiments: [
    {
      id: 'linear-fails',
      title: 'A linear SVM on crescents',
      text: 'The best straight maximum-margin line still cuts through both crescents.',
      params: { kernel: 'linear' },
      claim: (run) => run.scores.trainAccuracy < 0.9,
    },
    {
      id: 'rbf',
      title: 'The RBF kernel bends it',
      text: 'The kernel trick lets the same margin idea work in an implicit high-dimensional space: a curved boundary, few support vectors.',
      params: { kernel: 'rbf', gamma: 1, C: 1 },
      claim: (run) => {
        const m = run.model instanceof SVC ? run.model : null;
        return (
          run.scores.trainAccuracy > 0.93 &&
          !!m &&
          m.supportIndices.length < 0.5 * run.data.train.x.rows
        );
      },
    },
    {
      id: 'wide-gamma',
      title: 'γ far too large: memorise',
      text: 'With tiny, spiky kernels every point gets its own island: perfect on training, poor on new points, and almost every point is a support vector.',
      params: { kernel: 'rbf', gamma: 50, C: 100 },
      claim: (run) => {
        const m = run.model instanceof SVC ? run.model : null;
        return (
          run.scores.trainAccuracy === 1 &&
          (run.scores.testAccuracy ?? 1) < 1 &&
          !!m &&
          m.supportIndices.length > 0.5 * run.data.train.x.rows
        );
      },
    },
    {
      id: 'tiny-c',
      title: 'C very small: tolerate everything',
      text: 'When violations cost almost nothing the margin becomes huge, nearly every point is a support vector and the boundary flattens.',
      params: { kernel: 'rbf', gamma: 1, C: 0.01 },
      claim: (run) => {
        const m = run.model instanceof SVC ? run.model : null;
        return !!m && m.supportIndices.length > 0.8 * run.data.train.x.rows;
      },
    },
    {
      id: 'poly-circles',
      title: 'Polynomial kernel on circles',
      text: 'A degree-2 polynomial kernel measures squared distances from the centre, which is exactly what a ring needs.',
      setup: { dataset: 'circles' },
      params: { kernel: 'poly', degree: 2, C: 10, gamma: 1 },
      claim: (run) => run.scores.trainAccuracy > 0.9,
    },
  ],
};
