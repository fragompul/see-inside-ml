import { PolynomialRegression, type PolynomialPenalty } from '@sim/classic';
import type { Regressor } from '@sim/core';
import { num, str, type ParamSpec, type RegressorLabSpec } from '@/shared/lab-kit';

export const ALPHAS = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100] as const;

const params: ParamSpec[] = [
  { key: 'degree', label: 'Polynomial degree', kind: 'slider', min: 1, max: 15, step: 1 },
  {
    key: 'penalty',
    label: 'Penalty',
    kind: 'segmented',
    options: [
      { value: 'none', label: 'None' },
      { value: 'ridge', label: 'Ridge (L2)' },
      { value: 'lasso', label: 'Lasso (L1)' },
    ],
  },
  {
    key: 'alpha',
    label: 'Penalty strength α',
    kind: 'ladder',
    values: ALPHAS,
    format: (v) => String(v),
    showIf: (v) => v.penalty !== 'none',
  },
];

export const polynomialOf = (m: Regressor): PolynomialRegression | null =>
  m instanceof PolynomialRegression ? m : null;

export const polynomialSpec: RegressorLabSpec = {
  id: 'polynomial-regularization',
  create: (p) =>
    new PolynomialRegression({
      degree: num(p, 'degree'),
      penalty: str(p, 'penalty') as PolynomialPenalty,
      alpha: num(p, 'alpha'),
    }),
  params,
  defaults: { degree: 3, penalty: 'none', alpha: 0.01 },
  dataset: 'sine',
  data: { n: 50, noise: 0.3 },
  experiments: [
    {
      id: 'underfit',
      title: 'Degree 1: underfit',
      text: 'A straight line cannot follow a wave: error is high on both training and test points.',
      params: { degree: 1 },
      claim: (run) => run.scores.trainR2 < 0.3,
    },
    {
      id: 'right',
      title: 'Degree 7: about right',
      text: 'Enough flexibility to follow the wave, not enough to chase the noise.',
      params: { degree: 7 },
      claim: (run) => run.scores.trainR2 > 0.85 && (run.scores.testR2 ?? 0) > 0.7,
    },
    {
      id: 'overfit',
      title: 'Degree 15: overfit',
      text: 'The curve wiggles through the noise: training error shrinks while the error on held-out points grows.',
      params: { degree: 15 },
      claim: (run) => (run.scores.testRmse ?? 0) > 1.5 * run.scores.trainRmse,
    },
    {
      id: 'ridge',
      title: 'Ridge tames degree 15',
      text: 'The L2 penalty shrinks the huge coefficients and smooths the curve, so the test error falls back.',
      params: { degree: 15, penalty: 'ridge', alpha: 0.001 },
      claim: (run) => (run.scores.testR2 ?? 0) > 0.75,
    },
    {
      id: 'lasso',
      title: 'Lasso switches terms off',
      text: 'The L1 penalty sets many coefficients to exactly zero: the model keeps only the powers it needs.',
      params: { degree: 15, penalty: 'lasso', alpha: 0.001 },
      claim: (run) => {
        const m = run.model instanceof PolynomialRegression ? run.model : null;
        return !!m && m.coef.some((c) => c === 0) && (run.scores.testR2 ?? 0) > 0.65;
      },
    },
  ],
};
