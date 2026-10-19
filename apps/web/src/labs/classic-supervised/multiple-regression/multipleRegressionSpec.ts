import { LinearRegression } from '@sim/classic';
import { flag, num, type ParamSpec, type TabularLabSpec } from '@/shared/lab-kit';

export const asLinear = (model: unknown): LinearRegression | null =>
  model instanceof LinearRegression ? model : null;

const params: ParamSpec[] = [
  { key: 'intercept', label: 'Fit an intercept', kind: 'switch' },
  { key: 'ridge', label: 'Shrink the coefficients (ridge)', kind: 'switch' },
  {
    key: 'l2',
    label: 'Ridge penalty λ',
    kind: 'ladder',
    values: [0.01, 0.1, 1, 10, 100],
    hint: 'Larger λ shrinks the coefficients harder.',
    showIf: (v) => flag(v, 'ridge'),
    format: (v) => String(v),
  },
];

export const multipleRegressionSpec: TabularLabSpec = {
  id: 'multiple-regression',
  create: (p) =>
    new LinearRegression({
      intercept: flag(p, 'intercept'),
      l2: flag(p, 'ridge') ? num(p, 'l2') : 0,
    }),
  params,
  defaults: { intercept: true, ridge: false, l2: 1 },
  experiments: [
    {
      id: 'independent',
      title: 'Independent features',
      text: 'With uncorrelated features every coefficient lands close to the true value (2, −1, 0.4) and the variance inflation factors are all about 1.',
      setup: { source: 'synthetic', correlation: 0, noise: 0.5 },
      claim: (run) => {
        const m = asLinear(run.model);
        if (!m) return false;
        const [a, b, c] = m.coef;
        return (
          Math.abs((a ?? 0) - 2) < 0.2 &&
          Math.abs((b ?? 0) + 1) < 0.2 &&
          Math.abs((c ?? 0) - 0.4) < 0.2 &&
          Math.max(...m.vif()) < 1.5
        );
      },
    },
    {
      id: 'collinear',
      title: 'Nearly the same feature twice',
      text: 'With x₂ almost a copy of x₁ the variance inflation factor shoots up and the coefficients of the pair become uncertain, even though predictions stay good.',
      setup: { source: 'synthetic', correlation: 0.98, noise: 0.5 },
      claim: (run) => {
        const m = asLinear(run.model);
        if (!m) return false;
        const se = m.summary().coefficients[2]?.stdError ?? 0;
        return Math.max(...m.vif()) > 10 && se > 0.15 && (run.scores.testR2 ?? 0) > 0.5;
      },
    },
    {
      id: 'ridge',
      title: 'Ridge on collinear features',
      text: 'The penalty shrinks the unstable pair towards zero, with the same predictive accuracy: it trades bias for stability.',
      setup: { source: 'synthetic', correlation: 0.98, noise: 0.5 },
      params: { ridge: true, l2: 1 },
      claim: (run) => {
        const m = asLinear(run.model);
        if (!m) return false;
        return Math.abs(m.coef[0] ?? 0) < 1.8 && (run.scores.testRmse ?? 9) < 0.6;
      },
    },
    {
      id: 'noisy',
      title: 'Plenty of noise',
      text: 'The more noise, the bigger the gap between the fit on the training rows and on held-out rows.',
      setup: { source: 'synthetic', correlation: 0, noise: 2.5 },
      claim: (run) => run.scores.trainR2 - (run.scores.testR2 ?? 0) > 0.3,
    },
    {
      id: 'diabetes',
      title: 'Real data: Diabetes',
      text: 'On the Diabetes data two blood-serum measurements (s1, s2) are strongly collinear, so their coefficients are unreliable; overall about half the variance is explained.',
      setup: { source: 'diabetes' },
      claim: (run) => {
        const m = asLinear(run.model);
        if (!m) return false;
        return Math.max(...m.vif()) > 30 && (run.scores.testR2 ?? 0) > 0.4;
      },
    },
  ],
};
