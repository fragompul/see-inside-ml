import { LinearRegression } from '@sim/classic';
import { flag, type ParamSpec, type RegressorLabSpec } from '@/shared/lab-kit';

/** The fitted line behind a run, when the model is the least-squares one. */
export const linearModel = (model: unknown): LinearRegression | null =>
  model instanceof LinearRegression ? model : null;

const params: ParamSpec[] = [
  {
    key: 'intercept',
    label: 'Fit an intercept',
    kind: 'switch',
    hint: 'Off forces the line through the origin.',
  },
];

export const linearRegressionSpec: RegressorLabSpec = {
  id: 'linear-regression',
  create: (p) => new LinearRegression({ intercept: flag(p, 'intercept') }),
  params,
  defaults: { intercept: true },
  dataset: 'line',
  data: { n: 80, noise: 0.3 },
  experiments: [
    {
      id: 'clean-line',
      title: 'Almost no noise',
      text: 'With barely any scatter the line recovers the trend almost exactly: R² is close to 1.',
      setup: { dataset: 'line', noise: 0.05 },
      claim: (run) => run.scores.trainR2 > 0.99,
    },
    {
      id: 'noisy-line',
      title: 'Lots of noise',
      text: 'The same trend buried in noise: the line is still the best one, but it explains far less of the variation.',
      setup: { dataset: 'line', noise: 1.5 },
      claim: (run) => run.scores.trainR2 > 0.3 && run.scores.trainR2 < 0.75,
    },
    {
      id: 'parabola',
      title: 'A line on a parabola',
      text: 'The data bend, the model cannot: it underfits, and no amount of data would fix that.',
      setup: { dataset: 'curve', noise: 0.25 },
      claim: (run) => run.scores.trainR2 < 0.5,
    },
    {
      id: 'sine',
      title: 'A line on a sine wave',
      text: 'A wave has no overall slope, so the best line is nearly flat and explains almost nothing.',
      setup: { dataset: 'sine', noise: 0.25 },
      claim: (run) => run.scores.trainR2 < 0.2,
    },
    {
      id: 'outliers',
      title: 'Outliers tug the line',
      text: 'A few wild points drag the fit off the true intercept (0.5) and wreck R²; see their Cook’s distance.',
      setup: { dataset: 'outliers', n: 80, noise: 0.3 },
      claim: (run) => {
        const m = linearModel(run.model);
        return !!m && run.scores.trainR2 < 0.65 && Math.abs(m.intercept - 0.5) > 0.25;
      },
    },
  ],
};
