import { Glm, HuberRegression, LinearRegression, QuantileRegression, TheilSen } from '@sim/classic';
import { num, str, type ParamSpec, type RegressorLabSpec } from '@/shared/lab-kit';

export type RobustModelId = 'ols' | 'huber' | 'median' | 'theilsen' | 'quantile' | 'poisson';

export const MODELS: { value: RobustModelId; label: string }[] = [
  { value: 'ols', label: 'Least squares' },
  { value: 'huber', label: 'Huber' },
  { value: 'median', label: 'Median (L1)' },
  { value: 'theilsen', label: 'Theil–Sen' },
  { value: 'quantile', label: 'Quantile' },
  { value: 'poisson', label: 'Poisson GLM' },
];

const params: ParamSpec[] = [
  { key: 'model', label: 'Model', kind: 'select', options: MODELS },
  {
    key: 'tau',
    label: 'Quantile τ',
    kind: 'slider',
    min: 0.05,
    max: 0.95,
    step: 0.05,
    format: (v) => v.toFixed(2),
    showIf: (v) => v.model === 'quantile',
    hint: 'The line below which this share of the points fall.',
  },
  {
    key: 'huberT',
    label: 'Huber threshold',
    kind: 'slider',
    min: 0.5,
    max: 4,
    step: 0.1,
    format: (v) => `${v.toFixed(1)} σ`,
    showIf: (v) => v.model === 'huber',
    hint: 'Residuals beyond this many robust σ are down-weighted.',
  },
];

export function makeModel(id: RobustModelId, tau = 0.5, huberT = 1.345) {
  switch (id) {
    case 'huber':
      return new HuberRegression({ t: huberT });
    case 'median':
      return new QuantileRegression({ quantile: 0.5 });
    case 'quantile':
      return new QuantileRegression({ quantile: tau });
    case 'theilsen':
      return new TheilSen();
    case 'poisson':
      return new Glm({ family: 'poisson' });
    default:
      return new LinearRegression();
  }
}

export const robustSpec: RegressorLabSpec = {
  id: 'robust-glm',
  create: (p) => makeModel(str(p, 'model') as RobustModelId, num(p, 'tau'), num(p, 'huberT')),
  params,
  defaults: { model: 'ols', tau: 0.9, huberT: 1.345 },
  dataset: 'outliers',
  data: { n: 80, noise: 0.3 },
  experiments: [
    {
      id: 'ols-outliers',
      title: 'Least squares and outliers',
      text: 'A few wild points drag the least-squares line off the true intercept (0.5).',
      params: { model: 'ols' },
      claim: (run) => {
        const m = run.model instanceof LinearRegression ? run.model : null;
        return !!m && Math.abs(m.intercept - 0.5) > 0.25;
      },
    },
    {
      id: 'huber',
      title: 'Huber shrugs them off',
      text: 'Huber treats big residuals linearly instead of quadratically and gives outliers a small weight: the line returns to the trend.',
      params: { model: 'huber' },
      claim: (run) => {
        const m = run.model instanceof HuberRegression ? run.model : null;
        return !!m && Math.abs(m.intercept - 0.5) < 0.2 && Math.abs((m.coef[0] ?? 0) - 0.9) < 0.1;
      },
    },
    {
      id: 'median',
      title: 'The median line',
      text: 'Minimising absolute errors fits the median: half the points lie above, half below, however far the outliers are.',
      params: { model: 'median' },
      claim: (run) => {
        const m = run.model instanceof QuantileRegression ? run.model : null;
        return !!m && Math.abs(m.intercept - 0.5) < 0.2 && Math.abs((m.coef[0] ?? 0) - 0.9) < 0.1;
      },
    },
    {
      id: 'theil-sen',
      title: 'Theil–Sen: median of slopes',
      text: 'Take the slope of every pair of points and keep the median: robust to almost 30% outliers.',
      params: { model: 'theilsen' },
      claim: (run) => {
        const m = run.model instanceof TheilSen ? run.model : null;
        return !!m && Math.abs(m.slope - 0.9) < 0.1;
      },
    },
    {
      id: 'quantile-90',
      title: 'The 90th percentile line',
      text: 'Where the noise grows with x, the upper quantile fans away from the median: regression can describe more than the average.',
      setup: { dataset: 'heteroscedastic', n: 200 },
      params: { model: 'quantile', tau: 0.9 },
      claim: (run) => {
        const pred = run.model.predict(run.data.train.x);
        const below =
          Array.from(run.data.train.y).filter((v, i) => v <= (pred[i] as number) + 1e-9).length /
          run.data.train.y.length;
        return Math.abs(below - 0.9) < 0.07;
      },
    },
    {
      id: 'poisson',
      title: 'Counts need a Poisson model',
      text: 'For counts a straight line predicts impossible negative values; a Poisson GLM bends exponentially and stays positive.',
      setup: { dataset: 'counts', n: 150 },
      params: { model: 'poisson' },
      claim: (run) => {
        const { x } = run.data.all;
        const pred = run.model.predict(x);
        const ols = new LinearRegression().fit(run.data.train.x, run.data.train.y).predict(x);
        return pred.every((v) => v > 0) && ols.some((v) => v < 0);
      },
    },
  ],
};
