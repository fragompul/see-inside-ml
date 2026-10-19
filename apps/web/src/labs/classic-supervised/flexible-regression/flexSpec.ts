import { IsotonicRegression, LinearRegression, Mars, PSpline } from '@sim/classic';
import { num, str, type ParamSpec, type RegressorLabSpec } from '@/shared/lab-kit';

export type FlexModelId = 'linear' | 'isotonic' | 'spline' | 'mars';

export const MODELS: { value: FlexModelId; label: string }[] = [
  { value: 'linear', label: 'Straight line' },
  { value: 'isotonic', label: 'Isotonic (monotone steps)' },
  { value: 'spline', label: 'Penalised spline' },
  { value: 'mars', label: 'MARS (hinge functions)' },
];

const params: ParamSpec[] = [
  { key: 'model', label: 'Model', kind: 'select', options: MODELS },
  {
    key: 'increasing',
    label: 'Increasing',
    kind: 'switch',
    showIf: (v) => v.model === 'isotonic',
    hint: 'Off fits a non-increasing function instead.',
  },
  {
    key: 'lambda',
    label: 'Smoothing penalty λ',
    kind: 'ladder',
    values: [0.001, 0.1, 1, 10, 100, 1e5],
    format: (v) => (v >= 1e4 ? '10⁵' : String(v)),
    showIf: (v) => v.model === 'spline',
    hint: 'How strongly bends are penalised.',
  },
  {
    key: 'terms',
    label: 'Most terms before pruning',
    kind: 'slider',
    min: 3,
    max: 15,
    step: 2,
    format: (v) => String(v),
    showIf: (v) => v.model === 'mars',
  },
];

export function makeModel(id: FlexModelId, increasing: boolean, lambda: number, terms: number) {
  switch (id) {
    case 'isotonic':
      return new IsotonicRegression({ increasing });
    case 'spline':
      return new PSpline({ bumps: 15, lambda });
    case 'mars':
      return new Mars({ maxTerms: terms, degree: 1, knots: 14, penalty: 2 });
    default:
      return new LinearRegression();
  }
}

function sse(pred: ArrayLike<number>, y: ArrayLike<number>): number {
  let s = 0;
  for (let i = 0; i < y.length; i++) s += ((y[i] as number) - (pred[i] as number)) ** 2;
  return s;
}

function total(y: ArrayLike<number>): number {
  const mean = Array.from(y).reduce((a, v) => a + v, 0) / y.length;
  let s = 0;
  for (let i = 0; i < y.length; i++) s += ((y[i] as number) - mean) ** 2;
  return s;
}

type Run = Parameters<RegressorLabSpec['experiments'][number]['claim']>[0];

/** R² on the held-out points of a model fitted to the training points of the run. */
const heldOutR2 = (
  run: Run,
  model: { fit: (x: never, y: never) => unknown; predict: (x: never) => ArrayLike<number> },
) => {
  (model.fit as (x: unknown, y: unknown) => unknown)(run.data.train.x, run.data.train.y);
  return 1 - sse(model.predict(run.data.test.x as never), run.data.test.y) / total(run.data.test.y);
};

export const flexSpec: RegressorLabSpec = {
  id: 'flexible-regression',
  create: (p) =>
    makeModel(
      str(p, 'model') as FlexModelId,
      Boolean(p.increasing),
      num(p, 'lambda'),
      num(p, 'terms'),
    ),
  params,
  defaults: { model: 'isotonic', increasing: true, lambda: 1, terms: 11 },
  dataset: 'step',
  data: { n: 200, noise: 0.25 },
  experiments: [
    {
      id: 'isotonic-step',
      title: 'Isotonic regression finds the steps',
      text: 'Only the order is assumed: the fit goes up, never down. On a step the fitted staircase has a handful of pieces and explains 97% of the variation in new points, where a straight line explains 72%.',
      params: { model: 'isotonic', increasing: true },
      claim: (run) => {
        const m = run.model instanceof IsotonicRegression ? run.model : null;
        const line = heldOutR2(run, new LinearRegression());
        return (
          !!m &&
          m.pieces <= 15 &&
          (run.scores.testR2 ?? 0) >= 0.95 &&
          (run.scores.testR2 ?? 0) > line + 0.2
        );
      },
    },
    {
      id: 'isotonic-direction',
      title: 'The direction has to be right',
      text: 'Told to expect a decrease, isotonic regression follows the falling arm of the parabola (about 89% of the variation in new points); told to expect an increase it can only draw a flat line.',
      setup: { dataset: 'curve' },
      params: { model: 'isotonic', increasing: false },
      claim: (run) => {
        const up = heldOutR2(run, new IsotonicRegression({ increasing: true }));
        return (run.scores.testR2 ?? 0) > 0.8 && up < 0.1;
      },
    },
    {
      id: 'isotonic-limit',
      title: 'A monotone model cannot follow a wave',
      text: 'On a sine wave the best a model that may only go one way can do is trace one arm: 43% of the variation at most. A penalised spline explains 96%.',
      setup: { dataset: 'sine' },
      params: { model: 'isotonic', increasing: false },
      claim: (run) => {
        const spline = heldOutR2(run, new PSpline({ bumps: 15, lambda: 1 }));
        return (run.scores.testR2 ?? 1) < 0.5 && spline > 0.9;
      },
    },
    {
      id: 'spline-penalty',
      title: 'The penalty turns a curve into a line',
      text: 'With a small penalty the spline follows the sine wave (96% of the variation explained). Raise it and the curve is flattened: at 100 it explains 58%, and with an enormous penalty it is exactly the straight line.',
      setup: { dataset: 'sine' },
      params: { model: 'spline', lambda: 1e5 },
      claim: (run) => {
        const line = new LinearRegression()
          .fit(run.data.train.x, run.data.train.y)
          .predict(run.data.test.x);
        const mine = run.model.predict(run.data.test.x);
        const wavy = heldOutR2(run, new PSpline({ bumps: 15, lambda: 1 }));
        return mine.every((v, i) => Math.abs(v - (line[i] as number)) < 0.01) && wavy > 0.9;
      },
    },
    {
      id: 'mars-sine',
      title: 'MARS builds a wave from hinges',
      text: 'The forward pass adds pairs of hinge functions where they help most, up to eleven terms; the backward pass then deletes the ones that do not pay for themselves under cross-validation. On the sine wave five terms remain and explain 94% of the variation in new points.',
      setup: { dataset: 'sine' },
      params: { model: 'mars', terms: 11 },
      claim: (run) => {
        const m = run.model instanceof Mars ? run.model : null;
        return !!m && m.terms.length < 9 && m.terms.length >= 3 && (run.scores.testR2 ?? 0) > 0.9;
      },
    },
    {
      id: 'mars-line',
      title: 'And knows when a line will do',
      text: 'When the truth is a straight line the backward pass removes almost every hinge: the pruned model has at most three terms and matches the accuracy of the line.',
      setup: { dataset: 'line' },
      params: { model: 'mars', terms: 11 },
      claim: (run) => {
        const m = run.model instanceof Mars ? run.model : null;
        const line = heldOutR2(run, new LinearRegression());
        return !!m && m.terms.length <= 3 && Math.abs((run.scores.testR2 ?? 0) - line) < 0.02;
      },
    },
  ],
};
