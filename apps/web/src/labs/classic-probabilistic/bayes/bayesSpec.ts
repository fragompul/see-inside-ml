import type { ParamSpec } from '@/shared/lab-kit';
import { intervalWidth, type BayesModel, type BayesRun, type BayesScenario } from './bayesCore';

export const bayesParams: ParamSpec[] = [
  {
    key: 'model',
    label: 'Question',
    kind: 'segmented',
    options: [
      { value: 'coin', label: 'Is the coin fair?' },
      { value: 'mean', label: 'What is the true value?' },
      { value: 'rate', label: 'How often does it happen?' },
    ],
  },
  {
    key: 'priorCoin',
    label: 'Prior belief: chance of heads',
    kind: 'slider',
    min: 0.05,
    max: 0.95,
    step: 0.05,
    format: (v) => v.toFixed(2),
    showIf: (v) => v.model === 'coin',
  },
  {
    key: 'priorMean',
    label: 'Prior belief: the value',
    kind: 'slider',
    min: -10,
    max: 10,
    step: 0.5,
    format: (v) => v.toFixed(1),
    showIf: (v) => v.model === 'mean',
  },
  {
    key: 'priorRate',
    label: 'Prior belief: events per interval',
    kind: 'slider',
    min: 0.5,
    max: 8,
    step: 0.5,
    format: (v) => v.toFixed(1),
    showIf: (v) => v.model === 'rate',
  },
  {
    key: 'priorStrength',
    label: 'How strongly you believe it',
    kind: 'ladder',
    values: [1, 2, 5, 10, 20, 50, 100],
    format: (v) => `worth ${v} observations`,
    hint: 'The prior counts as this many imaginary observations.',
  },
  {
    key: 'n',
    label: 'Observations',
    kind: 'ladder',
    values: [0, 1, 3, 5, 10, 20, 50, 100, 300, 1000],
    format: (v) => String(v),
  },
  {
    key: 'obsCoin',
    label: 'Data: share of heads',
    kind: 'slider',
    min: 0,
    max: 1,
    step: 0.05,
    format: (v) => v.toFixed(2),
    showIf: (v) => v.model === 'coin',
  },
  {
    key: 'obsMean',
    label: 'Data: average measurement',
    kind: 'slider',
    min: -10,
    max: 10,
    step: 0.5,
    format: (v) => v.toFixed(1),
    showIf: (v) => v.model === 'mean',
  },
  {
    key: 'obsRate',
    label: 'Data: average count',
    kind: 'slider',
    min: 0,
    max: 10,
    step: 0.5,
    format: (v) => v.toFixed(1),
    showIf: (v) => v.model === 'rate',
  },
];

export interface BayesExperiment {
  id: string;
  title: string;
  text: string;
  scenario: BayesScenario;
  claim: (run: BayesRun) => boolean;
}

const coin = (
  priorCenter: number,
  strength: number,
  n: number,
  observed: number,
): BayesScenario => ({
  model: 'coin',
  priorCenter: { coin: priorCenter, mean: 0, rate: 1 },
  priorStrength: strength,
  n,
  observed: { coin: observed, mean: 6, rate: 3 },
});

export const bayesExperiments: BayesExperiment[] = [
  {
    id: 'first-flips',
    title: 'Ten flips, seven heads',
    text: 'With no strong opinion (a flat prior), ten flips cannot pin the coin down: 7 heads is the best estimate, but the 95% credible interval is wide and includes a fair coin.',
    scenario: coin(0.5, 2, 10, 0.7),
    claim: (r) => r.mle === 0.7 && intervalWidth(r) > 0.4 && r.interval[0] < 0.5,
  },
  {
    id: 'more-data',
    title: 'A thousand flips',
    text: 'The same proportion over 1000 flips leaves little room for doubt: the belief narrows to a spike around 0.7 and a fair coin is ruled out.',
    scenario: coin(0.5, 2, 1000, 0.7),
    claim: (r) =>
      intervalWidth(r) < 0.06 && r.interval[0] > 0.5 && Math.abs(r.posterior.mean - 0.7) < 0.01,
  },
  {
    id: 'stubborn',
    title: 'A stubborn prior',
    text: 'You are sure the coin favours heads (prior worth 50 flips) and then see only 3 heads in 10. The posterior stays close to your prior: ten observations cannot outvote fifty imaginary ones.',
    scenario: coin(0.8, 50, 10, 0.3),
    claim: (r) => r.posterior.mean > 0.6 && r.priorWeight > 0.8,
  },
  {
    id: 'data-wins',
    title: 'Data outvotes the prior',
    text: 'With the same stubborn prior but 1000 flips, the data dominate: the posterior lands almost exactly on the observed 30%, and the prior is forgotten.',
    scenario: coin(0.8, 50, 1000, 0.3),
    claim: (r) => Math.abs(r.posterior.mean - 0.3) < 0.03 && r.priorWeight < 0.06,
  },
  {
    id: 'no-heads',
    title: 'Zero heads is not impossible',
    text: 'Five tails in a row: the data alone say the chance of heads is exactly 0. A Bayesian posterior keeps some room for heads, with a mean of 1/7 and an upper limit well above 0.3.',
    scenario: coin(0.5, 2, 5, 0),
    claim: (r) => r.mle === 0 && Math.abs(r.posterior.mean - 1 / 7) < 1e-9 && r.interval[1] > 0.3,
  },
  {
    id: 'no-data',
    title: 'Before any data',
    text: 'With zero observations the posterior is just the prior: you start with what you believe.',
    scenario: coin(0.3, 10, 0, 0.5),
    claim: (r) => Math.abs(r.posterior.mean - r.prior.mean) < 1e-12 && r.priorWeight === 1,
  },
  {
    id: 'measure',
    title: 'Combining a measurement with a hunch',
    text: 'You guessed the value is about 0, then took four measurements averaging 6. The posterior is a precision-weighted compromise, much closer to the data (more precise) than to the hunch, and narrower than either.',
    scenario: {
      model: 'mean',
      priorCenter: { coin: 0.5, mean: 0, rate: 1 },
      priorStrength: 1,
      n: 4,
      observed: { coin: 0.7, mean: 6, rate: 3 },
    },
    claim: (r) =>
      r.posterior.mean > 4 &&
      r.posterior.mean < 5.5 &&
      Math.sqrt(r.posterior.variance) < Math.sqrt(r.prior.variance),
  },
  {
    id: 'counts',
    title: 'Counting events',
    text: 'You expected about one event per interval (a weak prior) and saw an average of 3 over 20 intervals. The belief moves from your guess to just below the observed rate.',
    scenario: {
      model: 'rate',
      priorCenter: { coin: 0.5, mean: 0, rate: 1 },
      priorStrength: 2,
      n: 20,
      observed: { coin: 0.7, mean: 6, rate: 3 },
    },
    claim: (r) => Math.abs(r.posterior.mean - 62 / 22) < 1e-9 && r.posterior.mean < 3,
  },
];

export const modelOf = (s: BayesScenario): BayesModel => s.model;
