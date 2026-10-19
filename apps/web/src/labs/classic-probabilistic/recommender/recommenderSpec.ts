import type { ParamSpec } from '@/shared/lab-kit';
import type { RecommenderRun, RecommenderScenario } from './recommenderCore';

export const recommenderParams: ParamSpec[] = [
  {
    key: 'observed',
    label: 'Share of ratings we know',
    kind: 'ladder',
    values: [0.05, 0.1, 0.2, 0.3, 0.45, 0.6],
    format: (v) => `${Math.round(v * 100)}%`,
    hint: 'The rest is what the model must predict.',
  },
  {
    key: 'factors',
    label: 'Taste dimensions',
    kind: 'slider',
    min: 1,
    max: 8,
    step: 1,
    hint: 'How many hidden traits the model uses to describe people and films.',
  },
  {
    key: 'lambda',
    label: 'Regularisation',
    kind: 'ladder',
    values: [0.01, 0.1, 0.5, 1, 5, 20],
    format: (v) => String(v),
    hint: 'A penalty on large factors: keeps the model from memorising.',
  },
  {
    key: 'method',
    label: 'Optimiser',
    kind: 'segmented',
    options: [
      { value: 'als', label: 'Alternating least squares' },
      { value: 'sgd', label: 'Gradient descent' },
    ],
  },
  {
    key: 'iterations',
    label: 'Iterations',
    kind: 'slider',
    min: 1,
    max: 80,
    step: 1,
  },
  {
    key: 'noise',
    label: 'Noise in people’s ratings',
    kind: 'slider',
    min: 0,
    max: 1.2,
    step: 0.1,
    format: (v) => v.toFixed(1),
    hint: 'How inconsistent people are with their own taste.',
  },
];

export interface RecommenderExperiment {
  id: string;
  title: string;
  text: string;
  scenario: Partial<RecommenderScenario>;
  claim: (run: RecommenderRun) => boolean;
}

const last = (v: number[]): number => v[v.length - 1] as number;

export const recommenderExperiments: RecommenderExperiment[] = [
  {
    id: 'taste',
    title: 'Hidden tastes',
    text: 'Knowing only 30% of the ratings, three hidden taste dimensions predict the rest with an error about a quarter smaller than guessing the global average and well below the average rating of each film.',
    scenario: {},
    claim: (r) => last(r.testRmse) < r.baselineMean * 0.75 && last(r.testRmse) < r.baselineItem,
  },
  {
    id: 'one-factor',
    title: 'One dimension is too few',
    text: 'With a single taste dimension the model can say who likes things more or less, but not who likes which kind of film. It under-fits: even the ratings it has seen are not reproduced well.',
    scenario: { factors: 1 },
    claim: (r) => last(r.trainRmse) > 0.5 && last(r.testRmse) > 0.62,
  },
  {
    id: 'memorise',
    title: 'Memorising the ratings',
    text: 'Eight dimensions with almost no penalty are enough to reproduce every known rating. The model has learned the noise: on unseen ratings it is worse than just guessing the average.',
    scenario: { factors: 8, lambda: 0.01 },
    claim: (r) => last(r.trainRmse) < 0.05 && last(r.testRmse) > r.baselineMean,
  },
  {
    id: 'too-much-penalty',
    title: 'Too much regularisation',
    text: 'A huge penalty squeezes every factor to zero, and the model falls back to predicting the global average for everyone.',
    scenario: { lambda: 20 },
    claim: (r) => Math.abs(last(r.testRmse) - r.baselineMean) < 0.02,
  },
  {
    id: 'sparse',
    title: 'Almost nothing known',
    text: 'With only 5% of the ratings, most films have one or two ratings and many have none. The factors barely beat the global average, and the item average is even worse than it (it trusts single ratings).',
    scenario: { observed: 0.05 },
    claim: (r) => last(r.testRmse) > r.baselineMean * 0.9 && r.baselineItem > r.baselineMean,
  },
  {
    id: 'more-data',
    title: 'More ratings, better guesses',
    text: 'Knowing 60% of the table, the same model predicts the remaining ratings with an error under 0.55 stars.',
    scenario: { observed: 0.6 },
    claim: (r) => last(r.testRmse) < 0.55,
  },
  {
    id: 'noisy',
    title: 'Inconsistent people',
    text: 'If people rate almost at random around their taste, no model can predict them well; a model that trusts the noisy ratings does worse than the plain average.',
    scenario: { noise: 1 },
    claim: (r) => last(r.testRmse) > r.baselineMean,
  },
  {
    id: 'one-pass',
    title: 'One pass is not enough',
    text: 'After a single round of alternating least squares the factors are still crude. The error falls quickly over the next few iterations and then levels off.',
    scenario: { iterations: 1 },
    claim: (r) => last(r.testRmse) > 0.7,
  },
  {
    id: 'sgd',
    title: 'The same idea by gradient descent',
    text: 'Instead of solving each user and film exactly, nudge the factors a little after each rating. It takes more passes but ends up just as good.',
    scenario: { method: 'sgd', iterations: 60 },
    claim: (r) => last(r.testRmse) < 0.6,
  },
];
