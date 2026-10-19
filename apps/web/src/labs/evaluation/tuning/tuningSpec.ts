import { LANDSCAPES } from '@sim/classic';
import type { ParamSpec } from '@/shared/lab-kit';
import {
  METHODS,
  METHOD_LABELS,
  raceOf,
  type LandscapeId,
  type TuningRun,
  type TuningScenario,
} from './tuningCore';

export const tuningParams: ParamSpec[] = [
  {
    key: 'landscape',
    label: 'The score surface',
    kind: 'select',
    options: (Object.keys(LANDSCAPES) as LandscapeId[]).map((id) => ({
      value: id,
      label: LANDSCAPES[id]!.label,
    })),
  },
  {
    key: 'method',
    label: 'Search method shown',
    kind: 'segmented',
    options: METHODS.map((m) => ({ value: m, label: METHOD_LABELS[m] })),
  },
  {
    key: 'budget',
    label: 'Budget (full experiments)',
    kind: 'ladder',
    values: [6, 9, 16, 20, 40, 60, 80],
    format: (v) => String(v),
    hint: 'One full experiment = training and validating a model once.',
  },
  {
    key: 'noise',
    label: 'Noise in a validation score',
    kind: 'slider',
    min: 0,
    max: 0.4,
    step: 0.05,
    format: (v) => v.toFixed(2),
    hint: 'Random splits and random initialisation make every measurement a little different.',
  },
];

export interface TuningExperiment {
  id: string;
  title: string;
  text: string;
  scenario: Partial<TuningScenario>;
  claim: (run: TuningRun) => boolean;
}

export const tuningExperiments: TuningExperiment[] = [
  {
    id: 'easy',
    title: 'An easy surface',
    text: 'One broad optimum: every method ends within a few hundredths of the best achievable score. Nothing clever is needed when the problem is smooth.',
    scenario: {},
    claim: (r) => r.race.every((e) => e.meanRegret < 0.1),
  },
  {
    id: 'irrelevant',
    title: 'One setting that matters',
    text: 'Only the first hyperparameter changes the score. A 4 × 4 grid tests just four values of it, none close to the best; sixteen random trials test sixteen different values and get closer on average.',
    scenario: { landscape: 'oneThing', budget: 16 },
    claim: (r) => raceOf(r, 'grid').meanRegret > 1.5 * raceOf(r, 'random').meanRegret,
  },
  {
    id: 'ridge',
    title: 'A narrow ridge',
    text: 'Good settings lie on a thin diagonal band. A grid steps over it, random search stumbles on it now and then, and the methods that learn from earlier results (halving, Bayesian) get about twice as close.',
    scenario: { landscape: 'ridge' },
    claim: (r) =>
      raceOf(r, 'grid').meanRegret > 0.3 &&
      raceOf(r, 'bayesian').meanRegret < raceOf(r, 'random').meanRegret &&
      raceOf(r, 'halving').meanRegret < raceOf(r, 'random').meanRegret,
  },
  {
    id: 'needle',
    title: 'A needle in a haystack',
    text: 'A single small spike of good settings in a flat plain: with 20 experiments hardly anyone finds it. Every method misses by a wide margin and almost no run gets close.',
    scenario: { landscape: 'needle' },
    claim: (r) => r.race.every((e) => e.meanRegret > 0.2 && e.success < 0.35),
  },
  {
    id: 'needle-more',
    title: 'The needle with a bigger budget',
    text: 'With 80 experiments Bayesian optimisation finds the needle in most runs (it keeps probing near what scored well and where it knows least), successive halving in about half, random search rarely and a grid never.',
    scenario: { landscape: 'needle', budget: 80 },
    claim: (r) =>
      raceOf(r, 'bayesian').success > 0.6 &&
      raceOf(r, 'random').success < 0.25 &&
      raceOf(r, 'grid').success === 0,
  },
  {
    id: 'noisy',
    title: 'Noisy measurements',
    text: 'When a validation score wobbles by 0.3 from run to run, the best-looking configuration is often just a lucky one, and every method pays: even on the easy surface the final regret is several times larger than before.',
    scenario: { noise: 0.3 },
    claim: (r) => raceOf(r, 'bayesian').meanRegret > 0.08 && raceOf(r, 'random').meanRegret > 0.1,
  },
  {
    id: 'ridge-more',
    title: 'More budget, smarter search',
    text: 'On the ridge with 60 experiments, Bayesian optimisation reaches the ridge in four runs out of five and ends with less than a fifth of the regret of a grid.',
    scenario: { landscape: 'ridge', budget: 60 },
    claim: (r) =>
      raceOf(r, 'bayesian').success > 0.6 &&
      raceOf(r, 'bayesian').meanRegret < raceOf(r, 'grid').meanRegret / 3,
  },
  {
    id: 'tiny-budget',
    title: 'Six experiments',
    text: 'With a tiny budget, the model-based method has barely started learning: a plain grid, which covers the space evenly, does better than random or Bayesian search on the easy surface.',
    scenario: { budget: 6 },
    claim: (r) =>
      raceOf(r, 'grid').meanRegret < raceOf(r, 'random').meanRegret &&
      raceOf(r, 'grid').meanRegret < raceOf(r, 'bayesian').meanRegret,
  },
];
