import type { ParamSpec } from '@/shared/lab-kit';
import {
  METHOD_LABELS,
  runFeatures,
  type FeaturesRun,
  type FeaturesScenario,
  type SelectionMethod,
} from './featuresCore';

export const featuresParams: ParamSpec[] = [
  {
    key: 'method',
    label: 'How to rank the features',
    kind: 'select',
    options: (Object.keys(METHOD_LABELS) as SelectionMethod[]).map((m) => ({
      value: m,
      label: METHOD_LABELS[m],
    })),
  },
  {
    key: 'keep',
    label: 'Features kept',
    kind: 'slider',
    min: 1,
    max: 12,
    step: 1,
    hint: 'The model may use only this many of the 20 columns.',
  },
  {
    key: 'n',
    label: 'Training rows',
    kind: 'ladder',
    values: [40, 80, 150, 400],
    format: (v) => String(v),
  },
  {
    key: 'copyNoise',
    label: 'How different a copy is from its original',
    kind: 'ladder',
    values: [0.05, 0.3, 1],
    format: (v) => String(v),
    hint: 'Small = a near-exact duplicate column.',
  },
];

export interface FeaturesExperiment {
  id: string;
  title: string;
  text: string;
  scenario: Partial<FeaturesScenario>;
  claim: (run: FeaturesRun) => boolean;
}

const withMethod = (r: FeaturesRun, method: SelectionMethod): FeaturesRun =>
  runFeatures({ ...r.scenario, method });

export const featuresExperiments: FeaturesExperiment[] = [
  {
    id: 'filter-redundant',
    title: 'The filter picks twins',
    text: 'The target depends on four signals; four more columns are noisy copies of them and twelve are pure noise. Ranking each feature on its own by correlation, the top four contain a signal and its own copy: one real signal is missed and the model is worse than one that knows the truth.',
    scenario: { method: 'filter', keep: 4 },
    claim: (r) => r.redundant >= 1 && r.signalsCovered <= 3 && r.testR2 < r.oracleR2 - 0.03,
  },
  {
    id: 'forward-clean',
    title: 'Forward selection skips duplicates',
    text: 'Forward selection asks what each feature adds to the ones already chosen, so once a signal is in, its copy brings nothing new. Its four picks cover all four signals and match a model that knows the truth.',
    scenario: { method: 'forward', keep: 4 },
    claim: (r) => r.signalsCovered === 4 && r.redundant === 0 && r.testR2 > r.oracleR2 - 0.01,
  },
  {
    id: 'near-duplicates',
    title: 'Near-exact duplicates',
    text: 'When the copies are almost identical to the originals (noise 0.05), the filter fills its list with pairs and covers only two signals. The L1 penalty of the lasso, which prefers one member of a correlated group, and forward selection both cover all four.',
    scenario: { method: 'filter', keep: 4, copyNoise: 0.05 },
    claim: (r) =>
      r.signalsCovered === 2 &&
      r.testR2 < 0.8 &&
      withMethod(r, 'lasso').signalsCovered === 4 &&
      withMethod(r, 'forward').signalsCovered === 4,
  },
  {
    id: 'importance-shared',
    title: 'Importance is shared between twins',
    text: "A random forest can use either a signal or its copy, so its importance is split between them: both rank high, and the forest's top four again contain twins. Permutation importance gives the same picture: shuffling one twin hurts less because the other still carries the information.",
    scenario: { method: 'forest', keep: 4 },
    claim: (r) =>
      r.redundant >= 1 && withMethod(r, 'permutation').redundant >= 1 && r.noisePicked === 0,
  },
  {
    id: 'too-many',
    title: 'Keeping too many',
    text: 'Beyond the four real signals every extra feature is noise. Forward selection with eight slots starts adding noise columns; the test score does not improve and, with less data, would get worse.',
    scenario: { method: 'forward', keep: 8 },
    claim: (r) => r.noisePicked >= 2 && Math.abs(r.testR2 - r.oracleR2) < 0.005,
  },
  {
    id: 'lasso-path',
    title: 'The lasso picks one of each pair',
    text: 'The lasso zeroes out one member of a group of near-duplicates, so with four slots it covers all four signals, without a model being refitted for every candidate.',
    scenario: { method: 'lasso', keep: 4, copyNoise: 0.05 },
    claim: (r) => r.signalsCovered === 4 && r.redundant === 0 && r.testR2 > r.oracleR2 - 0.01,
  },
  {
    id: 'little-data',
    title: 'With little data',
    text: "With only 40 rows the cheap rankings get noisier: the lasso's four picks miss signals and the model is much worse. Forward selection, which evaluates each addition by cross-validation, still finds all four.",
    scenario: { method: 'lasso', keep: 4, n: 40 },
    claim: (r) => r.testR2 < 0.8 && withMethod(r, 'forward').signalsCovered === 4,
  },
];
