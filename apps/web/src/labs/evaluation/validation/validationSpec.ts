import type { ParamSpec } from '@/shared/lab-kit';
import type {
  EstimatorsRun,
  LeakageRun,
  SelectionRun,
  ValidationRun,
  ValidationScenario,
} from './validationCore';

export const validationParams: ParamSpec[] = [
  {
    key: 'kind',
    label: 'What to test',
    kind: 'segmented',
    options: [
      { value: 'estimators', label: 'Validation methods' },
      { value: 'selection', label: 'Picking the best model' },
      { value: 'leakage', label: 'Data leakage' },
    ],
  },
  {
    key: 'n',
    label: 'Data points',
    kind: 'ladder',
    values: [20, 40, 100, 300],
    format: (v) => String(v),
    showIf: (v) => v.kind === 'estimators',
  },
  {
    key: 'neighbors',
    label: 'Neighbours k of the model',
    kind: 'ladder',
    values: [1, 5, 15],
    format: (v) => String(v),
    showIf: (v) => v.kind === 'estimators',
    hint: 'A small k is a flexible model; a large k a smooth one.',
  },
  {
    key: 'candidates',
    label: 'Candidate models tried',
    kind: 'ladder',
    values: [1, 10, 50, 500],
    format: (v) => String(v),
    showIf: (v) => v.kind === 'selection',
  },
  {
    key: 'signal',
    label: 'Share of real signal (candidates with skill, or informative features)',
    kind: 'ladder',
    values: [0, 0.1, 0.3, 0.6],
    format: (v) => `${Math.round(v * 100)}%`,
    showIf: (v) => v.kind === 'selection' || v.kind === 'leakage',
    hint: '0 = the labels are pure coin flips.',
  },
  {
    key: 'samples',
    label: 'Rows',
    kind: 'ladder',
    values: [30, 50, 100, 200],
    format: (v) => String(v),
    showIf: (v) => v.kind === 'leakage',
  },
  {
    key: 'features',
    label: 'Features',
    kind: 'ladder',
    values: [20, 100, 500],
    format: (v) => String(v),
    showIf: (v) => v.kind === 'leakage',
  },
  {
    key: 'keep',
    label: 'Features kept by the selection step',
    kind: 'ladder',
    values: [1, 10, 50],
    format: (v) => String(v),
    showIf: (v) => v.kind === 'leakage',
  },
];

export interface ValidationExperiment {
  id: string;
  title: string;
  text: string;
  scenario: Partial<ValidationScenario>;
  claim: (run: ValidationRun) => boolean;
}

const est = (r: ValidationRun) => r as EstimatorsRun;
const sel = (r: ValidationRun) => r as SelectionRun;
const leak = (r: ValidationRun) => r as LeakageRun;
const stat = (r: EstimatorsRun, name: string) => r.stats.find((s) => s.name === name)!;

export const validationExperiments: ValidationExperiment[] = [
  {
    id: 'noisy-holdout',
    title: 'One split is a lottery',
    text: 'The same model, the same data, 60 different random splits: a single 20% hold-out gives estimates that scatter far more than 5-fold cross-validation, which averages five different splits. Which one you happened to get can change the story.',
    scenario: { kind: 'estimators' },
    claim: (r) => stat(est(r), 'holdout 20%').std > 1.4 * stat(est(r), '5-fold').std,
  },
  {
    id: 'pessimistic',
    title: 'Holding out half is pessimistic',
    text: 'Training on only half of the rows gives a model that is worse than the one you will deploy, so a 50% hold-out underestimates its quality (here by several points); 10-fold, which trains on 90%, is nearly unbiased.',
    scenario: { kind: 'estimators', neighbors: 15 },
    claim: (r) => stat(est(r), 'holdout 50%').bias < -0.04 && stat(est(r), '10-fold').bias > -0.02,
  },
  {
    id: 'tiny-data',
    title: 'Very little data',
    text: 'With 20 rows every estimator is noisy; a single 20% hold-out tests the model on just 4 rows and swings by about ten points.',
    scenario: { kind: 'estimators', n: 20 },
    claim: (r) => stat(est(r), 'holdout 20%').std > 0.1,
  },
  {
    id: 'lots-of-data',
    title: 'Plenty of data',
    text: 'With 300 rows all the methods agree and the scatter is small: the choice of validation method matters most when data are scarce.',
    scenario: { kind: 'estimators', n: 300 },
    claim: (r) => {
      const means = est(r).stats.map((s) => s.mean);
      return (
        Math.max(...means) - Math.min(...means) < 0.01 && est(r).stats.every((s) => s.std < 0.035)
      );
    },
  },
  {
    id: 'one-pick',
    title: 'Even one choice is optimistic',
    text: 'Every candidate here is a single-feature rule whose direction is chosen on the validation set. With just one candidate and labels that are pure noise, the validation accuracy is already above chance, and on fresh data it is exactly 50%.',
    scenario: { kind: 'selection', candidates: 1 },
    claim: (r) => sel(r).winnerMean > 0.54 && Math.abs(sel(r).testMean - 0.5) < 0.02,
  },
  {
    id: 'best-of-500',
    title: 'The best of 500 coin flips',
    text: 'Try 500 candidate rules on 60 rows of pure noise and keep the best: its validation score looks like real skill (over 70%). On fresh data it is 50%. The more models you compare on the same rows, the more the winner’s score is luck.',
    scenario: { kind: 'selection', candidates: 500 },
    claim: (r) => sel(r).winnerMean > 0.68 && Math.abs(sel(r).testMean - 0.5) < 0.02,
  },
  {
    id: 'real-skill',
    title: 'Genuine skill is overstated too',
    text: 'When 10% of the candidates have real skill, the winner is genuinely better than chance on fresh data, but its validation score still overstates it: the winner is partly good and partly lucky.',
    scenario: { kind: 'selection', candidates: 500, signal: 0.1 },
    claim: (r) => sel(r).testMean > 0.6 && sel(r).optimism > 0.1,
  },
  {
    id: 'leak',
    title: 'Selecting features on all the data',
    text: 'Fifty rows, five hundred features, labels that are pure noise. Choose the ten features that best match the labels using every row, then cross-validate: the accuracy looks like 85%. Redo the selection inside every training fold and it is 50%, which is the truth.',
    scenario: { kind: 'leakage' },
    claim: (r) =>
      leak(r).leakyMean > 0.8 &&
      Math.abs(leak(r).honestMean - 0.5) < 0.05 &&
      leak(r).independent < 0.55,
  },
  {
    id: 'leak-more',
    title: 'The more you keep, the more you leak',
    text: 'Keeping 50 features makes the leak worse: the selection step has more freedom to memorise which columns happened to match the validation rows.',
    scenario: { kind: 'leakage', keep: 50 },
    claim: (r) => leak(r).leakyMean > 0.9,
  },
  {
    id: 'leak-real',
    title: 'With real signal the honest score is real',
    text: 'If a few features truly carry signal, the honest cross-validation score matches what an independent test set gives, while the leaky one is still inflated.',
    scenario: { kind: 'leakage', signal: 0.6 },
    claim: (r) =>
      Math.abs(leak(r).honestMean - leak(r).independent) < 0.05 &&
      leak(r).leakyMean > leak(r).honestMean + 0.05,
  },
];
