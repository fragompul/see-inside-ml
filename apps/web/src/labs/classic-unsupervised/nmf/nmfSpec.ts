import type { ParamSpec } from '@/shared/lab-kit';
import { partMeanCorrelation, type NmfRun, type NmfScenario } from './nmfCore';

export const nmfParams: ParamSpec[] = [
  {
    key: 'components',
    label: 'Parts',
    kind: 'ladder',
    values: [1, 2, 4, 6, 10, 16, 25, 40],
    format: (v) => String(v),
    hint: 'How many building blocks the digits are made of.',
  },
  {
    key: 'loss',
    label: 'Error measure',
    kind: 'segmented',
    options: [
      { value: 'frobenius', label: 'Squared error' },
      { value: 'kl', label: 'KL divergence' },
    ],
  },
  {
    key: 'iterations',
    label: 'Iterations',
    kind: 'ladder',
    values: [1, 5, 10, 30, 100, 200],
    format: (v) => String(v),
  },
  {
    key: 'digits',
    label: 'Digits',
    kind: 'segmented',
    options: [
      { value: 'all', label: 'All ten' },
      { value: 'few', label: '0, 1, 4, 7' },
    ],
  },
];

export interface NmfExperiment {
  id: string;
  title: string;
  text: string;
  scenario: Partial<NmfScenario>;
  claim: (run: NmfRun) => boolean;
}

export const nmfExperiments: NmfExperiment[] = [
  {
    id: 'parts',
    title: 'Digits made of strokes',
    text: 'Ten non-negative parts can only be added, never subtracted, so each learns a stroke or a blob that many digits share. PCA components (right) mix light and dark and are far less sparse.',
    scenario: { components: 10 },
    claim: (r) => r.nmfSparsity > 0.55 && r.pcaSparsity < 0.5 && r.pcaNegative > 0.4,
  },
  {
    id: 'one-part',
    title: 'One part: the average digit',
    text: 'With a single part the best additive description of every digit is “the average digit, scaled”: the part is the mean picture.',
    scenario: { components: 1 },
    claim: (r) => partMeanCorrelation(r) > 0.99,
  },
  {
    id: 'pca-better',
    title: 'PCA reconstructs better',
    text: 'PCA is the best possible linear compression for squared error, and it is allowed negative weights. NMF pays a little accuracy for parts you can read.',
    scenario: { components: 10 },
    claim: (r) => r.pcaRelError < r.nmfRelError && r.nmfRelError - r.pcaRelError < 0.1,
  },
  {
    id: 'many-parts',
    title: 'More parts, finer strokes',
    text: 'With 40 parts the error drops a lot and the parts become even sparser: tiny local strokes rather than whole shapes.',
    scenario: { components: 40 },
    claim: (r) => r.nmfRelError < 0.3 && r.nmfSparsity > 0.7,
  },
  {
    id: 'unfinished',
    title: 'Stopped too early',
    text: 'After 5 multiplicative updates the parts are still blurry blends and the error is far above what the same model reaches when it converges.',
    scenario: { components: 10, iterations: 5 },
    claim: (r) => r.nmfRelError > 0.58,
  },
  {
    id: 'kl',
    title: 'A different error measure',
    text: 'Using the KL divergence instead of squared error gives parts of the same kind (it treats pixel intensities as counts), with a slightly different trade-off.',
    scenario: { components: 10, loss: 'kl' },
    claim: (r) => r.nmfSparsity > 0.55 && r.nmfRelError > 0.4 && r.nmfRelError < 0.5,
  },
  {
    id: 'few-digits',
    title: 'Four digits, four parts',
    text: 'Restricted to 0, 1, 4 and 7, a handful of parts is enough and each one is easy to read as a piece of one of the digits.',
    scenario: { components: 4, digits: 'few' },
    claim: (r) => r.nmfRelError < 0.52 && r.nmfSparsity > 0.5,
  },
];
