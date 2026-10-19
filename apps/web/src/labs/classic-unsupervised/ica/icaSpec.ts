import type { ParamSpec } from '@/shared/lab-kit';
import {
  mean,
  worst,
  type IcaRun,
  type IcaScenario,
  SOURCE_LABELS,
  type SourceKind,
} from './icaCore';

export const icaParams: ParamSpec[] = [
  {
    key: 'sources',
    label: 'Sources',
    kind: 'segmented',
    options: (Object.keys(SOURCE_LABELS) as SourceKind[]).map((k) => ({
      value: k,
      label: SOURCE_LABELS[k],
    })),
  },
  {
    key: 'count',
    label: 'Sources and microphones',
    kind: 'segmented',
    options: [
      { value: '2', label: '2' },
      { value: '3', label: '3' },
    ],
  },
  {
    key: 'noise',
    label: 'Microphone noise',
    kind: 'slider',
    min: 0,
    max: 2,
    step: 0.1,
    format: (v) => v.toFixed(1),
    hint: 'Sources have unit variance; noise is added after mixing.',
  },
  {
    key: 'nonlinearity',
    label: 'Contrast function',
    kind: 'segmented',
    options: [
      { value: 'logcosh', label: 'log cosh' },
      { value: 'exp', label: 'Gaussian' },
      { value: 'cube', label: 'Kurtosis' },
    ],
    hint: 'How “non-Gaussian” is measured. log cosh is the all-round default.',
  },
  {
    key: 'samples',
    label: 'Time samples',
    kind: 'ladder',
    values: [40, 100, 200, 400, 800],
    format: (v) => String(v),
  },
];

export interface IcaExperiment {
  id: string;
  title: string;
  text: string;
  scenario: Partial<IcaScenario>;
  claim: (run: IcaRun) => boolean;
}

export const icaExperiments: IcaExperiment[] = [
  {
    id: 'cocktail',
    title: 'The cocktail party',
    text: 'Three instruments are heard through three microphones, each a different blend of all three. PCA finds uncorrelated mixtures that still blend them; ICA pulls out signals that match the originals almost perfectly.',
    scenario: { sources: 'waves', count: 3, noise: 0 },
    claim: (r) => worst(r.icaRecovery) > 0.98 && worst(r.pcaRecovery) < 0.8,
  },
  {
    id: 'tilted-square',
    title: 'A tilted square',
    text: 'Two independent flat noises, mixed, fill a tilted parallelogram. PCA picks perpendicular axes along the longest directions; ICA lines its axes up with the parallelogram’s sides, which are the sources.',
    scenario: { sources: 'uniform', count: 2, noise: 0 },
    claim: (r) => worst(r.icaRecovery) > 0.99 && worst(r.pcaRecovery) < 0.85,
  },
  {
    id: 'speech',
    title: 'Sparse, speech-like sources',
    text: 'Spiky sources are strongly non-Gaussian (high kurtosis), the easiest case for ICA: three voices are separated cleanly while PCA stays muddled.',
    scenario: { sources: 'laplace', count: 3, noise: 0 },
    claim: (r) =>
      worst(r.icaRecovery) > 0.99 && worst(r.pcaRecovery) < 0.7 && r.kurtosis.every((k) => k > 1),
  },
  {
    id: 'gaussian',
    title: 'Gaussian sources: nothing to find',
    text: 'Mixtures of Gaussian noises are Gaussian in every direction, so no rotation looks “more independent” than another. ICA returns some rotation, but not the original sources.',
    scenario: { sources: 'gaussian', count: 3, noise: 0 },
    claim: (r) => worst(r.icaRecovery) < 0.9,
  },
  {
    id: 'noisy',
    title: 'Noisy microphones',
    text: 'With noise on the microphones the model (a clean mix of the sources) no longer holds exactly. ICA still beats PCA on average, but the weakest source is no longer recovered well.',
    scenario: { sources: 'waves', count: 3, noise: 0.3 },
    claim: (r) => mean(r.icaRecovery) > mean(r.pcaRecovery) + 0.1 && worst(r.icaRecovery) < 0.9,
  },
  {
    id: 'drowned',
    title: 'Drowned in noise',
    text: 'When the noise is as loud as the signals, nothing separates the sources any more: both methods collapse.',
    scenario: { sources: 'waves', count: 3, noise: 1 },
    claim: (r) => mean(r.icaRecovery) < 0.65,
  },
  {
    id: 'few-samples',
    title: 'A short recording is enough',
    text: 'Only 60 time samples are enough here: the unmixing has just nine numbers to find.',
    scenario: { sources: 'waves', count: 3, noise: 0, samples: 60 },
    claim: (r) => worst(r.icaRecovery) > 0.97,
  },
];
