import { GaussianProcess, type GpKernel } from '@sim/classic';
import type { Matrix, Regressor } from '@sim/core';
import { flag, num, str, type ParamSpec, type RegressorLabSpec } from '@/shared/lab-kit';

export const LENGTHS = [0.1, 0.2, 0.4, 0.7, 1, 2, 5] as const;
export const NOISES = [0.001, 0.01, 0.05, 0.1, 0.3, 1] as const;

const params: ParamSpec[] = [
  {
    key: 'kernel',
    label: 'Kernel',
    kind: 'select',
    options: [
      { value: 'rbf', label: 'RBF (very smooth)' },
      { value: 'matern52', label: 'Matérn 5/2' },
      { value: 'matern32', label: 'Matérn 3/2 (rougher)' },
      { value: 'periodic', label: 'Periodic' },
    ],
  },
  {
    key: 'tune',
    label: 'Optimise the hyperparameters',
    kind: 'switch',
    hint: 'Maximise the marginal likelihood.',
  },
  {
    key: 'length',
    label: 'Length scale ℓ',
    kind: 'ladder',
    values: LENGTHS,
    format: (v) => String(v),
    showIf: (v) => !v.tune,
  },
  {
    key: 'variance',
    label: 'Signal variance σ²',
    kind: 'ladder',
    values: [0.1, 0.5, 1, 2, 5],
    format: (v) => String(v),
    showIf: (v) => !v.tune,
  },
  {
    key: 'noise',
    label: 'Noise variance',
    kind: 'ladder',
    values: NOISES,
    format: (v) => String(v),
    showIf: (v) => !v.tune,
  },
];

/** A Gaussian process that optionally tunes its own hyperparameters when fitted. */
export class TunedGp implements Regressor {
  readonly gp: GaussianProcess;
  likelihood = NaN;

  constructor(
    options: ConstructorParameters<typeof GaussianProcess>[0],
    private readonly tune: boolean,
  ) {
    this.gp = new GaussianProcess(options);
  }

  fit(x: Matrix, y: ArrayLike<number>): this {
    this.gp.fit(x, y);
    this.likelihood = this.tune ? this.gp.optimize() : this.gp.logMarginalLikelihood();
    return this;
  }

  predict(x: Matrix): Float64Array {
    return this.gp.predict(x);
  }
}

export const gpOf = (m: Regressor): TunedGp | null => (m instanceof TunedGp ? m : null);

export const gpSpec: RegressorLabSpec = {
  id: 'gaussian-process',
  create: (p) =>
    new TunedGp(
      {
        kernel: str(p, 'kernel') as GpKernel,
        lengthScale: num(p, 'length'),
        variance: num(p, 'variance'),
        noise: num(p, 'noise'),
        period: (2 * Math.PI) / 1.7,
      },
      flag(p, 'tune'),
    ),
  params,
  defaults: { kernel: 'rbf', tune: false, length: 1, variance: 1, noise: 0.1 },
  dataset: 'sine',
  data: { n: 30, noise: 0.25 },
  experiments: [
    {
      id: 'smooth',
      title: 'A good length scale',
      text: 'A moderate length scale follows the wave; the shaded band is the model’s own uncertainty.',
      params: { length: 1, noise: 0.1 },
      claim: (run) => (run.scores.testR2 ?? 0) > 0.75,
    },
    {
      id: 'short',
      title: 'Length scale too short',
      text: 'Each point only informs its immediate neighbours, so the curve spikes through the data and falls back to zero in between: overfitting, with wide uncertainty between points.',
      params: { length: 0.1, noise: 0.001 },
      claim: (run) => run.scores.trainR2 > 0.95 && (run.scores.testR2 ?? 1) < 0.6,
    },
    {
      id: 'long',
      title: 'Length scale too long',
      text: 'A very long scale only allows slow changes, so the wave is smoothed away: underfitting.',
      params: { length: 5, noise: 0.1 },
      claim: (run) => run.scores.trainR2 < 0.5,
    },
    {
      id: 'tuned',
      title: 'Let the evidence choose',
      text: 'Maximising the marginal likelihood picks the length scale and noise from the data alone, with no validation set.',
      params: { tune: true, length: 0.1, noise: 0.5 },
      claim: (run) => {
        const m = gpOf(run.model);
        return (
          !!m &&
          m.gp.hyper.lengthScale > 0.4 &&
          m.gp.hyper.lengthScale < 3 &&
          (run.scores.testR2 ?? 0) > 0.75
        );
      },
    },
    {
      id: 'periodic',
      title: 'A periodic kernel extrapolates',
      text: 'When the kernel knows the signal repeats, the prediction keeps oscillating beyond the data instead of decaying to zero.',
      params: { kernel: 'periodic', length: 1, noise: 0.05 },
      claim: (run) => (run.scores.testR2 ?? 0) > 0.8,
    },
  ],
};
