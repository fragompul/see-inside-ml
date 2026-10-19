import { GaussianMixture, type CovarianceType } from '@sim/classic';
import type { Clusterer } from '@sim/core';
import { num, str, type ClusterLabSpec, type ParamSpec } from '@/shared/lab-kit';

const params: ParamSpec[] = [
  { key: 'k', label: 'Components k', kind: 'slider', min: 1, max: 8, step: 1 },
  {
    key: 'covariance',
    label: 'Covariance shape',
    kind: 'segmented',
    options: [
      { value: 'full', label: 'Full' },
      { value: 'tied', label: 'Tied' },
      { value: 'diag', label: 'Diagonal' },
      { value: 'spherical', label: 'Spherical' },
    ],
    hint: 'Full: any ellipse. Tied: the same ellipse for all. Diagonal: axis-aligned. Spherical: circles.',
  },
  { key: 'seed', label: 'Seed of the start', kind: 'slider', min: 1, max: 20, step: 1 },
];

export const gmmOf = (m: Clusterer): GaussianMixture | null =>
  m instanceof GaussianMixture ? m : null;

export const gmmSpec: ClusterLabSpec = {
  id: 'gmm',
  create: (p) =>
    new GaussianMixture({
      k: num(p, 'k'),
      covariance: str(p, 'covariance') as CovarianceType,
      seed: num(p, 'seed'),
      maxIterations: 200,
    }),
  params,
  defaults: { k: 3, covariance: 'full', seed: 3 },
  dataset: 'blobs',
  data: { n: 240, noise: 0.1 },
  experiments: [
    {
      id: 'blobs',
      title: 'Three Gaussians',
      text: 'When the data really are Gaussian clouds, EM recovers them: centres, spreads and mixing proportions.',
      setup: { dataset: 'blobs', noise: 0.1 },
      params: { k: 3, covariance: 'full' },
      claim: (run) => run.scores.ari > 0.95,
    },
    {
      id: 'tilted-full',
      title: 'Tilted clouds: ellipses help',
      text: 'The two groups are stretched along a diagonal. A full covariance can tilt its ellipse to match them.',
      setup: { dataset: 'tilted', noise: 0.1 },
      params: { k: 2, covariance: 'full' },
      claim: (run) => run.scores.ari > 0.75,
    },
    {
      id: 'tilted-spherical',
      title: 'Tilted clouds, round components',
      text: 'Circles cannot tilt: they cut the diagonal clouds the wrong way and the clusters no longer match the truth.',
      setup: { dataset: 'tilted', noise: 0.1 },
      params: { k: 2, covariance: 'spherical' },
      claim: (run) => run.scores.ari < 0.3,
    },
    {
      id: 'soft',
      title: 'Soft assignments',
      text: 'Where groups overlap, a point belongs partly to each. The ringed points are those the model is less than 90% sure about.',
      setup: { dataset: 'overlap', noise: 0.15 },
      params: { k: 2, covariance: 'full' },
      claim: (run) => {
        const m = run.model instanceof GaussianMixture ? run.model : null;
        if (!m) return false;
        let unsure = 0;
        for (let i = 0; i < m.responsibilities.rows; i++) {
          let best = 0;
          for (let c = 0; c < m.responsibilities.cols; c++)
            best = Math.max(best, m.responsibilities.get(i, c));
          if (best < 0.9) unsure++;
        }
        return unsure / m.responsibilities.rows > 0.2;
      },
    },
    {
      id: 'moons',
      title: 'Crescents are not Gaussian',
      text: 'A mixture of ellipses can only approximate a curved shape by chopping it into pieces: the match with the truth stays poor.',
      setup: { dataset: 'moons', noise: 0.08 },
      params: { k: 2, covariance: 'full' },
      claim: (run) => run.scores.ari < 0.55,
    },
  ],
};
