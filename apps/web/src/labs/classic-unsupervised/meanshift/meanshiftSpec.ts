import { MeanShift } from '@sim/classic';
import type { Clusterer } from '@sim/core';
import { num, type ClusterLabSpec, type ParamSpec } from '@/shared/lab-kit';

export const BANDWIDTHS = [0.2, 0.4, 0.7, 1, 1.5, 2.5, 4] as const;

const params: ParamSpec[] = [
  {
    key: 'bandwidth',
    label: 'Bandwidth (window radius)',
    kind: 'ladder',
    values: BANDWIDTHS,
    format: (v) => String(v),
    hint: 'How far a point looks when it averages its neighbours.',
  },
];

export const meanShiftOf = (m: Clusterer): MeanShift | null => (m instanceof MeanShift ? m : null);

export const meanshiftSpec: ClusterLabSpec = {
  id: 'meanshift',
  create: (p) => new MeanShift({ bandwidth: num(p, 'bandwidth') }),
  params,
  defaults: { bandwidth: 1.5 },
  dataset: 'blobs',
  data: { n: 240, noise: 0.1 },
  experiments: [
    {
      id: 'blobs',
      title: 'Peaks of the density',
      text: 'Every point climbs the density estimate until it reaches a peak; points that end at the same peak form a cluster. With a suitable window it finds the three blobs without being told how many.',
      setup: { dataset: 'blobs', noise: 0.1 },
      params: { bandwidth: 1.5 },
      claim: (run) => run.scores.clusters === 3 && run.scores.ari > 0.95,
    },
    {
      id: 'small',
      title: 'Window too small',
      text: 'A tiny window sees only a few neighbours, so the density estimate is bumpy and every little bump becomes a cluster.',
      setup: { dataset: 'blobs', noise: 0.1 },
      params: { bandwidth: 0.4 },
      claim: (run) => run.scores.clusters > 15,
    },
    {
      id: 'large',
      title: 'Window too large',
      text: 'A window as wide as the whole picture averages everything: one peak, one cluster.',
      setup: { dataset: 'blobs', noise: 0.1 },
      params: { bandwidth: 2.5 },
      claim: (run) => run.scores.clusters === 1,
    },
    {
      id: 'moons',
      title: 'Crescents: peaks are not shapes',
      text: 'Mean shift looks for peaks of the density, and a crescent is a ridge, not a peak: it breaks each crescent into blobs.',
      setup: { dataset: 'moons', noise: 0.08 },
      params: { bandwidth: 1.5 },
      claim: (run) => run.scores.ari < 0.5,
    },
  ],
};
