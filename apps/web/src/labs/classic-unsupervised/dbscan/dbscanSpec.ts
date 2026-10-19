import { Dbscan } from '@sim/classic';
import type { Clusterer } from '@sim/core';
import { num, type ClusterLabSpec, type ParamSpec } from '@/shared/lab-kit';

export const EPS_VALUES = [0.05, 0.1, 0.15, 0.2, 0.3, 0.45, 0.7, 1, 1.5] as const;

const params: ParamSpec[] = [
  {
    key: 'eps',
    label: 'Neighbourhood radius ε',
    kind: 'ladder',
    values: EPS_VALUES,
    format: (v) => String(v),
    hint: 'How close two points must be to count as neighbours.',
  },
  {
    key: 'minSamples',
    label: 'Neighbours needed (min samples)',
    kind: 'slider',
    min: 2,
    max: 15,
    step: 1,
    hint: 'A point with at least this many neighbours (itself included) is a core point.',
  },
];

export const dbscanOf = (m: Clusterer): Dbscan | null => (m instanceof Dbscan ? m : null);

export const dbscanSpec: ClusterLabSpec = {
  id: 'dbscan',
  create: (p) => new Dbscan({ eps: num(p, 'eps'), minSamples: num(p, 'minSamples') }),
  params,
  defaults: { eps: 0.45, minSamples: 5 },
  dataset: 'moons',
  data: { n: 240, noise: 0.08 },
  experiments: [
    {
      id: 'moons',
      title: 'Crescents, no k needed',
      text: 'DBSCAN grows clusters through chains of dense neighbourhoods, so it follows the crescents and finds that there are two without being told.',
      setup: { dataset: 'moons', noise: 0.08 },
      params: { eps: 0.45, minSamples: 5 },
      claim: (run) => run.scores.clusters === 2 && run.scores.ari > 0.95,
    },
    {
      id: 'rings',
      title: 'A ring around a disc',
      text: 'The same setting separates a ring from the disc inside it, which defeats k-means.',
      setup: { dataset: 'circles', noise: 0.05 },
      params: { eps: 0.45, minSamples: 5 },
      claim: (run) => run.scores.clusters === 2 && run.scores.ari > 0.95,
    },
    {
      id: 'tiny-eps',
      title: 'ε too small',
      text: 'With a tiny radius almost nothing is dense: most points are labelled noise and the rest shatter into fragments.',
      setup: { dataset: 'moons', noise: 0.08 },
      params: { eps: 0.1, minSamples: 5 },
      claim: (run) => run.scores.noise > 100 && run.scores.clusters > 5,
    },
    {
      id: 'huge-eps',
      title: 'ε too large',
      text: 'With a huge radius every point is a neighbour of the next: everything merges into one cluster.',
      setup: { dataset: 'moons', noise: 0.08 },
      params: { eps: 1.5, minSamples: 5 },
      claim: (run) => run.scores.clusters === 1,
    },
    {
      id: 'blobs-noise',
      title: 'Noise is a feature',
      text: 'DBSCAN does not force every point into a cluster: the sparse fringe of each blob is flagged as noise (grey).',
      setup: { dataset: 'blobs', noise: 0.1 },
      params: { eps: 0.45, minSamples: 5 },
      claim: (run) => run.scores.noise > 5 && run.scores.clusters >= 3,
    },
    {
      id: 'spirals',
      title: 'One ε cannot fit varying density',
      text: 'On the spirals the points thin out towards the arms’ ends, so no single radius works: it either fragments the outside or merges the arms.',
      setup: { dataset: 'spirals', noise: 0.05 },
      params: { eps: 0.7, minSamples: 5 },
      claim: (run) => run.scores.ari < 0.2,
    },
  ],
};
