import { KMeans, type KMeansInit } from '@sim/classic';
import type { Clusterer } from '@sim/core';
import { num, str, type ClusterLabSpec, type ParamSpec } from '@/shared/lab-kit';

const params: ParamSpec[] = [
  { key: 'k', label: 'Number of clusters k', kind: 'slider', min: 1, max: 10, step: 1 },
  {
    key: 'init',
    label: 'Starting centres',
    kind: 'segmented',
    options: [
      { value: 'random', label: 'Random points' },
      { value: 'kmeans++', label: 'k-means++' },
    ],
    hint: 'k-means++ spreads the starting centres apart.',
  },
  {
    key: 'restarts',
    label: 'Restarts',
    kind: 'slider',
    min: 1,
    max: 15,
    step: 1,
    hint: 'Run from several starts and keep the best (lowest inertia).',
  },
  { key: 'seed', label: 'Seed of the starting centres', kind: 'slider', min: 1, max: 20, step: 1 },
];

export const kmeansOf = (m: Clusterer): KMeans | null => (m instanceof KMeans ? m : null);

export const kmeansSpec: ClusterLabSpec = {
  id: 'kmeans',
  create: (p) =>
    new KMeans({
      k: num(p, 'k'),
      init: str(p, 'init') as KMeansInit,
      restarts: num(p, 'restarts'),
      seed: num(p, 'seed'),
    }),
  params,
  defaults: { k: 3, init: 'kmeans++', restarts: 5, seed: 3 },
  dataset: 'blobs',
  data: { n: 240, noise: 0.1 },
  experiments: [
    {
      id: 'blobs',
      title: 'Round blobs: k-means shines',
      text: 'Three compact, similarly sized clouds are exactly what k-means assumes: it recovers them almost perfectly.',
      setup: { dataset: 'blobs', noise: 0.1 },
      params: { k: 3 },
      claim: (run) => run.scores.ari > 0.95,
    },
    {
      id: 'too-few',
      title: 'k too small',
      text: 'With k = 2 two real groups are forced together: inertia is far higher and the match with the truth drops.',
      setup: { dataset: 'blobs', noise: 0.1 },
      params: { k: 2 },
      claim: (run) => run.scores.ari < 0.7,
    },
    {
      id: 'too-many',
      title: 'k too large',
      text: 'With k = 6 real groups are cut into pieces. Inertia always falls as k grows, so it cannot tell you when to stop.',
      setup: { dataset: 'blobs', noise: 0.1 },
      params: { k: 6 },
      claim: (run) => run.scores.ari < 0.7,
    },
    {
      id: 'moons',
      title: 'Crescents: wrong shape',
      text: 'k-means cuts space into convex cells around centres, so it slices the two crescents across instead of following them.',
      setup: { dataset: 'moons', noise: 0.1 },
      params: { k: 2 },
      claim: (run) => run.scores.ari < 0.5,
    },
    {
      id: 'rings',
      title: 'A ring around a core',
      text: 'The inner disc and outer ring share a centre, so no assignment by distance to centres can separate them.',
      setup: { dataset: 'circles', noise: 0.05 },
      params: { k: 2 },
      claim: (run) => run.scores.ari < 0.1,
    },
    {
      id: 'unlucky',
      title: 'An unlucky start',
      text: 'On four tight corners even a k-means++ start can leave two centres in one group and none in another: the run settles in a poor local minimum.',
      setup: { dataset: 'xor', noise: 0 },
      params: { k: 4, init: 'kmeans++', restarts: 1, seed: 5 },
      claim: (run) => {
        const m = kmeansOf(run.model);
        return !!m && m.inertia > 190;
      },
    },
    {
      id: 'restarts',
      title: 'Restarts fix it',
      text: 'Running from ten different starts and keeping the best escapes the poor minimum.',
      setup: { dataset: 'xor', noise: 0 },
      params: { k: 4, init: 'kmeans++', restarts: 10, seed: 5 },
      claim: (run) => {
        const m = kmeansOf(run.model);
        return !!m && m.inertia < 165;
      },
    },
  ],
};
