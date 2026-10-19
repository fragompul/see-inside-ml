import { SpectralClustering, type Affinity } from '@sim/classic';
import type { Clusterer } from '@sim/core';
import { num, str, type ClusterLabSpec, type ParamSpec } from '@/shared/lab-kit';

const params: ParamSpec[] = [
  { key: 'k', label: 'Clusters k', kind: 'slider', min: 2, max: 8, step: 1 },
  {
    key: 'affinity',
    label: 'Similarity graph',
    kind: 'segmented',
    options: [
      { value: 'knn', label: 'Nearest neighbours' },
      { value: 'rbf', label: 'Gaussian (RBF)' },
    ],
  },
  {
    key: 'neighbors',
    label: 'Neighbours per point',
    kind: 'slider',
    min: 3,
    max: 30,
    step: 1,
    showIf: (v) => v.affinity === 'knn',
  },
  {
    key: 'gamma',
    label: 'RBF sharpness γ',
    kind: 'ladder',
    values: [0.5, 2, 8, 30, 100, 300],
    format: (v) => String(v),
    hint: 'Large γ: only very close points are similar.',
    showIf: (v) => v.affinity === 'rbf',
  },
];

export const spectralOf = (m: Clusterer): SpectralClustering | null =>
  m instanceof SpectralClustering ? m : null;

export const spectralSpec: ClusterLabSpec = {
  id: 'spectral',
  create: (p) => {
    const affinity: Affinity =
      str(p, 'affinity') === 'knn'
        ? { kind: 'knn', neighbors: num(p, 'neighbors') }
        : { kind: 'rbf', gamma: num(p, 'gamma') };
    return new SpectralClustering({ k: num(p, 'k'), affinity, seed: 3 });
  },
  params,
  defaults: { k: 2, affinity: 'knn', neighbors: 10, gamma: 8 },
  dataset: 'moons',
  data: { n: 200, noise: 0.08 },
  experiments: [
    {
      id: 'moons',
      title: 'Crescents, solved',
      text: 'Spectral clustering looks at the connectivity graph, not at distance to a centre: two chains of neighbours are two clusters.',
      setup: { dataset: 'moons', noise: 0.08 },
      params: { k: 2, affinity: 'knn', neighbors: 10 },
      claim: (run) => run.scores.ari > 0.95,
    },
    {
      id: 'rings',
      title: 'A ring around a disc',
      text: 'The inner disc and the outer ring are each internally connected and only weakly connected to each other.',
      setup: { dataset: 'circles', noise: 0.05 },
      params: { k: 2, affinity: 'knn', neighbors: 10 },
      claim: (run) => run.scores.ari > 0.95,
    },
    {
      id: 'blobs',
      title: 'It also handles blobs',
      text: 'On ordinary clusters it behaves like a good k-means, at a higher cost.',
      setup: { dataset: 'blobs', noise: 0.1 },
      params: { k: 3, affinity: 'knn', neighbors: 10 },
      claim: (run) => run.scores.ari > 0.9,
    },
    {
      id: 'sharp',
      title: 'A too sharp kernel',
      text: 'With a very large γ almost every similarity is zero, the graph falls into many little pieces and the eigenvectors stop describing the shape.',
      setup: { dataset: 'circles', noise: 0.05 },
      params: { k: 2, affinity: 'rbf', gamma: 300 },
      claim: (run) => run.scores.ari < 0.6,
    },
    {
      id: 'spirals',
      title: 'Spirals: a graph that leaks',
      text: 'A ten-neighbour graph hops across the narrow gap between the two spiral arms, joining them: the spectrum no longer separates them.',
      setup: { dataset: 'spirals', noise: 0.05 },
      params: { k: 2, affinity: 'knn', neighbors: 10 },
      claim: (run) => run.scores.ari < 0.3,
    },
  ],
};
