import { Agglomerative, type Linkage } from '@sim/classic';
import type { Clusterer } from '@sim/core';
import { num, str, type ClusterLabSpec, type ParamSpec } from '@/shared/lab-kit';

const params: ParamSpec[] = [
  {
    key: 'linkage',
    label: 'Linkage (distance between clusters)',
    kind: 'segmented',
    options: [
      { value: 'single', label: 'Single' },
      { value: 'complete', label: 'Complete' },
      { value: 'average', label: 'Average' },
      { value: 'ward', label: 'Ward' },
    ],
  },
  { key: 'k', label: 'Cut into k clusters', kind: 'slider', min: 1, max: 10, step: 1 },
];

export const agglomerativeOf = (m: Clusterer): Agglomerative | null =>
  m instanceof Agglomerative ? m : null;

/** `Agglomerative` has the same shape as a `Clusterer` (fit + labels). */
export const hierarchicalSpec: ClusterLabSpec = {
  id: 'hierarchical',
  create: (p) =>
    new Agglomerative({ k: num(p, 'k'), linkage: str(p, 'linkage') as Linkage }) as Clusterer,
  params,
  defaults: { linkage: 'ward', k: 3 },
  dataset: 'blobs',
  data: { n: 200, noise: 0.1 },
  experiments: [
    {
      id: 'ward-blobs',
      title: 'Ward on round blobs',
      text: 'Ward merges the pair that adds the least variance, which favours compact, similar-sized groups: perfect on blobs.',
      setup: { dataset: 'blobs', noise: 0.1 },
      params: { linkage: 'ward', k: 3 },
      claim: (run) => run.scores.ari > 0.95,
    },
    {
      id: 'single-moons',
      title: 'Single linkage follows crescents',
      text: 'Single linkage joins clusters through their nearest pair of points, so it can follow a thin curved chain: the crescents come out exactly.',
      setup: { dataset: 'moons', noise: 0.08 },
      params: { linkage: 'single', k: 2 },
      claim: (run) => run.scores.ari > 0.95,
    },
    {
      id: 'ward-moons',
      title: 'Ward on the same crescents',
      text: 'Ward wants compact groups, so it cuts each crescent in the middle instead.',
      setup: { dataset: 'moons', noise: 0.08 },
      params: { linkage: 'ward', k: 2 },
      claim: (run) => run.scores.ari < 0.5,
    },
    {
      id: 'single-rings',
      title: 'Single linkage separates rings',
      text: 'The ring and the disc inside it are each a connected chain of near neighbours.',
      setup: { dataset: 'circles', noise: 0.05 },
      params: { linkage: 'single', k: 2 },
      claim: (run) => run.scores.ari > 0.95,
    },
    {
      id: 'chaining',
      title: 'Chaining: single linkage’s weakness',
      text: 'On ordinary blobs a single stray point links groups together or gets peeled off alone: the biggest cluster swallows nearly everything.',
      setup: { dataset: 'blobs', noise: 0.1 },
      params: { linkage: 'single', k: 3 },
      claim: (run) => {
        const sizes = Array.from({ length: 3 }, (_, c) => run.labels.filter((l) => l === c).length);
        return Math.max(...sizes) / run.labels.length > 0.95 && run.scores.ari < 0.1;
      },
    },
  ],
};
