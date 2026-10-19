import {
  KMeans,
  PCA,
  classicalMds,
  isomap,
  lle,
  pairwiseDistances,
  randomEmbedding,
  smacofSteps,
  tsneSteps,
  umapSteps,
} from '@sim/classic';
import { adjustedRandIndex, type Matrix } from '@sim/core';
import {
  num,
  oneShot,
  scoreEmbedding,
  str,
  type EmbedModel,
  type ParamSpec,
  type ProjectorExperiment,
  type ProjectorLabSpec,
} from '@/shared/lab-kit';

export type ManifoldMethod = 'mds' | 'smacof' | 'isomap' | 'lle' | 'tsne' | 'umap';

export const METHODS: { value: ManifoldMethod; label: string }[] = [
  { value: 'mds', label: 'Classical MDS' },
  { value: 'smacof', label: 'Metric MDS (SMACOF)' },
  { value: 'isomap', label: 'Isomap' },
  { value: 'lle', label: 'Locally linear embedding' },
  { value: 'tsne', label: 't-SNE' },
  { value: 'umap', label: 'UMAP' },
];

const params: ParamSpec[] = [
  { key: 'method', label: 'Method', kind: 'select', options: METHODS },
  {
    key: 'neighbors',
    label: 'Neighbours k',
    kind: 'slider',
    min: 3,
    max: 30,
    step: 1,
    hint: 'How many neighbours define the local structure.',
    showIf: (v) => v.method === 'isomap' || v.method === 'lle' || v.method === 'umap',
  },
  {
    key: 'perplexity',
    label: 'Perplexity',
    kind: 'ladder',
    values: [2, 5, 10, 20, 30, 50],
    format: (v) => String(v),
    hint: 'The effective number of neighbours each point pays attention to.',
    showIf: (v) => v.method === 'tsne',
  },
  {
    key: 'minDist',
    label: 'Minimum distance',
    kind: 'ladder',
    values: [0, 0.1, 0.5, 1],
    format: (v) => String(v),
    hint: 'How tightly UMAP may pack points together.',
    showIf: (v) => v.method === 'umap',
  },
  {
    key: 'seed',
    label: 'Seed',
    kind: 'slider',
    min: 1,
    max: 20,
    step: 1,
    showIf: (v) => ['smacof', 'tsne', 'umap'].includes(String(v.method)),
  },
];

const progressOf = (i: number, total: number) => Math.min(1, i / total);

export function makeEmbedder(p: Record<string, number | string | boolean>): EmbedModel {
  const method = str(p, 'method') as ManifoldMethod;
  const seed = num(p, 'seed');
  switch (method) {
    case 'mds':
      return oneShot((x) => {
        const r = classicalMds(pairwiseDistances(x), x.rows, 2);
        const total = r.eigenvalues.reduce((a, b) => a + Math.max(b, 0), 0);
        return {
          embedding: r.embedding,
          info: `two axes keep ${((((r.eigenvalues[0] as number) + (r.eigenvalues[1] as number)) / total) * 100).toFixed(1)}% of the squared distances`,
        };
      });
    case 'isomap':
      return oneShot((x) => ({
        embedding: isomap(x, num(p, 'neighbors'), 2).embedding,
        info: `geodesic distances over a ${num(p, 'neighbors')}-neighbour graph`,
      }));
    case 'lle':
      return oneShot((x) => {
        const r = lle(x, num(p, 'neighbors'), 2);
        return {
          embedding: r.embedding,
          info: `reconstruction error ${r.reconstructionError.toExponential(2)}`,
        };
      });
    case 'smacof':
      return {
        *run(x) {
          const d = pairwiseDistances(x);
          for (const s of smacofSteps(d, x.rows, randomEmbedding(x.rows, 2, seed, 1), {
            maxIterations: 200,
            tolerance: 1e-4,
          }))
            yield {
              embedding: s.embedding,
              info: `iteration ${s.iteration}, stress ${s.stress.toFixed(1)}`,
              progress: progressOf(s.iteration, 200),
            };
        },
      };
    case 'tsne':
      return {
        *run(x) {
          const iterations = 500;
          for (const s of tsneSteps(x, {
            perplexity: Math.min(num(p, 'perplexity'), (x.rows - 1) / 3),
            iterations,
            init: 'random',
            seed,
          }))
            yield {
              embedding: s.embedding,
              info: `iteration ${s.iteration}, KL ${s.kl.toFixed(3)}${s.exaggerating ? ' (early exaggeration)' : ''}`,
              progress: progressOf(s.iteration, iterations),
            };
        },
      };
    default:
      return {
        *run(x) {
          const epochs = 200;
          for (const s of umapSteps(
            x,
            { neighbors: num(p, 'neighbors'), minDist: num(p, 'minDist'), epochs, seed },
            5,
          ))
            yield {
              embedding: s.embedding,
              info: `epoch ${s.epoch} of ${epochs}`,
              progress: progressOf(s.epoch, epochs),
            };
        },
      };
  }
}

const clusterAri = (e: Matrix, y: Float64Array, k: number) =>
  adjustedRandIndex(y, new KMeans({ k, restarts: 5, seed: 2 }).fit(e).labels);

export const manifoldSpec: ProjectorLabSpec = {
  id: 'manifold',
  create: makeEmbedder,
  params,
  defaults: { method: 'tsne', neighbors: 10, perplexity: 20, minDist: 0.1, seed: 3 },
  dataset: 'swiss',
  datasets: ['swiss', 'scurve', 'sphere', 'clusters', 'iris', 'wine', 'digits'],
  data: { n: 150, noise: 0.1 },
  experiments: [
    {
      id: 'scurve-isomap',
      title: 'Isomap unrolls the S-curve',
      text: 'Isomap measures distance along the surface (hopping between neighbours) instead of through the air, so the bent sheet unfolds flat and neighbours stay neighbours.',
      setup: { dataset: 'scurve', noise: 0.1 },
      params: { method: 'isomap', neighbors: 10 },
      claim: (run) => run.scores.trustworthiness > 0.98 && run.scores.preservation > 0.7,
    },
    {
      id: 'roll-shortcut',
      title: 'Isomap on the roll: a short-circuit',
      text: 'On the tightly rolled sheet the neighbour graph hops between neighbouring layers, so the “surface distances” are wrong and Isomap does worse than a plain projection.',
      setup: { dataset: 'swiss', noise: 0.1 },
      params: { method: 'isomap', neighbors: 10 },
      claim: (run) => {
        const pca = new PCA({ components: 2 }).fit(run.data.x).transform(run.data.x);
        return run.scores.trustworthiness < scoreEmbedding(run.data.x, pca).trustworthiness;
      },
    },
    {
      id: 'roll-tsne',
      title: 't-SNE unrolls the roll',
      text: 't-SNE only cares that each point keeps its own neighbours, so it can tear the roll open and lay it flat: nearly every neighbour is true.',
      setup: { dataset: 'swiss', noise: 0.1 },
      params: { method: 'tsne', perplexity: 20 },
      claim: (run) => run.scores.trustworthiness > 0.98 && run.scores.preservation > 0.8,
    },
    {
      id: 'ten-d',
      title: 'Ten dimensions, five groups',
      text: 'Neighbourhoods in ten dimensions survive the trip to two: t-SNE keeps both the groups and the local structure inside them.',
      setup: { dataset: 'clusters', noise: 0.1 },
      params: { method: 'tsne', perplexity: 20 },
      claim: (run) =>
        run.scores.trustworthiness > 0.98 && clusterAri(run.embedding, run.data.y, 5) > 0.95,
    },
    {
      id: 'digits-umap',
      title: 'Digits with UMAP',
      text: 'Handwritten digits live in 256 dimensions; UMAP draws them as islands of the same digit, far better than PCA’s smear.',
      setup: { dataset: 'digits', n: 150 },
      params: { method: 'umap', neighbors: 12 },
      claim: (run) => {
        const pca = new PCA({ components: 2 }).fit(run.data.x).transform(run.data.x);
        return (
          run.scores.trustworthiness > 0.95 &&
          clusterAri(run.embedding, run.data.y, 10) > clusterAri(pca, run.data.y, 10)
        );
      },
    },
    {
      id: 'low-perplexity',
      title: 'Perplexity too low',
      text: 'With a perplexity of 2 every point listens to just one or two neighbours: the picture fragments into small clumps and many of the neighbours it shows are false.',
      setup: { dataset: 'iris' },
      params: { method: 'tsne', perplexity: 2 },
      claim: (run) => run.scores.trustworthiness < 0.97,
    },
  ] satisfies ProjectorExperiment[],
};
