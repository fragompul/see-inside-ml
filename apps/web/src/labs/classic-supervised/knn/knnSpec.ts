import { KnnClassifier, type Metric, type NeighborWeights } from '@sim/classic';
import { num, str, type ClassifierLabSpec, type ParamSpec } from '@/shared/lab-kit';

const params: ParamSpec[] = [
  { key: 'k', label: 'Neighbours k', kind: 'slider', min: 1, max: 60, step: 1 },
  {
    key: 'weights',
    label: 'Vote weights',
    kind: 'segmented',
    options: [
      { value: 'uniform', label: 'Equal' },
      { value: 'distance', label: '1 / distance' },
    ],
  },
  {
    key: 'metric',
    label: 'Distance',
    kind: 'segmented',
    options: [
      { value: 'euclidean', label: 'Euclidean' },
      { value: 'manhattan', label: 'Manhattan' },
      { value: 'chebyshev', label: 'Chebyshev' },
    ],
  },
];

export const knnSpec: ClassifierLabSpec = {
  id: 'knn',
  create: (p) =>
    new KnnClassifier({
      k: num(p, 'k'),
      weights: str(p, 'weights') as NeighborWeights,
      metric: str(p, 'metric') as Metric,
    }),
  params,
  defaults: { k: 5, weights: 'uniform', metric: 'euclidean' },
  dataset: 'moons',
  data: { n: 200, noise: 0.3 },
  experiments: [
    {
      id: 'k1',
      title: 'k = 1: memorise',
      text: 'Each point is its own nearest neighbour, so training accuracy is 100%: the boundary hugs every noisy point.',
      setup: { dataset: 'moons', noise: 0.35 },
      params: { k: 1 },
      claim: (run) => run.scores.trainAccuracy === 1 && (run.scores.testAccuracy ?? 1) < 1,
    },
    {
      id: 'k-moderate',
      title: 'A moderate k',
      text: 'Voting among several neighbours smooths the boundary over the noise and the test accuracy improves.',
      setup: { dataset: 'moons', noise: 0.35 },
      params: { k: 15 },
      claim: (run) => (run.scores.testAccuracy ?? 0) > 0.8,
    },
    {
      id: 'k-huge',
      title: 'k too large',
      text: 'Ask too many neighbours and everything drifts to the majority: the model underfits and the moons blur together.',
      setup: { dataset: 'moons', noise: 0.35 },
      params: { k: 60 },
      claim: (run) => run.scores.trainAccuracy < 0.9,
    },
    {
      id: 'xor',
      title: 'XOR is easy for neighbours',
      text: 'Unlike a straight line, k-NN needs no assumption about the shape of the boundary.',
      setup: { dataset: 'xor' },
      params: { k: 5 },
      claim: (run) => run.scores.trainAccuracy > 0.9,
    },
  ],
};
