import { RandomForestClassifier } from '@sim/classic';
import type { Classifier } from '@sim/core';
import { flag, num, type ClassifierLabSpec, type ParamSpec } from '@/shared/lab-kit';

const params: ParamSpec[] = [
  { key: 'trees', label: 'Number of trees', kind: 'slider', min: 1, max: 200, step: 1 },
  { key: 'maxDepth', label: 'Maximum depth', kind: 'slider', min: 1, max: 14, step: 1 },
  {
    key: 'maxFeatures',
    label: 'Features tried per split',
    kind: 'slider',
    min: 1,
    max: 2,
    step: 1,
    hint: 'With two features: 1 = pick one at random per split (decorrelates trees), 2 = best of both.',
  },
  { key: 'bootstrap', label: 'Bootstrap rows for each tree', kind: 'switch' },
];

export const forestOf = (m: Classifier): RandomForestClassifier | null =>
  m instanceof RandomForestClassifier ? m : null;

export const forestSpec: ClassifierLabSpec = {
  id: 'random-forest',
  create: (p) =>
    new RandomForestClassifier({
      trees: num(p, 'trees'),
      maxDepth: num(p, 'maxDepth'),
      maxFeatures: num(p, 'maxFeatures'),
      bootstrap: flag(p, 'bootstrap'),
      seed: 7,
    }),
  params,
  defaults: { trees: 100, maxDepth: 12, maxFeatures: 1, bootstrap: true },
  dataset: 'moons',
  data: { n: 240, noise: 0.5 },
  experiments: [
    {
      id: 'one-tree',
      title: 'One deep tree',
      text: 'A single unrestricted tree memorises the training points: jagged boundary, perfect training accuracy.',
      params: { trees: 1, bootstrap: false, maxFeatures: 2 },
      claim: (run) => run.scores.trainAccuracy === 1 && (run.scores.testAccuracy ?? 1) < 1,
    },
    {
      id: 'forest',
      title: 'Average 100 trees',
      text: 'Each tree overfits differently; averaging their votes cancels the noise and the boundary smooths out.',
      claim: (run) => (run.scores.testAccuracy ?? 0) >= 0.85,
    },
    {
      id: 'oob',
      title: 'Free error estimate',
      text: 'Every tree leaves out about a third of the rows. Judging each row only by the trees that never saw it gives an honest accuracy without a test set.',
      claim: (run) => {
        const f = run.model instanceof RandomForestClassifier ? run.model : null;
        return (
          !!f &&
          f.oobAccuracy !== null &&
          Math.abs(f.oobAccuracy - (run.scores.testAccuracy ?? 0)) < 0.1
        );
      },
    },
    {
      id: 'stumps',
      title: 'Depth 1 trees: too weak',
      text: 'A forest of stumps can only make axis-aligned cuts; averaging helps a little but cannot build the crescents.',
      params: { maxDepth: 1 },
      claim: (run) => run.scores.trainAccuracy < 0.9,
    },
    {
      id: 'xor',
      title: 'XOR',
      text: 'Many random trees together carve out the four quadrants.',
      setup: { dataset: 'xor' },
      claim: (run) => run.scores.trainAccuracy > 0.9,
    },
  ],
};
