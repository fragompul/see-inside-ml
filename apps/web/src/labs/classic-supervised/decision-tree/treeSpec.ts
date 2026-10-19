import { DecisionTree, leafCount, treeDepth, type Criterion } from '@sim/classic';
import type { Classifier } from '@sim/core';
import { num, str, type ClassifierLabSpec, type ParamSpec } from '@/shared/lab-kit';

const params: ParamSpec[] = [
  {
    key: 'criterion',
    label: 'Split quality',
    kind: 'segmented',
    options: [
      { value: 'gini', label: 'Gini' },
      { value: 'entropy', label: 'Entropy' },
    ],
  },
  { key: 'maxDepth', label: 'Maximum depth', kind: 'slider', min: 1, max: 12, step: 1 },
  { key: 'minLeaf', label: 'Minimum points per leaf', kind: 'slider', min: 1, max: 30, step: 1 },
  {
    key: 'ccp',
    label: 'Pruning strength α',
    kind: 'ladder',
    values: [0.0005, 0.001, 0.005, 0.01, 0.02, 0.05],
    format: (v) => String(v),
    hint: 'Cost-complexity pruning: a larger α removes more branches.',
    showIf: (v) => Number(v.pruning) === 1,
  },
  { key: 'pruning', label: 'Prune after growing', kind: 'switch' },
];

export const treeOf = (m: Classifier): DecisionTree | null =>
  m instanceof DecisionTree ? m : null;

export const treeSpec: ClassifierLabSpec = {
  id: 'decision-tree',
  create: (p) =>
    new DecisionTree({
      criterion: str(p, 'criterion') as Criterion,
      maxDepth: num(p, 'maxDepth'),
      minSamplesLeaf: num(p, 'minLeaf'),
      ...(p.pruning ? { ccpAlpha: num(p, 'ccp') } : {}),
    }),
  params,
  defaults: { criterion: 'gini', maxDepth: 3, minLeaf: 1, ccp: 0.01, pruning: false },
  dataset: 'moons',
  data: { n: 200, noise: 0.5 },
  experiments: [
    {
      id: 'stump',
      title: 'A single split',
      text: 'One question gives one straight, axis-aligned boundary: too simple for crescents.',
      params: { maxDepth: 1 },
      claim: (run) => run.scores.trainAccuracy < 0.9,
    },
    {
      id: 'memorise',
      title: 'No limits: memorise',
      text: 'Left alone the tree keeps splitting until every training point is classified: training accuracy hits 100%, but the boundary chases noise.',
      params: { maxDepth: 12 },
      claim: (run) => run.scores.trainAccuracy === 1 && (run.scores.testAccuracy ?? 1) < 1,
    },
    {
      id: 'leaf-size',
      title: 'Demand 15 points per leaf',
      text: 'A minimum leaf size forbids tiny, noise-fitting regions: a much smaller tree for a little lower training accuracy.',
      params: { maxDepth: 12, minLeaf: 15 },
      claim: (run) => {
        const t = run.model instanceof DecisionTree ? run.model : null;
        return !!t && leafCount(t.root) <= 12 && run.scores.trainAccuracy > 0.8;
      },
    },
    {
      id: 'pruned',
      title: 'Grow, then prune',
      text: 'Cost-complexity pruning cuts the branches that buy the least accuracy per leaf.',
      params: { maxDepth: 12, pruning: true, ccp: 0.01 },
      claim: (run) => {
        const t = run.model instanceof DecisionTree ? run.model : null;
        return !!t && leafCount(t.root) <= 10 && run.scores.trainAccuracy > 0.9;
      },
    },
    {
      id: 'xor',
      title: 'XOR needs depth',
      text: 'No single question separates XOR; a deeper tree carves out the four quadrants.',
      setup: { dataset: 'xor' },
      params: { maxDepth: 6 },
      claim: (run) => run.scores.trainAccuracy > 0.9,
    },
    {
      id: 'depth-limits',
      title: 'Depth is bounded',
      text: 'The maximum depth caps how many questions in a row the tree may ask.',
      params: { maxDepth: 2 },
      claim: (run) => {
        const t = run.model instanceof DecisionTree ? run.model : null;
        return !!t && treeDepth(t.root) <= 2;
      },
    },
  ],
};
