import type { ParamSpec } from '@/shared/lab-kit';
import { DEPTHS, runGnnLab, type GnnResult, type GnnScenario } from './gnnCore';

export const gnnParams: ParamSpec[] = [
  {
    key: 'pOut',
    label: 'Edges between communities',
    kind: 'ladder',
    values: [0.01, 0.03],
    format: (v) => `${v * 100}% of pairs`,
    hint: 'Inside a community 15% of pairs are connected.',
  },
  {
    key: 'signal',
    label: 'How much the features say',
    kind: 'ladder',
    values: [0.3, 0.7],
    format: (v) => (v < 0.5 ? 'little' : 'some'),
  },
  {
    key: 'labelled',
    label: 'Labelled nodes per community',
    kind: 'ladder',
    values: [2, 4, 8],
    format: (v) => String(v),
  },
  {
    key: 'layers',
    label: 'Layers of the network shown',
    kind: 'ladder',
    values: DEPTHS,
    format: (v) => String(v),
  },
  { key: 'useGraph', label: 'Use the graph', kind: 'switch' },
];

export interface GnnExperiment {
  id: string;
  title: string;
  text: string;
  scenario: Partial<GnnScenario>;
  claim: (run: GnnResult) => boolean;
}

const at = (values: number[], layers: number) => values[DEPTHS.indexOf(layers)] as number;
const sibling = (r: GnnResult, change: Partial<GnnScenario>) =>
  runGnnLab({ ...r.scenario, ...change });

export const gnnExperiments: GnnExperiment[] = [
  {
    id: 'graph-helps',
    title: 'The graph carries the labels',
    text: 'With four labelled nodes in each of three communities and features that only weakly hint at the community, a network that ignores the graph classifies about two thirds of the other nodes correctly. The same network averaging over neighbours gets 98% right with two layers.',
    scenario: { pOut: 0.01, signal: 0.7, labelled: 4 },
    claim: (r) => at(r.graph, 2) > at(r.plain, 2) + 0.25 && at(r.graph, 2) > 0.95,
  },
  {
    id: 'plain-flat',
    title: 'Depth does nothing for a network that ignores the graph',
    text: 'Without the graph, accuracy stays between 64% and 67% whether the network has one layer or eight: more layers cannot create information that the features do not contain.',
    scenario: { pOut: 0.01, signal: 0.7, useGraph: false },
    claim: (r) => Math.max(...r.plain) - Math.min(...r.plain) < 0.06 && Math.max(...r.plain) < 0.75,
  },
  {
    id: 'smoothing',
    title: 'Too many layers blur the communities together',
    text: 'When 3% of the pairs of nodes in different communities are linked, every extra layer mixes a node with more of the other communities. Accuracy peaks at two layers (92%) and falls to 58% at eight, while the similarity of representations from different communities climbs from 0.50 to 0.80.',
    scenario: { pOut: 0.03, signal: 0.7 },
    claim: (r) =>
      at(r.graph, 2) - at(r.graph, 8) > 0.25 && at(r.otherClass, 8) > at(r.otherClass, 2) + 0.2,
  },
  {
    id: 'weak-features',
    title: 'Weak features want more reach',
    text: 'When the features say little, one layer, which only sees direct neighbours, reaches 72%; two reach 89% and four reach 94%. Deeper networks gather more evidence before they decide, up to the point where smoothing takes over.',
    scenario: { pOut: 0.01, signal: 0.3 },
    claim: (r) => at(r.graph, 4) > at(r.graph, 1) + 0.15 && at(r.graph, 4) > 0.9,
  },
  {
    id: 'bridges',
    title: 'Edges between communities cost accuracy',
    text: 'Tripling the number of edges between communities lowers the two-layer accuracy from 98% to 92%: each such edge hands a node a misleading neighbour.',
    scenario: { pOut: 0.03, signal: 0.7 },
    claim: (r) => at(sibling(r, { pOut: 0.01 }).graph, 2) - at(r.graph, 2) > 0.03,
  },
];
