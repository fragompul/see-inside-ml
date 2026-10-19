import type { ParamSpec } from '@/shared/lab-kit';
import { WEBS, type PageRankRun, type PageRankScenario, type WebId } from './pagerankCore';

export const pagerankParams: ParamSpec[] = [
  {
    key: 'web',
    label: 'The web',
    kind: 'segmented',
    options: (Object.keys(WEBS) as WebId[]).map((id) => ({ value: id, label: WEBS[id].label })),
  },
  {
    key: 'damping',
    label: 'Damping (chance of following a link)',
    kind: 'slider',
    min: 0,
    max: 0.99,
    step: 0.01,
    format: (v) => v.toFixed(2),
    hint: 'The rest of the time the surfer gets bored and jumps to a random page.',
  },
  {
    key: 'iterations',
    label: 'Power-iteration steps shown',
    kind: 'slider',
    min: 0,
    max: 40,
    step: 1,
  },
  {
    key: 'surfers',
    label: 'Random-surfer steps',
    kind: 'ladder',
    values: [100, 1000, 5000, 20000, 200000],
    format: (v) => String(v),
  },
];

export interface PageRankExperiment {
  id: string;
  title: string;
  text: string;
  scenario: Partial<PageRankScenario>;
  claim: (run: PageRankRun) => boolean;
}

const argmax = (v: readonly number[]): number => v.indexOf(Math.max(...v));

export const pagerankExperiments: PageRankExperiment[] = [
  {
    id: 'hub',
    title: 'Rank flows to the hub',
    text: 'Almost every page links to C, and C passes its rank on to A. After many steps C and A hold two thirds of the rank, while the pages nobody links to keep only a few percent.',
    scenario: {},
    claim: (r) => argmax(r.exact) === 2 && r.exact[0]! + r.exact[2]! > 0.65 && r.exact[3]! < 0.03,
  },
  {
    id: 'no-links',
    title: 'Ignoring the links',
    text: 'With damping 0 the surfer always teleports: every page is equally likely and the links make no difference at all.',
    scenario: { damping: 0 },
    claim: (r) => Math.max(...r.exact) - Math.min(...r.exact) < 1e-9,
  },
  {
    id: 'gentle',
    title: 'A restless surfer',
    text: 'At damping 0.5 the surfer teleports half the time: the hub is less dominant, and the ranks settle in about five steps.',
    scenario: { damping: 0.5 },
    claim: (r) => Math.max(...r.exact) < 0.3 && r.stepsToConverge <= 6,
  },
  {
    id: 'trap',
    title: 'A spider trap',
    text: 'Four pages that only link to each other soak up rank: nothing leaves the group except by teleporting. Half the pages end up with almost three quarters of the rank, and the convergence is as slow as the damping factor allows.',
    scenario: { web: 'trap' },
    claim: (r) =>
      r.exact.slice(4).reduce((a, b) => a + b, 0) > 0.7 &&
      r.stepsToConverge > 25 &&
      Math.abs(r.rate - 0.85) < 0.01,
  },
  {
    id: 'trap-teleport',
    title: 'Teleporting defuses the trap',
    text: 'Raising the teleport probability (damping 0.5) lets rank escape from the trap: the farm keeps less than half.',
    scenario: { web: 'trap', damping: 0.5 },
    claim: (r) => r.exact.slice(4).reduce((a, b) => a + b, 0) < 0.6,
  },
  {
    id: 'dead-end',
    title: 'A page with no links',
    text: 'Page D has no outgoing links. A surfer that reaches it jumps to a random page, so no rank is lost (the ranks still add to one), and because many pages cite D it comes out on top.',
    scenario: { web: 'deadend' },
    claim: (r) => argmax(r.exact) === 3 && Math.abs(r.exact.reduce((a, b) => a + b, 0) - 1) < 1e-9,
  },
  {
    id: 'start',
    title: 'Where the iteration starts',
    text: 'Zero steps: every page starts with the same rank, which is far from the final answer. The first step does most of the work.',
    scenario: { iterations: 0 },
    claim: (r) => r.error[0]! > 0.4,
  },
  {
    id: 'few-steps',
    title: 'Three steps in',
    text: 'After three multiplications by the matrix the ranks are already close, but not exact: each step shrinks the error by about the second eigenvalue.',
    scenario: { iterations: 3 },
    claim: (r) => r.error[3]! > 0.03 && r.error[3]! < 0.1,
  },
  {
    id: 'surfer-short',
    title: 'A surfer who just started',
    text: 'After a hundred clicks the fraction of time on each page is only a rough guide to the rank.',
    scenario: { surfers: 100 },
    claim: (r) => r.surferGap > 0.05,
  },
  {
    id: 'surfer-long',
    title: 'A surfer who never stops',
    text: 'After 200,000 clicks the share of time on each page matches the PageRank to three decimals: PageRank is the long-run behaviour of a random walk.',
    scenario: { surfers: 200000 },
    claim: (r) => r.surferGap < 0.005,
  },
];
