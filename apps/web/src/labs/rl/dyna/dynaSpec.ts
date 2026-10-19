import { GRID_PRESETS } from '@sim/deep';
import type { ParamSpec } from '@/shared/lab-kit';
import {
  episodesToLearn,
  runDyna,
  type DynaLabRun,
  type DynaLayout,
  type DynaScenario,
} from './dynaCore';

export const dynaParams: ParamSpec[] = [
  {
    key: 'layout',
    label: 'The world',
    kind: 'select',
    options: (['open', 'maze', 'detour'] as DynaLayout[]).map((id) => ({
      value: id,
      label: (GRID_PRESETS[id] as { label: string }).label,
    })),
  },
  {
    key: 'planning',
    label: 'Imagined updates per real step',
    kind: 'ladder',
    values: [0, 5, 20, 50],
    format: (v) => String(v),
    hint: '0 is ordinary Q-learning.',
  },
  {
    key: 'slip',
    label: 'Slippery floor',
    kind: 'ladder',
    values: [0, 0.2, 0.4],
    format: (v) => `${Math.round(v * 100)}%`,
  },
  {
    key: 'episodes',
    label: 'Episodes',
    kind: 'ladder',
    values: [15, 30, 60],
    format: (v) => String(v),
  },
];

export interface DynaExperiment {
  id: string;
  title: string;
  text: string;
  scenario: Partial<DynaScenario>;
  claim: (run: DynaLabRun) => boolean;
}

export const dynaExperiments: DynaExperiment[] = [
  {
    id: 'q-learning',
    title: 'Plain Q-learning is slow',
    text: 'Without planning, each real step updates one table entry, and the news of the goal travels back along the path one step per episode. In the maze even 30 episodes are not enough to settle on the shortest route.',
    scenario: { layout: 'maze', planning: 0 },
    claim: (r) => episodesToLearn(r.curve, r.shortest) === Infinity,
  },
  {
    id: 'planning',
    title: 'Rehearsing in imagination',
    text: 'With 20 imagined updates after every real step the agent replays what it remembers, and the news spreads through the table in a few episodes: less than half the real steps are needed, and the route is learned in about nine episodes.',
    scenario: { layout: 'maze', planning: 20 },
    claim: (r) => r.realSteps < r.baselineSteps * 0.5 && episodesToLearn(r.curve, r.shortest) <= 12,
  },
  {
    id: 'open-room',
    title: 'In the open room',
    text: 'With 20 imagined updates the agent settles on the shortest path in about five episodes. Plain Q-learning needs more than twenty.',
    scenario: { layout: 'open', planning: 20 },
    claim: (r) =>
      episodesToLearn(r.curve, r.shortest) <= 6 && episodesToLearn(r.baseline, r.shortest) >= 20,
  },
  {
    id: 'diminishing',
    title: 'Diminishing returns',
    text: 'Going from 5 to 50 imagined updates per step makes the agent ten times as busy in its head but saves under a third of the real steps: once the model has been replayed enough, more thinking adds little.',
    scenario: { layout: 'detour', planning: 50 },
    claim: (r) => {
      const few = runDyna({ ...r.scenario, planning: 5 });
      return r.realSteps > 0.65 * few.realSteps && r.realSteps < few.realSteps;
    },
  },
  {
    id: 'cost',
    title: 'Imagination is not free',
    text: 'Real steps are the scarce resource (a robot wears out), imagined ones cost only computing: with 20 per real step this agent does twenty times as many imagined updates as real moves.',
    scenario: { planning: 20 },
    claim: (r) => r.single.imaginedSteps === 20 * r.single.realSteps,
  },
  {
    id: 'slippery',
    title: 'A slippery floor',
    text: 'When moves sometimes go astray the model (what happened last time) is only a sample, but replaying it still helps a lot: about 45% fewer real steps.',
    scenario: { layout: 'maze', planning: 20, slip: 0.2 },
    claim: (r) => r.realSteps < r.baselineSteps * 0.65,
  },
  {
    id: 'wrong-model',
    title: 'When the model is wrong',
    text: 'On a very slippery floor each remembered step is a single random outcome, treated as if it always happens. Replaying those flukes teaches a distorted picture: after 30 episodes the agent with planning is worse than plain Q-learning.',
    scenario: { layout: 'maze', planning: 20, slip: 0.4 },
    claim: (r) =>
      (r.curve[r.curve.length - 1] as number) > 1.5 * (r.baseline[r.baseline.length - 1] as number),
  },
];
