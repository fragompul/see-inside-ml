import { GRID_PRESETS } from '@sim/deep';
import type { ParamSpec } from '@/shared/lab-kit';
import type { DqnLayout, DqnRunAll, DqnScenario } from './dqnCore';

export const dqnParams: ParamSpec[] = [
  {
    key: 'layout',
    label: 'The world',
    kind: 'select',
    options: (['open', 'maze'] as DqnLayout[]).map((id) => ({
      value: id,
      label: (GRID_PRESETS[id] as { label: string }).label,
    })),
  },
  {
    key: 'episodes',
    label: 'Episodes',
    kind: 'ladder',
    values: [100, 150],
    format: (v) => String(v),
  },
  {
    key: 'targetEvery',
    label: 'Target network refresh (steps)',
    kind: 'ladder',
    values: [20, 100, 400],
    format: (v) => String(v),
    hint: 'How often the target network copies the online one.',
  },
];

export interface DqnExperiment {
  id: string;
  title: string;
  text: string;
  scenario: Partial<DqnScenario>;
  claim: (run: DqnRunAll) => boolean;
}

export const dqnExperiments: DqnExperiment[] = [
  {
    id: 'works',
    title: 'Replay and a target network learn the room',
    text: 'With both devices, three of four runs find the goal within 150 episodes and the average return settles near 0.6. The largest value the network ever predicts stays close to 1, the reward for reaching the goal: it is learning real values.',
    scenario: { layout: 'open', targetEvery: 100 },
    claim: (r) => {
      const full = r.results.full as DqnRunAll['results'][string];
      return full.solved >= 3 && full.tail > 0.5 && full.maxQ < 2;
    },
  },
  {
    id: 'target',
    title: 'Without a target network the values run away',
    text: 'If the network chases a target made from itself, every update moves its own goalposts. Its largest predicted value reaches 15 even though no reward in this world exceeds 1, twelve times as high as with a target network.',
    scenario: { layout: 'open', targetEvery: 100 },
    claim: (r) =>
      (r.results['no-target']?.maxQ as number) > 8 * (r.results.full?.maxQ as number) &&
      (r.results['no-target']?.maxQ as number) > 5,
  },
  {
    id: 'replay',
    title: 'One transition at a time cannot keep up',
    text: 'Updating from the latest single step, against a target that only changes every 100 steps, passes information back along the route far too slowly: after 150 episodes none of the runs has found the goal and they time out every episode.',
    scenario: { layout: 'open', targetEvery: 100 },
    claim: (r) =>
      (r.results['no-replay']?.solved as number) === 0 &&
      (r.results['no-replay']?.tail as number) < -2 &&
      (r.results.full?.tail as number) > 0.5,
  },
  {
    id: 'slower',
    title: 'The plain version gets there, later',
    text: 'In an easy room even the plain online network, with neither device, learns the route eventually. After 50 episodes it is still at an average return of about −0.1 while the full method is already at 0.5.',
    scenario: { layout: 'open', targetEvery: 100 },
    claim: (r) =>
      (r.results.plain?.curve[49] as number) < (r.results.full?.curve[49] as number) - 0.4 &&
      (r.results.plain?.solved as number) >= 2,
  },
  {
    id: 'maze',
    title: 'In the maze only the full method copes',
    text: 'With walls to get round, three of four full runs reach the exit, one of four with replay alone, and none of the plain or target-only runs.',
    scenario: { layout: 'maze', targetEvery: 100 },
    claim: (r) =>
      (r.results.full?.solved as number) >= 3 &&
      (r.results['no-target']?.solved as number) <= 1 &&
      (r.results.plain?.solved as number) === 0 &&
      (r.results['no-replay']?.solved as number) === 0,
  },
];
