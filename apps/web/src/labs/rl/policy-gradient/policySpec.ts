import { GRID_PRESETS } from '@sim/deep';
import type { ParamSpec } from '@/shared/lab-kit';
import {
  BASELINES,
  BASELINE_LABELS,
  type LayoutId,
  type PolicyRun,
  type PolicyScenario,
} from './policyCore';

export const policyParams: ParamSpec[] = [
  {
    key: 'layout',
    label: 'The world',
    kind: 'select',
    options: (['open', 'maze', 'detour', 'cliff'] as LayoutId[]).map((id) => ({
      value: id,
      label: (GRID_PRESETS[id] as { label: string }).label,
    })),
  },
  {
    key: 'baseline',
    label: 'Baseline subtracted from the return',
    kind: 'segmented',
    options: BASELINES.map((b) => ({ value: b, label: BASELINE_LABELS[b] })),
  },
  {
    key: 'learningRate',
    label: 'Learning rate',
    kind: 'ladder',
    values: [0.03, 0.1, 0.3, 1, 3],
    format: (v) => String(v),
  },
  {
    key: 'episodes',
    label: 'Training episodes',
    kind: 'ladder',
    values: [100, 400, 1000],
    format: (v) => String(v),
  },
  {
    key: 'batch',
    label: 'Episodes per update',
    kind: 'ladder',
    values: [1, 4, 8],
    format: (v) => String(v),
  },
  {
    key: 'entropy',
    label: 'Entropy bonus',
    kind: 'ladder',
    values: [0, 0.05, 0.3],
    format: (v) => String(v),
    hint: 'Rewards keeping the policy random, which keeps it exploring.',
  },
  {
    key: 'slip',
    label: 'Slippery floor',
    kind: 'ladder',
    values: [0, 0.2, 0.4],
    format: (v) => `${Math.round(v * 100)}%`,
    hint: 'The chance that a move goes in a random direction.',
  },
];

export interface PolicyExperiment {
  id: string;
  title: string;
  text: string;
  scenario: Partial<PolicyScenario>;
  claim: (run: PolicyRun) => boolean;
}

export const policyExperiments: PolicyExperiment[] = [
  {
    id: 'learns',
    title: 'Learning by trial and reward',
    text: 'The policy starts uniformly random. Every episode that reaches the goal makes the actions it contained more likely, and every long wander makes them less likely. After a few hundred episodes the agent walks the shortest path every time.',
    scenario: { layout: 'open', learningRate: 0.3 },
    claim: (r) => r.goalRate === 1 && r.finalReturn > r.optimal - 0.1,
  },
  {
    id: 'baseline-maze',
    title: 'A baseline steadies the maze',
    text: 'In the maze the plain estimator is so noisy that some runs never find a stable route: averaged over six runs its final return is about zero, with wild variation. Subtracting a baseline leaves the expected update unchanged but cuts the noise, and every run finds the path.',
    scenario: { layout: 'maze', learningRate: 0.3, episodes: 1000 },
    claim: (r) =>
      r.summary.none.mean < 0.3 &&
      r.summary.average.mean > 0.6 &&
      r.summary.value.mean > 0.6 &&
      r.summary.none.sd > 0.5,
  },
  {
    id: 'baseline-detour',
    title: 'A baseline in the detour',
    text: 'Again with the plain estimator some runs get stuck by the pit while the baseline versions learn the safe detour reliably.',
    scenario: { layout: 'detour', learningRate: 0.3, episodes: 1000 },
    claim: (r) => r.summary.value.mean - r.summary.none.mean > 0.2 && r.summary.none.sd > 0.5,
  },
  {
    id: 'big-steps',
    title: 'A learning rate that is too big',
    text: 'At a learning rate of 3 a single lucky or unlucky episode can swing the policy to one extreme. Without a baseline half the runs collapse (high spread); with a learned baseline the advantages stay small and the agent still learns.',
    scenario: { learningRate: 3 },
    claim: (r) => r.summary.none.sd > 0.8 && r.summary.value.mean > 0.7,
  },
  {
    id: 'slippery',
    title: 'A slippery floor',
    text: 'With a 20% chance that any move goes astray, the agent still learns to reach the goal every time, with a lower return because it wanders more.',
    scenario: { slip: 0.2, learningRate: 0.3 },
    claim: (r) => r.goalRate === 1 && r.finalReturn > 0.5 && r.finalReturn < r.optimal + 0.05,
  },
  {
    id: 'batch',
    title: 'Fewer, steadier updates',
    text: 'Averaging eight episodes per update gives a cleaner gradient but eight times fewer updates for the same number of episodes: after 400 episodes the agent has learned less than with one episode per update.',
    scenario: { batch: 8, learningRate: 0.3 },
    claim: (r) => r.finalReturn < r.optimal - 0.2 && r.goalRate > 0.9,
  },
  {
    id: 'few-episodes',
    title: 'Only 100 episodes',
    text: 'Early in training the policy still wanders: most episodes reach the goal, but slowly.',
    scenario: { episodes: 100, learningRate: 0.3 },
    claim: (r) => r.finalReturn < 0.5 && r.goalRate > 0.8,
  },
  {
    id: 'cliff',
    title: 'The cliff: exploration fails',
    text: 'On the cliff the goal is far and one random step at the edge ends the episode with a heavy penalty. The policy learns to avoid the cliff by staying put, which is safe but pays nothing: policy gradients can settle in a poor local optimum when the rewarding path is hard to find by chance.',
    scenario: { layout: 'cliff', learningRate: 0.3, episodes: 1000 },
    claim: (r) => BASELINES.every((b) => r.summary[b].mean < -1),
  },
];
