import { GRID_PRESETS } from '@sim/deep';
import type { ParamSpec } from '@/shared/lab-kit';
import { LAMBDAS, runAc, type AcLayout, type AcRun, type AcScenario } from './acCore';

export const acParams: ParamSpec[] = [
  {
    key: 'layout',
    label: 'The world',
    kind: 'select',
    options: (['open', 'cliff', 'maze'] as AcLayout[]).map((id) => ({
      value: id,
      label: (GRID_PRESETS[id] as { label: string }).label,
    })),
  },
  {
    key: 'slip',
    label: 'Slippery floor',
    kind: 'ladder',
    values: [0, 0.2],
    format: (v) => `${Math.round(v * 100)}%`,
  },
  {
    key: 'lambda',
    label: 'Trace decay λ of the shown run',
    kind: 'ladder',
    values: LAMBDAS,
    format: (v) => String(v),
    hint: '0 trusts the critic after one step (A2C); 1 trusts only observed rewards.',
  },
  {
    key: 'actorRate',
    label: 'Actor learning rate',
    kind: 'ladder',
    values: [0.05, 0.1, 0.2],
    format: (v) => String(v),
  },
];

export interface AcExperiment {
  id: string;
  title: string;
  text: string;
  scenario: Partial<AcScenario>;
  claim: (run: AcRun) => boolean;
}

const v = (r: AcRun, k: number) => r.variance[String(k)] as number;
const b = (r: AcRun, k: number) => r.bias[String(k)] as number;
const f = (r: AcRun, k: number) => r.final[String(k)] as number;

export const acExperiments: AcExperiment[] = [
  {
    id: 'variance',
    title: 'The critic tames the noise',
    text: 'At one fixed policy, the single-episode gradient estimate that waits for the whole return (λ = 1) is five times noisier than the one that trusts the critic after one step (λ = 0). Intermediate values sit between.',
    scenario: { layout: 'open', slip: 0 },
    claim: (r) => v(r, 1) > 3 * v(r, 0) && v(r, 1) > v(r, 0.9) && v(r, 0.9) > v(r, 0.5),
  },
  {
    id: 'bias',
    title: 'The price is bias',
    text: 'The critic is still learning and wrong in places, so an estimate built on it points slightly the wrong way on average. The mean of the λ = 0 estimate lies three times as far from the unbiased one (λ = 1) as the λ = 0.9 estimate does.',
    scenario: { layout: 'open', slip: 0 },
    claim: (r) =>
      b(r, 0) > 3 * b(r, 0.9) &&
      b(r, 0) > b(r, 0.5) &&
      b(r, 0.5) > b(r, 0.9) &&
      b(r, 0.9) >= b(r, 1) - 1e-9,
  },
  {
    id: 'easy',
    title: 'In an easy world every λ works',
    text: 'In the open room all four settings reach most of the best possible return within 300 episodes: a noisy gradient is fine when the task is simple.',
    scenario: { layout: 'open', slip: 0 },
    claim: (r) => LAMBDAS.every((k) => f(r, k) > 0.7 * r.optimal),
  },
  {
    id: 'hard',
    title: 'In a hard world the noise decides',
    text: 'In the maze, the same three critic-driven settings find the exit within 300 episodes, but λ = 1, which waits for whole returns, is still stuck at a negative return, thrown about by its own noise.',
    scenario: { layout: 'maze', slip: 0 },
    claim: (r) => f(r, 1) < 0 && f(r, 0) > 0.4 && f(r, 0.5) > 0.4 && f(r, 0.9) > 0.4,
  },
  {
    id: 'slip',
    title: 'Randomness in the world raises the noise',
    text: 'With a slippery floor each return is partly luck. The variance of the gradient estimate rises for every λ, and the most for the ones that rely on whole returns.',
    scenario: { layout: 'open', slip: 0.2 },
    claim: (r) => {
      const calm = runAc({ ...r.scenario, slip: 0 });
      return (
        LAMBDAS.every((k) => v(r, k) > v(calm, k)) && v(r, 1) - v(calm, 1) > v(r, 0) - v(calm, 0)
      );
    },
  },
];
