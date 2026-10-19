import type { ParamSpec } from '@/shared/lab-kit';
import {
  PROBLEMS,
  raceOf,
  runBandits,
  type BanditScenario,
  type BanditsRun,
  type ProblemId,
} from './banditsCore';

export const banditParams: ParamSpec[] = [
  {
    key: 'problem',
    label: 'The machines',
    kind: 'select',
    options: (Object.keys(PROBLEMS) as ProblemId[]).map((id) => ({
      value: id,
      label: PROBLEMS[id].label,
    })),
  },
  {
    key: 'steps',
    label: 'Pulls',
    kind: 'ladder',
    values: [200, 500, 1000, 5000],
    format: (v) => String(v),
  },
  {
    key: 'epsilon',
    label: 'ε of ε-greedy',
    kind: 'ladder',
    values: [0.01, 0.05, 0.1, 0.3],
    format: (v) => String(v),
  },
];

export interface BanditExperiment {
  id: string;
  title: string;
  text: string;
  scenario: Partial<BanditScenario>;
  claim: (run: BanditsRun) => boolean;
}

export const banditExperiments: BanditExperiment[] = [
  {
    id: 'greedy-lock',
    title: 'Greedy locks onto a loser',
    text: 'Never exploring means one unlucky early pull of the best coin can hide it for ever: on average greedy loses more than four times as much as Thompson sampling, and its regret keeps growing at the same rate throughout.',
    scenario: {},
    claim: (r) =>
      raceOf(r, 'greedy').final > 4 * raceOf(r, 'thompson').final &&
      raceOf(r, 'greedy').lateShare > 0.95,
  },
  {
    id: 'optimism',
    title: 'Optimism is a cheap way to explore',
    text: 'Starting every estimate at 5 (far above any real payout) makes each arm disappoint in turn: all arms get tried early, systematically, and then the estimates settle. Here that beats every other strategy.',
    scenario: {},
    claim: (r) =>
      r.race.every((e) => e.strategy === 'optimistic' || raceOf(r, 'optimistic').final < e.final),
  },
  {
    id: 'smarter',
    title: 'Directed exploration beats random',
    text: 'After 5,000 pulls ε-greedy is still spending 10% of its pulls on random arms. Strategies that explore where the uncertainty is (UCB, Thompson sampling) and the gradient bandit lose a fraction of that.',
    scenario: { steps: 5000 },
    claim: (r) =>
      raceOf(r, 'thompson').final < raceOf(r, 'epsilon').final / 3 &&
      raceOf(r, 'ucb').final < raceOf(r, 'epsilon').final &&
      raceOf(r, 'gradient').final < raceOf(r, 'epsilon').final,
  },
  {
    id: 'too-little',
    title: 'Exploring too little',
    text: 'ε = 0.01 explores so rarely that it spends a long time on a wrong arm before finding out: its regret is much higher than with ε = 0.1.',
    scenario: { epsilon: 0.01 },
    claim: (r) => {
      const ref = runBandits({ ...r.scenario, epsilon: 0.1, strategies: ['epsilon'] }, 'epsilon');
      return raceOf(r, 'epsilon').final > 1.4 * raceOf(ref, 'epsilon').final;
    },
  },
  {
    id: 'close-coins',
    title: 'When arms are nearly equal',
    text: 'If the best coin is only three points better than the others, picking the wrong one costs little per pull: every strategy, even greedy, ends with a modest regret, and the differences between them are small.',
    scenario: { problem: 'close' },
    claim: (r) => r.race.every((e) => e.final < 30),
  },
  {
    id: 'noise',
    title: 'Noise makes greedy worse',
    text: 'With payout noise larger than the gaps between machines, an early sample says almost nothing, and greedy is stuck with whatever it saw first. Thompson sampling and UCB, which keep track of how uncertain each estimate is, do far better.',
    scenario: { problem: 'noisy' },
    claim: (r) =>
      raceOf(r, 'greedy').final > 3 * raceOf(r, 'thompson').final &&
      raceOf(r, 'greedy').final > 2 * raceOf(r, 'ucb').final,
  },
  {
    id: 'many-arms',
    title: 'Ten machines: UCB pays to try them all',
    text: 'UCB insists on trying every arm several times before it trusts any, which costs a lot when there are ten, and more than the random exploration of ε-greedy.',
    scenario: { problem: 'many' },
    claim: (r) => raceOf(r, 'ucb').final > 1.4 * raceOf(r, 'epsilon').final,
  },
];
