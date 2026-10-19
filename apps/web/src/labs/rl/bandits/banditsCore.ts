import {
  BANDIT_STRATEGIES,
  meanRegret,
  runBandit,
  type BanditProblem,
  type BanditRun,
  type BanditStrategy,
} from '@sim/deep';

export type ProblemId = 'easy' | 'close' | 'noisy' | 'many';

export const PROBLEMS: Record<ProblemId, { label: string; note: string; problem: BanditProblem }> =
  {
    easy: {
      label: 'Four coins',
      note: 'Four biased coins that pay 1 when they land heads, with probabilities 0.2, 0.4, 0.5 and 0.6.',
      problem: { means: [0.2, 0.4, 0.5, 0.6], noise: 0 },
    },
    close: {
      label: 'Nearly equal coins',
      note: 'Three coins with probabilities 0.50, 0.52 and 0.55: it takes thousands of flips to tell them apart.',
      problem: { means: [0.5, 0.52, 0.55], noise: 0 },
    },
    noisy: {
      label: 'Noisy payouts',
      note: 'Four machines with average payouts 0, 0.5, 1 and 0.8, and noise (standard deviation 1) bigger than the gaps.',
      problem: { means: [0, 0.5, 1, 0.8], noise: 1 },
    },
    many: {
      label: 'Ten machines',
      note: 'Ten machines with payouts from 0 to 0.9 and noise of 1: lots to explore.',
      problem: { means: [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], noise: 1 },
    },
  };

export const STRATEGY_LABELS: Record<BanditStrategy, string> = {
  greedy: 'Greedy',
  epsilon: 'ε-greedy (ε = 0.1)',
  decaying: 'Decaying ε-greedy',
  optimistic: 'Optimistic start',
  ucb: 'UCB',
  thompson: 'Thompson sampling',
  softmax: 'Softmax',
  gradient: 'Gradient bandit',
};

export const STRATEGY_NOTES: Record<BanditStrategy, string> = {
  greedy:
    'Always pull the arm that looks best so far. Never explores, so one unlucky early pull can doom it.',
  epsilon:
    'Pull the best-looking arm, but a random one 10% of the time: explores forever, so it keeps paying for it.',
  decaying: 'Like ε-greedy, but ε falls as 1/t: lots of exploring early, little later.',
  optimistic:
    'Start by believing every arm is great; disappointment drives exploration until the estimates fall to the truth.',
  ucb: 'Pull the arm with the highest optimistic bound: its average plus a bonus for how little it has been tried.',
  thompson:
    'Draw a plausible value for each arm from what you know about it and pull the arm whose draw is highest.',
  softmax: 'Pull arms with probability growing with their estimated value (temperature 0.1).',
  gradient:
    'Learn a preference for each arm by gradient ascent on the expected reward: a policy gradient in miniature.',
};

export interface BanditScenario {
  problem: ProblemId;
  steps: number;
  strategies: BanditStrategy[];
  epsilon: number;
  seed: number;
}

export const DEFAULT_BANDITS: BanditScenario = {
  problem: 'easy',
  steps: 1000,
  strategies: [...BANDIT_STRATEGIES],
  epsilon: 0.1,
  seed: 3,
};

export const RUNS = 60;

export interface BanditRace {
  strategy: BanditStrategy;
  /** Average cumulative regret after each step, over RUNS independent runs. */
  curve: number[];
  final: number;
  /** Regret accumulated in the second half divided by that of the first half: ~1 = linear, well below 1 = it has learned. */
  lateShare: number;
}

export interface BanditsRun {
  scenario: BanditScenario;
  problem: BanditProblem;
  race: BanditRace[];
  /** One run of the strategy chosen for the detail view. */
  single: BanditRun;
}

export function runBandits(s: BanditScenario, detail: BanditStrategy): BanditsRun {
  const problem = PROBLEMS[s.problem].problem;
  const options = { steps: s.steps, seed: s.seed, epsilon: s.epsilon };
  const race = s.strategies.map((strategy) => {
    const curve = meanRegret(problem, strategy, options, RUNS);
    const half = curve[Math.floor(s.steps / 2) - 1] as number;
    const final = curve[s.steps - 1] as number;
    return { strategy, curve, final, lateShare: half > 0 ? (final - half) / half : 0 };
  });
  return {
    scenario: s,
    problem,
    race,
    single: runBandit(problem, detail, { ...options, seed: s.seed }),
  };
}

export const raceOf = (run: BanditsRun, st: BanditStrategy): BanditRace =>
  run.race.find((r) => r.strategy === st) as BanditRace;
