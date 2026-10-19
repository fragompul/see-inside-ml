import { createRng, type Rng } from '@sim/core';

/**
 * Multi-armed bandits: the purest exploration-versus-exploitation problem. Several slot machines pay
 * out at different (unknown) rates; each pull both earns a reward and teaches you something. Regret
 * is what you gave up by not always pulling the best arm.
 */
export type BanditStrategy =
  'greedy' | 'epsilon' | 'decaying' | 'optimistic' | 'ucb' | 'thompson' | 'softmax' | 'gradient';

export const BANDIT_STRATEGIES: BanditStrategy[] = [
  'greedy',
  'epsilon',
  'decaying',
  'optimistic',
  'ucb',
  'thompson',
  'softmax',
  'gradient',
];

export interface BanditProblem {
  /** True mean reward of each arm. */
  means: number[];
  /** Standard deviation of the Gaussian reward noise; 0 means Bernoulli arms (reward 0 or 1 with probability = mean). */
  noise: number;
}

export interface BanditOptions {
  steps: number;
  seed: number;
  /** Exploration rate of ε-greedy. */
  epsilon?: number;
  /** Starting estimate for optimistic initialisation. */
  optimism?: number;
  /** Confidence multiplier of UCB. */
  confidence?: number;
  /** Temperature of the softmax strategy. */
  temperature?: number;
  /** Step size of the gradient bandit. */
  learningRate?: number;
}

export interface BanditRun {
  strategy: BanditStrategy;
  /** Arm pulled at every step. */
  pulls: number[];
  rewards: number[];
  /** Cumulative pseudo-regret: sum of (best mean - mean of the pulled arm). */
  regret: number[];
  counts: number[];
  estimates: number[];
}

export function pullArm(problem: BanditProblem, arm: number, rng: Rng): number {
  const mean = problem.means[arm] as number;
  return problem.noise > 0 ? mean + rng.normal() * problem.noise : rng.next() < mean ? 1 : 0;
}

function sampleBeta(a: number, b: number, rng: Rng): number {
  // via two gamma draws (Marsaglia-Tsang)
  const gamma = (shape: number): number => {
    if (shape < 1) return gamma(shape + 1) * rng.next() ** (1 / shape);
    const d = shape - 1 / 3;
    const c = 1 / Math.sqrt(9 * d);
    for (;;) {
      let x: number;
      let v: number;
      do {
        x = rng.normal();
        v = 1 + c * x;
      } while (v <= 0);
      v = v * v * v;
      const u = rng.next();
      if (u < 1 - 0.0331 * x ** 4 || Math.log(u) < 0.5 * x * x + d * (1 - v + Math.log(v)))
        return d * v;
    }
  };
  const x = gamma(a);
  const y = gamma(b);
  return x / (x + y);
}

export function runBandit(
  problem: BanditProblem,
  strategy: BanditStrategy,
  options: BanditOptions,
): BanditRun {
  const k = problem.means.length;
  const rng = createRng(options.seed);
  const best = Math.max(...problem.means);
  const {
    epsilon = 0.1,
    optimism = 5,
    confidence = 2,
    temperature = 0.1,
    learningRate = 0.1,
  } = options;
  const counts = new Array<number>(k).fill(0);
  const sums = new Array<number>(k).fill(0);
  const estimates = new Array<number>(k).fill(strategy === 'optimistic' ? optimism : 0);
  // gradient bandit
  const preference = new Array<number>(k).fill(0);
  let baseline = 0;
  // Thompson sampling: Beta posteriors for Bernoulli arms, Normal for Gaussian arms
  const wins = new Array<number>(k).fill(1);
  const losses = new Array<number>(k).fill(1);
  const pulls: number[] = [];
  const rewards: number[] = [];
  const regret: number[] = [];
  let total = 0;
  const argmax = (v: readonly number[]) => {
    let top = -Infinity;
    const ties: number[] = [];
    v.forEach((x, i) => {
      if (x > top + 1e-12) {
        top = x;
        ties.length = 0;
        ties.push(i);
      } else if (Math.abs(x - top) <= 1e-12) ties.push(i);
    });
    return ties[Math.floor(rng.next() * ties.length)] as number;
  };
  for (let t = 0; t < options.steps; t++) {
    let arm: number;
    if (strategy === 'greedy' || strategy === 'optimistic') arm = argmax(estimates);
    else if (strategy === 'epsilon' || strategy === 'decaying') {
      const eps = strategy === 'decaying' ? Math.min(1, (epsilon * 20) / (t + 1)) : epsilon;
      arm = rng.next() < eps ? Math.floor(rng.next() * k) : argmax(estimates);
    } else if (strategy === 'ucb') {
      const unpulled = counts.findIndex((c) => c === 0);
      arm =
        unpulled >= 0
          ? unpulled
          : argmax(
              estimates.map(
                (m, i) =>
                  m +
                  confidence *
                    Math.sqrt(Math.log(t + 1) / (counts[i] as number)) *
                    (problem.noise > 0 ? problem.noise : 0.5),
              ),
            );
    } else if (strategy === 'thompson') {
      arm = argmax(
        estimates.map((_, i) => {
          if (problem.noise === 0) return sampleBeta(wins[i] as number, losses[i] as number, rng);
          const n = counts[i] as number;
          return (estimates[i] as number) + (rng.normal() * problem.noise) / Math.sqrt(n + 1);
        }),
      );
    } else if (strategy === 'softmax') {
      const m = Math.max(...estimates);
      const w = estimates.map((e) => Math.exp((e - m) / temperature));
      const z = w.reduce((a, b) => a + b, 0);
      let u = rng.next() * z;
      arm = k - 1;
      for (let i = 0; i < k; i++) {
        u -= w[i] as number;
        if (u <= 0) {
          arm = i;
          break;
        }
      }
    } else {
      const m = Math.max(...preference);
      const w = preference.map((h) => Math.exp(h - m));
      const z = w.reduce((a, b) => a + b, 0);
      let u = rng.next() * z;
      arm = k - 1;
      for (let i = 0; i < k; i++) {
        u -= w[i] as number;
        if (u <= 0) {
          arm = i;
          break;
        }
      }
    }
    const r = pullArm(problem, arm, rng);
    counts[arm] = (counts[arm] as number) + 1;
    sums[arm] = (sums[arm] as number) + r;
    if (strategy === 'gradient') {
      baseline += (r - baseline) / (t + 1);
      const m = Math.max(...preference);
      const w = preference.map((h) => Math.exp(h - m));
      const z = w.reduce((a, b) => a + b, 0);
      for (let i = 0; i < k; i++) {
        const p = (w[i] as number) / z;
        preference[i] =
          (preference[i] as number) + learningRate * (r - baseline) * ((i === arm ? 1 : 0) - p);
      }
      estimates[arm] = (sums[arm] as number) / (counts[arm] as number);
    } else if (strategy === 'optimistic') {
      // the optimistic start counts as one pseudo-observation, then the usual average takes over
      estimates[arm] =
        ((estimates[arm] as number) * (counts[arm] as number) + r) / ((counts[arm] as number) + 1);
    } else {
      estimates[arm] = (sums[arm] as number) / (counts[arm] as number);
    }
    if (r > 0.5 || problem.noise > 0) wins[arm] = (wins[arm] as number) + (r > 0.5 ? 1 : 0);
    if (r <= 0.5 && problem.noise === 0) losses[arm] = (losses[arm] as number) + 1;
    total += best - (problem.means[arm] as number);
    pulls.push(arm);
    rewards.push(r);
    regret.push(total);
  }
  return { strategy, pulls, rewards, regret, counts, estimates };
}

/** Average regret curve over many independent runs. */
export function meanRegret(
  problem: BanditProblem,
  strategy: BanditStrategy,
  options: BanditOptions,
  runs: number,
): number[] {
  const acc = new Array<number>(options.steps).fill(0);
  for (let r = 0; r < runs; r++) {
    const run = runBandit(problem, strategy, { ...options, seed: options.seed * 1000 + r });
    run.regret.forEach((v, i) => (acc[i] = (acc[i] as number) + v / runs));
  }
  return acc;
}
