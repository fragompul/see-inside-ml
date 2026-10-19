import { createRng, type Rng } from '@sim/core';
import { ACTION_COUNT, GRID_PRESETS, GridWorld, parseGrid, type GridRewards } from './grid';
import { valueIteration, type OptimalSolution } from './planning';

export const RL_ALGORITHMS = ['qlearning', 'sarsa', 'expected-sarsa'] as const;
export type RlAlgorithm = (typeof RL_ALGORITHMS)[number];

export const algorithmInfo: Record<RlAlgorithm, { label: string; note: string; formula: string }> =
  {
    qlearning: {
      label: 'Q-learning',
      note: 'Off-policy: learns the value of the best action, whatever it actually does next.',
      formula: String.raw`Q(s,a)\leftarrow Q(s,a)+\alpha\Big[r+\gamma\max_{a'}Q(s',a')-Q(s,a)\Big]`,
    },
    sarsa: {
      label: 'SARSA',
      note: 'On-policy: learns the value of what it really does, including its exploratory slips.',
      formula: String.raw`Q(s,a)\leftarrow Q(s,a)+\alpha\Big[r+\gamma\,Q(s',a')-Q(s,a)\Big]`,
    },
    'expected-sarsa': {
      label: 'Expected SARSA',
      note: 'Averages over what it might do next instead of sampling one action: less noisy than SARSA.',
      formula: String.raw`Q(s,a)\leftarrow Q(s,a)+\alpha\Big[r+\gamma\sum_{a'}\pi(a'|s')Q(s',a')-Q(s,a)\Big]`,
    },
  };

export interface RlLabConfig {
  /** The layout, one string per row (see `grid.ts`). */
  grid: string[];
  algorithm: RlAlgorithm;
  alpha: number;
  gamma: number;
  /** Probability of a random action at the start of training. */
  epsilon: number;
  /** The exploration rate never decays below this. */
  epsilonMin: number;
  /** Multiplied into the exploration rate after every episode. */
  epsilonDecay: number;
  /** Probability that the floor is slippery and a random action happens instead. */
  slip: number;
  stepReward: number;
  /** An episode is cut off after this many steps. */
  maxSteps: number;
  seed: number;
}

export const DEFAULT_RL_LAB: RlLabConfig = {
  grid: [...(GRID_PRESETS.cliff as { rows: string[] }).rows],
  algorithm: 'qlearning',
  alpha: 0.5,
  gamma: 0.95,
  epsilon: 0.3,
  epsilonMin: 0.1,
  epsilonDecay: 0.995,
  slip: 0,
  stepReward: -1,
  maxSteps: 200,
  seed: 3,
};

export const RL_LAB_LIMITS = {
  alpha: [0.01, 1],
  gamma: [0, 0.999],
  epsilon: [0, 1],
  slip: [0, 0.6],
  stepReward: [-5, 1],
  maxSteps: [20, 1000],
} as const;

export interface EpisodeStats {
  episode: number;
  /** Total reward collected in the episode. */
  ret: number;
  steps: number;
  /** Exploration rate during the episode. */
  epsilon: number;
  outcome: 'goal' | 'pit' | 'timeout';
}

export function validateRlLab(c: RlLabConfig): void {
  parseGrid(c.grid);
  if (!(RL_ALGORITHMS as readonly string[]).includes(c.algorithm))
    throw new RangeError('Unknown algorithm');
  const within = (v: number, [lo, hi]: readonly [number, number], name: string) => {
    if (!Number.isFinite(v) || v < lo || v > hi)
      throw new RangeError(`${name} must be in [${lo}, ${hi}]`);
  };
  within(c.alpha, RL_LAB_LIMITS.alpha, 'Learning rate');
  within(c.gamma, RL_LAB_LIMITS.gamma, 'Discount');
  within(c.epsilon, RL_LAB_LIMITS.epsilon, 'Exploration');
  within(c.epsilonMin, RL_LAB_LIMITS.epsilon, 'Minimum exploration');
  within(c.slip, RL_LAB_LIMITS.slip, 'Slip');
  within(c.stepReward, RL_LAB_LIMITS.stepReward, 'Step reward');
  within(c.maxSteps, RL_LAB_LIMITS.maxSteps, 'Maximum steps');
  if (!(c.epsilonDecay > 0 && c.epsilonDecay <= 1)) throw new RangeError('Decay must be in (0, 1]');
  if (c.epsilonMin > c.epsilon) throw new RangeError('Minimum exploration cannot exceed the start');
}

const stream = (seed: number, index: number) =>
  createRng((seed + Math.imul(index + 1, 0x9e3779b1)) >>> 0);

const worldKey = (c: RlLabConfig) =>
  JSON.stringify([c.grid, c.slip, c.stepReward, c.maxSteps, c.seed]);

/** A tabular agent, its gridworld and its training loop behind one small API. */
export class RlSession {
  private cfg: RlLabConfig;
  private key = '';
  world!: GridWorld;
  /** `states × 4` action values. */
  q!: Float64Array;
  visits!: Int32Array;
  optimal!: OptimalSolution;
  /** States the agent can reach from the start without passing through a terminal state. */
  reachable!: Set<number>;
  private rng!: Rng;

  agent = 0;
  private nextAction: number | null = null;
  epsilon = 0;
  episode = 0;
  totalSteps = 0;
  private episodeReturn = 0;
  private episodeSteps = 0;
  history: EpisodeStats[] = [];

  constructor(config: RlLabConfig) {
    validateRlLab(config);
    this.cfg = { ...config, grid: [...config.grid] };
    this.build();
  }

  get config(): Readonly<RlLabConfig> {
    return this.cfg;
  }

  /** Steps taken in the current (unfinished) episode. */
  get currentSteps(): number {
    return this.episodeSteps;
  }

  /** Hyper-parameters apply live; changes to the world rebuild (and restart learning). */
  update(patch: Partial<RlLabConfig>): 'live' | 'rebuilt' {
    const next = { ...this.cfg, ...patch };
    if (patch.grid) next.grid = [...patch.grid];
    validateRlLab(next);
    const changed = worldKey(next) !== this.key;
    this.cfg = next;
    if (changed) {
      this.build();
      return 'rebuilt';
    }
    if (patch.gamma !== undefined) this.optimal = valueIteration(this.world, this.cfg.gamma);
    if (patch.epsilon !== undefined || patch.epsilonMin !== undefined)
      this.epsilon = Math.max(this.cfg.epsilonMin, Math.min(this.epsilon, this.cfg.epsilon));
    return 'live';
  }

  reset(): void {
    this.build();
  }

  /** Takes `count` environment steps (episodes restart on their own); returns the episodes finished. */
  train(count: number): EpisodeStats[] {
    const finished: EpisodeStats[] = [];
    for (let n = 0; n < count; n++) {
      const stats = this.stepOnce();
      if (stats) finished.push(stats);
    }
    return finished;
  }

  /** Trains until `count` more episodes have finished. */
  trainEpisodes(count: number): EpisodeStats[] {
    const finished: EpisodeStats[] = [];
    while (finished.length < count) {
      const stats = this.stepOnce();
      if (stats) finished.push(stats);
    }
    return finished;
  }

  /** The action with the largest value (ties go to the lowest index, so it is deterministic). */
  greedyAction(state: number): number {
    let best = 0;
    for (let a = 1; a < ACTION_COUNT; a++)
      if (
        (this.q[state * ACTION_COUNT + a] as number) >
        (this.q[state * ACTION_COUNT + best] as number)
      )
        best = a;
    return best;
  }

  /** Greedy value of a state. */
  value(state: number): number {
    return this.q[state * ACTION_COUNT + this.greedyAction(state)] as number;
  }

  private chooseAction(state: number): number {
    if (this.rng.next() < this.epsilon) return Math.floor(this.rng.next() * ACTION_COUNT);
    // break ties at random so an untrained agent does not always walk the same way
    let best = -Infinity;
    const ties: number[] = [];
    for (let a = 0; a < ACTION_COUNT; a++) {
      const v = this.q[state * ACTION_COUNT + a] as number;
      if (v > best + 1e-12) {
        best = v;
        ties.length = 0;
        ties.push(a);
      } else if (Math.abs(v - best) <= 1e-12) ties.push(a);
    }
    return ties[Math.floor(this.rng.next() * ties.length)] as number;
  }

  /** The probability the ε-greedy policy picks `action` in `state`. */
  private policyProbability(state: number, action: number): number {
    let best = -Infinity;
    for (let a = 0; a < ACTION_COUNT; a++)
      best = Math.max(best, this.q[state * ACTION_COUNT + a] as number);
    const ties = Array.from(
      { length: ACTION_COUNT },
      (_, a) => Math.abs((this.q[state * ACTION_COUNT + a] as number) - best) <= 1e-12,
    );
    const count = ties.filter(Boolean).length;
    return this.epsilon / ACTION_COUNT + (ties[action] ? (1 - this.epsilon) / count : 0);
  }

  private stepOnce(): EpisodeStats | null {
    const { alpha, gamma, algorithm, maxSteps } = this.cfg;
    const s = this.agent;
    const a = this.nextAction ?? this.chooseAction(s);
    const { next, reward, done } = this.world.step(s, a, this.rng);
    this.visits[s] = (this.visits[s] as number) + 1;
    this.totalSteps += 1;
    this.episodeSteps += 1;
    this.episodeReturn += reward;

    let target = reward;
    let upcoming: number | null = null;
    if (!done) {
      if (algorithm === 'qlearning') {
        target += gamma * this.value(next);
      } else if (algorithm === 'sarsa') {
        upcoming = this.chooseAction(next);
        target += gamma * (this.q[next * ACTION_COUNT + upcoming] as number);
      } else {
        let expected = 0;
        for (let b = 0; b < ACTION_COUNT; b++)
          expected += this.policyProbability(next, b) * (this.q[next * ACTION_COUNT + b] as number);
        target += gamma * expected;
      }
    }
    const i = s * ACTION_COUNT + a;
    this.q[i] = (this.q[i] as number) + alpha * (target - (this.q[i] as number));

    const timeout = !done && this.episodeSteps >= maxSteps;
    if (done || timeout) {
      const stats: EpisodeStats = {
        episode: ++this.episode,
        ret: this.episodeReturn,
        steps: this.episodeSteps,
        epsilon: this.epsilon,
        outcome: done ? (this.world.kind(next) === 'goal' ? 'goal' : 'pit') : 'timeout',
      };
      this.history.push(stats);
      this.epsilon = Math.max(this.cfg.epsilonMin, this.epsilon * this.cfg.epsilonDecay);
      this.agent = this.world.start;
      this.nextAction = null;
      this.episodeReturn = 0;
      this.episodeSteps = 0;
      return stats;
    }
    this.agent = next;
    this.nextAction = upcoming;
    return null;
  }

  /** The route the greedy policy follows from the start on a non-slippery floor. */
  greedyPath(): { cells: number[]; outcome: 'goal' | 'pit' | 'loop' } {
    const cells = [this.world.start];
    const seen = new Set(cells);
    let state = this.world.start;
    for (let i = 0; i < this.world.states; i++) {
      const next = this.world.move(state, this.greedyAction(state));
      cells.push(next);
      if (this.world.isTerminal(next))
        return { cells, outcome: this.world.kind(next) === 'goal' ? 'goal' : 'pit' };
      if (seen.has(next)) return { cells, outcome: 'loop' };
      seen.add(next);
      state = next;
    }
    return { cells, outcome: 'loop' };
  }

  /** Share of reachable decision states where the greedy action is an optimal one (0..1). */
  agreement(): number {
    let good = 0;
    let total = 0;
    for (const s of this.reachable) {
      if (this.world.isTerminal(s)) continue;
      total += 1;
      if ((this.optimal.best[s] as boolean[])[this.greedyAction(s)]) good += 1;
    }
    return total === 0 ? 1 : good / total;
  }

  /** Mean absolute difference between the learned and the optimal state values (reachable states). */
  valueError(): number {
    let sum = 0;
    let total = 0;
    for (const s of this.reachable) {
      if (this.world.isTerminal(s)) continue;
      total += 1;
      sum += Math.abs(this.value(s) - (this.optimal.values[s] as number));
    }
    return total === 0 ? 0 : sum / total;
  }

  /** Mean return of the last `window` episodes. */
  recentReturn(window = 50): number {
    const recent = this.history.slice(-window);
    return recent.length === 0 ? 0 : recent.reduce((a, e) => a + e.ret, 0) / recent.length;
  }

  private build(): void {
    const c = this.cfg;
    const rewards: GridRewards = { step: c.stepReward, goal: 10, pit: -100 };
    this.world = new GridWorld(parseGrid(c.grid), rewards, c.slip);
    this.q = new Float64Array(this.world.states * ACTION_COUNT);
    this.visits = new Int32Array(this.world.states);
    this.optimal = valueIteration(this.world, c.gamma);
    this.rng = stream(c.seed, 0);
    this.agent = this.world.start;
    this.nextAction = null;
    this.epsilon = c.epsilon;
    this.episode = 0;
    this.totalSteps = 0;
    this.episodeReturn = 0;
    this.episodeSteps = 0;
    this.history = [];
    // flood fill from the start, not passing through terminals or walls
    this.reachable = new Set([this.world.start]);
    const queue = [this.world.start];
    while (queue.length) {
      const s = queue.pop() as number;
      if (this.world.isTerminal(s)) continue;
      for (let a = 0; a < ACTION_COUNT; a++) {
        const n = this.world.move(s, a);
        if (!this.reachable.has(n)) {
          this.reachable.add(n);
          queue.push(n);
        }
      }
    }
    this.key = worldKey(c);
  }
}
