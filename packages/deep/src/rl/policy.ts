import { createRng, type Rng } from '@sim/core';
import { ACTION_COUNT, type GridWorld } from './grid';

/**
 * REINFORCE: the simplest policy-gradient method. The policy is a policySoftmax over a table of preferences
 * theta[s][a]. After each episode every action taken is made more likely in proportion to the return
 * that followed it, minus a baseline that does not change the expected update but cuts its variance:
 *     theta[s][a] += alpha * (G_t - b(s_t)) * (1[a = a_t] - pi(a | s_t))
 */
export type Baseline = 'none' | 'average' | 'value';

export interface PolicyOptions {
  gamma: number;
  learningRate: number;
  baseline: Baseline;
  /** Weight of an entropy bonus that keeps the policy from becoming deterministic too early. */
  entropy: number;
  /** Episodes collected before each update. */
  batch: number;
  maxSteps: number;
  seed: number;
}

export const DEFAULT_POLICY: PolicyOptions = {
  gamma: 0.97,
  learningRate: 0.1,
  baseline: 'value',
  entropy: 0,
  batch: 1,
  maxSteps: 120,
  seed: 1,
};

export interface Trajectory {
  states: number[];
  actions: number[];
  rewards: number[];
  /** Why it ended. */
  outcome: 'goal' | 'pit' | 'timeout';
}

export function policySoftmax(logits: ArrayLike<number>): number[] {
  const m = Math.max(...Array.from(logits));
  const e = Array.from(logits, (v) => Math.exp(v - m));
  const z = e.reduce((a, b) => a + b, 0);
  return e.map((v) => v / z);
}

/** Discounted returns G_t = r_t + gamma * G_(t+1). */
export function discountedReturns(rewards: readonly number[], gamma: number): number[] {
  const out = new Array<number>(rewards.length).fill(0);
  let g = 0;
  for (let t = rewards.length - 1; t >= 0; t--) {
    g = (rewards[t] as number) + gamma * g;
    out[t] = g;
  }
  return out;
}

/**
 * The REINFORCE gradient estimate of one trajectory: for every step, (G_t - baseline) times the
 * gradient of log pi(a_t | s_t) with respect to theta, accumulated into a sparse map.
 */
export function reinforceGradient(
  theta: Float64Array,
  trajectory: Pick<Trajectory, 'states' | 'actions' | 'rewards'>,
  gamma: number,
  baseline: (state: number, t: number, g: number) => number,
): Float64Array {
  const grad = new Float64Array(theta.length);
  const returns = discountedReturns(trajectory.rewards, gamma);
  trajectory.states.forEach((s, t) => {
    const logits = Array.from(
      { length: ACTION_COUNT },
      (_, a) => theta[s * ACTION_COUNT + a] as number,
    );
    const pi = policySoftmax(logits);
    const advantage = (returns[t] as number) - baseline(s, t, returns[t] as number);
    for (let a = 0; a < ACTION_COUNT; a++)
      grad[s * ACTION_COUNT + a] =
        (grad[s * ACTION_COUNT + a] as number) +
        advantage * ((a === (trajectory.actions[t] as number) ? 1 : 0) - (pi[a] as number));
  });
  return grad;
}

export class ReinforceAgent {
  readonly theta: Float64Array;
  readonly values: Float64Array;
  private average = 0;
  private updates = 0;
  readonly rng: Rng;

  constructor(
    readonly world: GridWorld,
    readonly options: PolicyOptions,
  ) {
    this.theta = new Float64Array(world.states * ACTION_COUNT);
    this.values = new Float64Array(world.states);
    this.rng = createRng(options.seed);
  }

  probabilities(state: number): number[] {
    return policySoftmax(
      Array.from(
        { length: ACTION_COUNT },
        (_, a) => this.theta[state * ACTION_COUNT + a] as number,
      ),
    );
  }

  private choose(state: number): number {
    const p = this.probabilities(state);
    let u = this.rng.next();
    for (let a = 0; a < ACTION_COUNT; a++) {
      u -= p[a] as number;
      if (u <= 0) return a;
    }
    return ACTION_COUNT - 1;
  }

  rollout(): Trajectory {
    const states: number[] = [];
    const actions: number[] = [];
    const rewards: number[] = [];
    let s = this.world.start;
    let outcome: Trajectory['outcome'] = 'timeout';
    for (let t = 0; t < this.options.maxSteps; t++) {
      const a = this.choose(s);
      const step = this.world.step(s, a, this.rng);
      states.push(s);
      actions.push(a);
      rewards.push(step.reward);
      s = step.next;
      if (step.done) {
        outcome = this.world.kind(s) === 'goal' ? 'goal' : 'pit';
        break;
      }
    }
    return { states, actions, rewards, outcome };
  }

  /** Updates the policy from a batch of trajectories; returns the undiscounted return of each. */
  update(batch: readonly Trajectory[]): number[] {
    const { gamma, learningRate, baseline, entropy } = this.options;
    const grad = new Float64Array(this.theta.length);
    const totals: number[] = [];
    for (const tr of batch) {
      totals.push(tr.rewards.reduce((a, b) => a + b, 0));
      const returns = discountedReturns(tr.rewards, gamma);
      const baseFn =
        baseline === 'none'
          ? () => 0
          : baseline === 'average'
            ? () => this.average
            : (s: number) => this.values[s] as number;
      const g = reinforceGradient(this.theta, tr, gamma, baseFn);
      for (let i = 0; i < grad.length; i++)
        grad[i] = (grad[i] as number) + (g[i] as number) / batch.length;
      if (entropy > 0) {
        tr.states.forEach((s) => {
          const p = this.probabilities(s);
          const h = -p.reduce((a, v) => a + v * Math.log(v + 1e-12), 0);
          for (let a = 0; a < ACTION_COUNT; a++)
            grad[s * ACTION_COUNT + a] =
              (grad[s * ACTION_COUNT + a] as number) +
              (entropy * (p[a] as number) * (-Math.log((p[a] as number) + 1e-12) - h)) /
                batch.length;
        });
      }
      // learn the baselines from the same data
      tr.states.forEach((s, t) => {
        this.values[s] =
          (this.values[s] as number) + 0.1 * ((returns[t] as number) - (this.values[s] as number));
      });
      this.updates += 1;
      this.average += ((returns[0] as number) - this.average) / Math.min(this.updates, 50);
    }
    for (let i = 0; i < this.theta.length; i++)
      this.theta[i] = (this.theta[i] as number) + learningRate * (grad[i] as number);
    return totals;
  }

  /** The most probable action in every state. */
  greedyPolicy(): number[] {
    return Array.from({ length: this.world.states }, (_, s) => {
      const p = this.probabilities(s);
      return p.indexOf(Math.max(...p));
    });
  }
}

export interface PolicyTraining {
  /** Undiscounted return of every training episode. */
  returns: number[];
  outcomes: Trajectory['outcome'][];
  agent: ReinforceAgent;
}

export function trainReinforce(
  world: GridWorld,
  options: PolicyOptions,
  episodes: number,
): PolicyTraining {
  const agent = new ReinforceAgent(world, options);
  const returns: number[] = [];
  const outcomes: Trajectory['outcome'][] = [];
  let done = 0;
  while (done < episodes) {
    const batch: Trajectory[] = [];
    for (let b = 0; b < options.batch && done + b < episodes; b++) batch.push(agent.rollout());
    returns.push(...agent.update(batch));
    outcomes.push(...batch.map((t) => t.outcome));
    done += batch.length;
  }
  return { returns, outcomes, agent };
}
