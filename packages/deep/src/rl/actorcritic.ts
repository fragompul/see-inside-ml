import { createRng, type Rng } from '@sim/core';
import { ACTION_COUNT, type GridWorld } from './grid';
import { policySoftmax } from './policy';

/**
 * Actor-critic with generalised advantage estimation. The actor is a softmax policy over a table of
 * preferences; the critic is a table of state values. The advantage of a step is estimated from the critic as
 *     A_t = sum_k (gamma lambda)^k delta_(t+k),   delta_t = r_t + gamma V(s_(t+1)) - V(s_t)
 * lambda = 0 trusts the critic after one step (low variance, biased when the critic is wrong: this is A2C);
 * lambda = 1 trusts only the observed rewards (unbiased, high variance: REINFORCE with a baseline).
 */
export interface ActorCriticOptions {
  gamma: number;
  lambda: number;
  actorRate: number;
  criticRate: number;
  maxSteps: number;
  seed: number;
}

export const DEFAULT_ACTOR_CRITIC: ActorCriticOptions = {
  gamma: 0.97,
  lambda: 0.9,
  actorRate: 0.1,
  criticRate: 0.2,
  maxSteps: 80,
  seed: 1,
};

export interface AcEpisode {
  states: number[];
  actions: number[];
  rewards: number[];
  /** The state reached after the last step, and whether the episode really ended there (not cut off by the step limit). */
  last: number;
  terminal: boolean;
  outcome: 'goal' | 'pit' | 'timeout';
}

/**
 * Generalised advantage estimates for one episode. `values[t]` is V(s_t); `bootstrap` is V of the state after
 * the final step (0 when the episode ended for real).
 */
export function gaeAdvantages(
  rewards: readonly number[],
  values: readonly number[],
  bootstrap: number,
  gamma: number,
  lambda: number,
): number[] {
  const out = new Array<number>(rewards.length).fill(0);
  let running = 0;
  for (let t = rewards.length - 1; t >= 0; t--) {
    const next = t + 1 < rewards.length ? (values[t + 1] as number) : bootstrap;
    const delta = (rewards[t] as number) + gamma * next - (values[t] as number);
    running = delta + gamma * lambda * running;
    out[t] = running;
  }
  return out;
}

export class ActorCritic {
  readonly theta: Float64Array;
  readonly values: Float64Array;
  readonly rng: Rng;

  constructor(
    readonly world: GridWorld,
    readonly options: ActorCriticOptions,
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

  rollout(): AcEpisode {
    const states: number[] = [];
    const actions: number[] = [];
    const rewards: number[] = [];
    let s = this.world.start;
    let outcome: AcEpisode['outcome'] = 'timeout';
    let terminal = false;
    for (let t = 0; t < this.options.maxSteps; t++) {
      const a = this.choose(s);
      const step = this.world.step(s, a, this.rng);
      states.push(s);
      actions.push(a);
      rewards.push(step.reward);
      s = step.next;
      if (step.done) {
        outcome = this.world.kind(s) === 'goal' ? 'goal' : 'pit';
        terminal = true;
        break;
      }
    }
    return { states, actions, rewards, last: s, terminal, outcome };
  }

  advantages(ep: AcEpisode, lambda = this.options.lambda): number[] {
    return gaeAdvantages(
      ep.rewards,
      ep.states.map((s) => this.values[s] as number),
      ep.terminal ? 0 : (this.values[ep.last] as number),
      this.options.gamma,
      lambda,
    );
  }

  /** The policy-gradient estimate (sum over steps of advantage times grad log pi) for one episode. */
  gradient(ep: AcEpisode, advantages: readonly number[]): Float64Array {
    const grad = new Float64Array(this.theta.length);
    ep.states.forEach((s, t) => {
      const pi = this.probabilities(s);
      for (let a = 0; a < ACTION_COUNT; a++)
        grad[s * ACTION_COUNT + a] =
          (grad[s * ACTION_COUNT + a] as number) +
          (advantages[t] as number) *
            ((a === (ep.actions[t] as number) ? 1 : 0) - (pi[a] as number));
    });
    return grad;
  }

  /** One learning step from one episode; returns its undiscounted return. */
  learn(ep: AcEpisode): number {
    const adv = this.advantages(ep);
    const grad = this.gradient(ep, adv);
    for (let i = 0; i < this.theta.length; i++)
      this.theta[i] = (this.theta[i] as number) + this.options.actorRate * (grad[i] as number);
    // the critic moves towards the lambda-return, which is the advantage plus the old value
    ep.states.forEach((s, t) => {
      this.values[s] = (this.values[s] as number) + this.options.criticRate * (adv[t] as number);
    });
    return ep.rewards.reduce((a, b) => a + b, 0);
  }
}

export interface ActorCriticTraining {
  returns: number[];
  outcomes: AcEpisode['outcome'][];
  agent: ActorCritic;
}

export function trainActorCritic(
  world: GridWorld,
  options: ActorCriticOptions,
  episodes: number,
): ActorCriticTraining {
  const agent = new ActorCritic(world, options);
  const returns: number[] = [];
  const outcomes: AcEpisode['outcome'][] = [];
  for (let e = 0; e < episodes; e++) {
    const ep = agent.rollout();
    returns.push(agent.learn(ep));
    outcomes.push(ep.outcome);
  }
  return { returns, outcomes, agent };
}

export interface GradientSpread {
  mean: Float64Array;
  /** Sum over parameters of the variance of the single-episode estimate. */
  variance: number;
}

/**
 * Mean and variance of the single-episode gradient estimate at the agent's current policy and critic, for a
 * given lambda, from many fresh episodes. The policy and critic are not changed.
 */
export function gradientSpread(
  agent: ActorCritic,
  lambda: number,
  samples: number,
): GradientSpread {
  const n = agent.theta.length;
  const sum = new Float64Array(n);
  const sumSq = new Float64Array(n);
  for (let i = 0; i < samples; i++) {
    const ep = agent.rollout();
    const g = agent.gradient(ep, agent.advantages(ep, lambda));
    for (let k = 0; k < n; k++) {
      sum[k] = (sum[k] as number) + (g[k] as number);
      sumSq[k] = (sumSq[k] as number) + (g[k] as number) ** 2;
    }
  }
  const mean = sum.map((v) => v / samples);
  let variance = 0;
  for (let k = 0; k < n; k++) variance += (sumSq[k] as number) / samples - (mean[k] as number) ** 2;
  return { mean, variance };
}
