import { describe, expect, it } from 'vitest';
import {
  ActorCritic,
  DEFAULT_ACTOR_CRITIC,
  gaeAdvantages,
  gradientSpread,
  trainActorCritic,
} from './actorcritic';
import { GRID_PRESETS, GridWorld, parseGrid } from './grid';
import { discountedReturns } from './policy';

const world = new GridWorld(
  parseGrid((GRID_PRESETS.open as { rows: string[] }).rows),
  { step: -0.02, goal: 1, pit: -1 },
  0,
);

describe('generalised advantage estimation', () => {
  const rewards = [0.5, -0.2, 1, 0.3];
  const values = [0.1, 0.4, -0.3, 0.2];
  const gamma = 0.9;

  it('lambda = 0 is the one-step TD error', () => {
    const a = gaeAdvantages(rewards, values, 0.7, gamma, 0);
    expect(a[0]).toBeCloseTo(0.5 + gamma * 0.4 - 0.1, 12);
    expect(a[3]).toBeCloseTo(0.3 + gamma * 0.7 - 0.2, 12);
  });

  it('lambda = 1 is the discounted return (plus the bootstrap) minus the value', () => {
    const bootstrap = 0.7;
    const a = gaeAdvantages(rewards, values, bootstrap, gamma, 1);
    const returns = discountedReturns(
      [...rewards.slice(0, 3), rewards[3]! + gamma * bootstrap],
      gamma,
    );
    for (let t = 0; t < rewards.length; t++) expect(a[t]).toBeCloseTo(returns[t]! - values[t]!, 12);
  });

  it('with the true values every advantage is zero, whatever lambda is', () => {
    // a deterministic chain: values satisfy V_t = r_t + gamma V_(t+1)
    const r = [1, 0, 2];
    const v = [0, 0, r[2]!];
    v[1] = r[1]! + gamma * v[2]!;
    v[0] = r[0]! + gamma * v[1]!;
    for (const lambda of [0, 0.5, 1])
      for (const x of gaeAdvantages(r, v, 0, gamma, lambda)) expect(x).toBeCloseTo(0, 12);
  });
});

describe('actor-critic agent', () => {
  it('learns to reach the goal in the open room', () => {
    const t = trainActorCritic(world, { ...DEFAULT_ACTOR_CRITIC, seed: 2 }, 300);
    const last = t.outcomes.slice(-50);
    expect(last.filter((o) => o === 'goal').length).toBeGreaterThan(40);
  });

  it('is deterministic for a seed', () => {
    const a = trainActorCritic(world, { ...DEFAULT_ACTOR_CRITIC, seed: 5 }, 30).returns;
    const b = trainActorCritic(world, { ...DEFAULT_ACTOR_CRITIC, seed: 5 }, 30).returns;
    expect(a).toEqual(b);
  });

  it('measuring the gradient spread leaves the agent unchanged', () => {
    const agent = new ActorCritic(world, DEFAULT_ACTOR_CRITIC);
    const before = Array.from(agent.theta);
    gradientSpread(agent, 0.5, 20);
    expect(Array.from(agent.theta)).toEqual(before);
  });

  it('a lower lambda gives a lower-variance gradient estimate than lambda = 1', () => {
    const t = trainActorCritic(world, { ...DEFAULT_ACTOR_CRITIC, seed: 3 }, 80);
    const low = gradientSpread(t.agent, 0, 400).variance;
    const high = gradientSpread(t.agent, 1, 400).variance;
    expect(low).toBeLessThan(high);
  });
});
