import { createRng } from '@sim/core';
import { describe, expect, it } from 'vitest';
import { ACTION_COUNT, GridWorld, parseGrid } from './grid';
import type { ReinforceAgent } from './policy';
import {
  DEFAULT_POLICY,
  discountedReturns,
  reinforceGradient,
  policySoftmax,
  trainReinforce,
} from './policy';

describe('policySoftmax and returns', () => {
  it('policySoftmax is a distribution and the gradient of log pi is one-hot minus pi', () => {
    const logits = [0.3, -1.2, 0.8, 0.1];
    const p = policySoftmax(logits);
    expect(p.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 12);
    for (let k = 0; k < 4; k++) {
      const h = 1e-6;
      for (let a = 0; a < 4; a++) {
        const up = [...logits];
        const down = [...logits];
        up[k] = up[k]! + h;
        down[k] = down[k]! - h;
        const numeric =
          (Math.log(policySoftmax(up)[a]!) - Math.log(policySoftmax(down)[a]!)) / (2 * h);
        expect(numeric).toBeCloseTo((a === k ? 1 : 0) - p[k]!, 6);
      }
    }
  });

  it('discounted returns follow G_t = r_t + gamma G_(t+1)', () => {
    expect(discountedReturns([1, 2, 3], 0.5)).toEqual([1 + 0.5 * (2 + 0.5 * 3), 2 + 0.5 * 3, 3]);
    expect(discountedReturns([], 0.9)).toEqual([]);
  });
});

describe('the REINFORCE estimator', () => {
  // a one-state problem: four actions with rewards 0, 1, 2, 5, every episode lasts one step
  const rewards = [0, 1, 2, 5];
  const theta = new Float64Array([0.2, -0.3, 0.5, 0.1]);
  const pi = policySoftmax(Array.from(theta));
  const analytic = rewards.map((_, k) =>
    pi.reduce((s, p, a) => s + p * rewards[a]! * ((a === k ? 1 : 0) - pi[k]!), 0),
  );

  const sample = (n: number, baseline: number, seed: number) => {
    const rng = createRng(seed);
    const grads: number[][] = [];
    for (let i = 0; i < n; i++) {
      let u = rng.next();
      let a = 3;
      for (let j = 0; j < 4; j++) {
        u -= pi[j]!;
        if (u <= 0) {
          a = j;
          break;
        }
      }
      const g = reinforceGradient(
        theta,
        { states: [0], actions: [a], rewards: [rewards[a]!] },
        0.9,
        () => baseline,
      );
      grads.push(Array.from(g));
    }
    const mean = [0, 1, 2, 3].map((k) => grads.reduce((s, g) => s + g[k]!, 0) / n);
    const variance = [0, 1, 2, 3].map(
      (k) => grads.reduce((s, g) => s + (g[k]! - mean[k]!) ** 2, 0) / n,
    );
    return { mean, variance };
  };

  it('is an unbiased estimate of the true policy gradient', () => {
    const { mean } = sample(200000, 0, 1);
    mean.forEach((m, k) => expect(m).toBeCloseTo(analytic[k]!, 1));
  });

  it('a baseline does not bias the estimate but reduces its variance', () => {
    const plain = sample(200000, 0, 2);
    const based = sample(200000, 2.3, 2);
    based.mean.forEach((m, k) => expect(m).toBeCloseTo(analytic[k]!, 1));
    expect(based.variance.reduce((a, b) => a + b, 0)).toBeLessThan(
      plain.variance.reduce((a, b) => a + b, 0),
    );
  });
});

describe('learning', () => {
  const world = new GridWorld(
    parseGrid(['S..', '...', '..G']),
    { step: -1, goal: 10, pit: -100 },
    0,
  );

  it('learns to reach the goal in an open room, close to the shortest path', () => {
    const { returns } = trainReinforce(
      world,
      { ...DEFAULT_POLICY, learningRate: 0.2, seed: 3 },
      600,
    );
    const tail = returns.slice(-50);
    const mean = tail.reduce((a, b) => a + b, 0) / tail.length;
    // the shortest path has 4 steps: return 10 - 3 = 7 undiscounted
    expect(mean).toBeGreaterThan(5);
    expect(returns.slice(0, 20).reduce((a, b) => a + b, 0) / 20).toBeLessThan(mean);
  });

  it('a value baseline learns faster than none on average', () => {
    const avg = (baseline: 'none' | 'value') => {
      let s = 0;
      for (let seed = 1; seed <= 6; seed++) {
        const { returns } = trainReinforce(
          world,
          { ...DEFAULT_POLICY, baseline, learningRate: 0.1, seed },
          300,
        );
        s += returns.slice(100, 300).reduce((a, b) => a + b, 0) / 200;
      }
      return s / 6;
    };
    expect(avg('value')).toBeGreaterThan(avg('none'));
  });

  it('the policy stays a distribution and the greedy policy points towards the goal from the start', () => {
    const { agent } = trainReinforce(world, { ...DEFAULT_POLICY, learningRate: 0.2, seed: 5 }, 600);
    for (let s = 0; s < world.states; s++)
      expect(agent.probabilities(s).reduce((a, b) => a + b, 0)).toBeCloseTo(1, 12);
    expect([1, 2]).toContain(agent.greedyPolicy()[world.start]);
  });

  it('an entropy bonus keeps the policy more random', () => {
    const entropyOf = (agent: ReinforceAgent) => {
      let h = 0;
      for (let s = 0; s < world.states; s++) {
        const p = agent.probabilities(s);
        h += -p.reduce((a, v) => a + v * Math.log(v + 1e-12), 0);
      }
      return h;
    };
    const plain = trainReinforce(
      world,
      { ...DEFAULT_POLICY, learningRate: 0.2, seed: 7 },
      500,
    ).agent;
    const bonus = trainReinforce(
      world,
      { ...DEFAULT_POLICY, learningRate: 0.2, seed: 7, entropy: 1 },
      500,
    ).agent;
    expect(entropyOf(bonus)).toBeGreaterThan(entropyOf(plain));
    expect(ACTION_COUNT).toBe(4);
  });

  it('is deterministic for a seed', () => {
    const a = trainReinforce(world, { ...DEFAULT_POLICY, seed: 9 }, 50).returns;
    const b = trainReinforce(world, { ...DEFAULT_POLICY, seed: 9 }, 50).returns;
    expect(a).toEqual(b);
  });
});
