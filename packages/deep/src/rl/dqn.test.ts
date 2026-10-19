import { createRng } from '@sim/core';
import { describe, expect, it } from 'vitest';
import { DEFAULT_DQN, QNet, bellmanTargets, featuresOf, trainDqn } from './dqn';
import { GRID_PRESETS, GridWorld, parseGrid } from './grid';

const world = (id: string) =>
  new GridWorld(
    parseGrid((GRID_PRESETS[id] as { rows: string[] }).rows),
    { step: -0.04, goal: 1, pit: -1 },
    0,
  );

describe('Q-network', () => {
  it('backpropagation matches finite differences', () => {
    const net = new QNet(3, 5, createRng(2));
    const xs = [
      [0.3, -0.5, 0.9],
      [-1, 0.2, 0.4],
      [0.7, 0.7, -0.6],
    ];
    const actions = [0, 2, 3];
    const targets = [0.4, -0.2, 1.1];
    const { grads } = net.gradients(xs, actions, targets);
    const h = 1e-6;
    net.parameters.forEach((p, k) => {
      for (const i of [0, Math.floor(p.length / 2), p.length - 1]) {
        const saved = p[i]!;
        p[i] = saved + h;
        const up = net.gradients(xs, actions, targets).loss;
        p[i] = saved - h;
        const down = net.gradients(xs, actions, targets).loss;
        p[i] = saved;
        expect(grads[k]![i]).toBeCloseTo((up - down) / (2 * h), 6);
      }
    });
  });

  it('copyFrom makes two networks agree', () => {
    const a = new QNet(2, 4, createRng(1));
    const b = new QNet(2, 4, createRng(9));
    b.copyFrom(a);
    expect(Array.from(b.forward([0.2, 0.3]).q)).toEqual(Array.from(a.forward([0.2, 0.3]).q));
  });
});

describe('Bellman targets and features', () => {
  it('bootstraps from the best next value unless the episode ended', () => {
    const net = new QNet(2, 4, createRng(3));
    const next = [0.5, -0.5];
    const best = Math.max(...net.forward(next).q);
    const t = bellmanTargets(net, 0.9, [1, 2], [next, next], [false, true]);
    expect(t[0]).toBeCloseTo(1 + 0.9 * best, 12);
    expect(t[1]).toBe(2);
  });

  it('coordinates span -1 to 1 and one-hot has a single one', () => {
    const w = world('open');
    expect(Array.from(featuresOf(w, w.start, 'coordinates'))).toEqual([-1, -1]);
    const oh = featuresOf(w, 7, 'one-hot');
    expect(oh.reduce((a, b) => a + b, 0)).toBe(1);
    expect(oh[7]).toBe(1);
  });
});

describe('DQN training', () => {
  it('learns to reach the goal in the open room', () => {
    const run = trainDqn(world('open'), { ...DEFAULT_DQN, seed: 3 });
    expect(run.solved).toBe(true);
    const late = run.returns.slice(-20).reduce((a, b) => a + b, 0) / 20;
    const early = run.returns.slice(0, 20).reduce((a, b) => a + b, 0) / 20;
    expect(late).toBeGreaterThan(early);
  });

  it('is deterministic for a seed and records one entry per episode', () => {
    const o = { ...DEFAULT_DQN, episodes: 20, seed: 4 };
    const a = trainDqn(world('open'), o);
    const b = trainDqn(world('open'), o);
    expect(a.returns).toEqual(b.returns);
    expect(a.returns).toHaveLength(20);
    expect(a.maxQ).toHaveLength(20);
  });
});
