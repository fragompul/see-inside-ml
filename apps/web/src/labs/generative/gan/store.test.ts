import { DEFAULT_GAN_LAB, EVAL_POINTS } from '@sim/deep';
import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import { D_GRID, WARP_GRID, resetGanSession, useGanStore } from './store';

const state = () => useGanStore.getState();

async function waitFor(condition: () => boolean, timeout = 8000) {
  const start = Date.now();
  while (!condition()) {
    if (Date.now() - start > timeout) throw new Error('waitFor timed out');
    await new Promise((resolve) => setTimeout(resolve, 10));
  }
}

const fast = { hidden: 12, batchSize: 32 } as const;

describe('gan store', () => {
  beforeEach(() => {
    resetGanSession();
    state().init();
  });
  afterEach(() => state().setRunning(false));

  it('starts with every view ready', () => {
    expect(state().real).toHaveLength(EVAL_POINTS * 2);
    expect(state().fake).toHaveLength(EVAL_POINTS * 2);
    expect(state().discriminator).toHaveLength(D_GRID * D_GRID);
    expect(state().warp).toHaveLength(WARP_GRID * WARP_GRID * 2);
    expect(state().iteration).toBe(0);
    expect(state().history).toHaveLength(0);
    expect(state().config.dataset).toBe(DEFAULT_GAN_LAB.dataset);
  });

  it('steps a number of iterations and pauses', () => {
    state().patch(fast);
    const before = Float64Array.from(state().fake ?? []);
    state().step(20);
    expect(state().iteration).toBe(20);
    expect(state().running).toBe(false);
    expect(state().history.length).toBe(2);
    expect(state().fake).not.toEqual(before);
  });

  it('applies learning settings live and rebuilds on structure or data changes', () => {
    state().patch(fast);
    state().step(10);
    state().patch({ lrD: 0.01, dSteps: 2, lossG: 'minimax' });
    expect(state().iteration).toBe(10);
    expect(state().history).toHaveLength(1);
    state().patch({ dataset: 'ring' });
    expect(state().iteration).toBe(0);
    expect(state().history).toHaveLength(0);
    expect(state().config.dataset).toBe('ring');
    state().patch({ noiseDim: 4 });
    expect(state().warp).toHaveLength(WARP_GRID * WARP_GRID * 2);
  });

  it('rejects invalid settings with a notice and keeps the previous ones', () => {
    state().patch({ dSteps: 0 });
    expect(state().notice).toMatch(/Discriminator steps/);
    expect(state().config.dSteps).toBe(DEFAULT_GAN_LAB.dSteps);
    state().patch({ dSteps: 2 });
    expect(state().notice).toBeNull();
  });

  it('runs continuously, records statistics, and stops when paused', async () => {
    state().patch(fast);
    state().setRunning(true);
    await waitFor(() => state().history.length >= 3);
    state().setRunning(false);
    const settled = state().iteration;
    await new Promise((resolve) => setTimeout(resolve, 150));
    expect(state().iteration).toBe(settled);
    state().history.forEach((h, i) => expect(h.iteration).toBe((i + 1) * 10));
  });

  it('reset restarts from the same weights', () => {
    state().patch(fast);
    const before = Float64Array.from(state().fake ?? []);
    state().step(30);
    state().reset();
    expect(state().iteration).toBe(0);
    expect(state().history).toHaveLength(0);
    expect(state().fake).toEqual(before);
  });
});
