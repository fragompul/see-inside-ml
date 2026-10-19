import { DEFAULT_DIFFUSION_LAB, DIFFUSION_EVAL_POINTS } from '@sim/deep';
import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import { FIELD_GRID, resetDiffusionSession, useDiffusionStore } from './store';

const state = () => useDiffusionStore.getState();

async function waitFor(condition: () => boolean, timeout = 8000) {
  const start = Date.now();
  while (!condition()) {
    if (Date.now() - start > timeout) throw new Error('waitFor timed out');
    await new Promise((resolve) => setTimeout(resolve, 10));
  }
}

const fast = { hidden: 16, batchSize: 32, steps: 20, sampleSteps: 5 } as const;

describe('diffusion store', () => {
  beforeEach(() => {
    resetDiffusionSession();
    state().init();
  });
  afterEach(() => state().setRunning(false));

  it('starts with every view ready', () => {
    expect(state().reference).toHaveLength(DIFFUSION_EVAL_POINTS * 2);
    expect(state().noised).toHaveLength(DIFFUSION_EVAL_POINTS * 2);
    expect(state().field?.score).toHaveLength(FIELD_GRID * FIELD_GRID * 2);
    expect(state().trajectory?.points).toBe(DIFFUSION_EVAL_POINTS);
    expect(state().trajectory?.times[0]).toBe(DEFAULT_DIFFUSION_LAB.steps);
    expect(state().trajectory?.times.at(-1)).toBe(0);
    expect(state().time).toBe(DEFAULT_DIFFUSION_LAB.steps / 2);
  });

  it('moving the time slider re-noises the data (and clamps to 0…T)', () => {
    const half = Float64Array.from(state().noised ?? []);
    state().setTime(0);
    expect(state().noised).toEqual(state().reference);
    state().setTime(state().config.steps);
    expect(state().noised).not.toEqual(half);
    state().setTime(999);
    expect(state().time).toBe(state().config.steps);
    state().setTime(-5);
    expect(state().time).toBe(0);
  });

  it('steps training and publishes the loss', () => {
    state().patch(fast);
    state().step(20);
    expect(state().iteration).toBe(20);
    expect(state().running).toBe(false);
    expect(state().history).toHaveLength(2);
  });

  it('applies live settings and rebuilds on structure or data changes, keeping time in range', () => {
    state().patch(fast);
    state().step(10);
    state().patch({ lr: 0.005, sampler: 'ddpm' });
    expect(state().history).toHaveLength(1);
    expect(state().trajectory?.times).toHaveLength(21); // DDPM visits every one of the 20 steps
    state().patch({ steps: 40 });
    expect(state().iteration).toBe(0);
    expect(state().time).toBe(20);
    state().patch({ dataset: 'ring' });
    expect(state().config.dataset).toBe('ring');
    state().patch({ steps: 10 });
    expect(state().time).toBeLessThanOrEqual(10);
  });

  it('rejects invalid settings with a notice and keeps the previous ones', () => {
    state().patch({ steps: 3 });
    expect(state().notice).toMatch(/Diffusion steps/);
    expect(state().config.steps).toBe(DEFAULT_DIFFUSION_LAB.steps);
    state().patch({ steps: 30 });
    expect(state().notice).toBeNull();
  });

  it('resample recomputes the path with the current sampler settings', () => {
    state().patch(fast);
    state().patch({ sampleSteps: 2 });
    expect(state().trajectory?.times).toHaveLength(3);
    state().patch({ sampleSteps: 10 });
    state().resample();
    expect(state().trajectory?.times).toHaveLength(11);
  });

  it('runs continuously, records the loss, and stops when paused', async () => {
    state().patch(fast);
    state().setRunning(true);
    await waitFor(() => state().history.length >= 3);
    state().setRunning(false);
    const settled = state().iteration;
    await new Promise((resolve) => setTimeout(resolve, 150));
    expect(state().iteration).toBe(settled);
  });

  it('reset restarts from the same weights', () => {
    state().patch(fast);
    const before = Float64Array.from(state().trajectory?.data ?? []);
    state().step(20);
    state().reset();
    expect(state().iteration).toBe(0);
    expect(state().history).toHaveLength(0);
    expect(state().trajectory?.data).toEqual(before);
  });
});
