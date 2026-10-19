import { DEFAULT_DESCENT } from '@sim/deep';
import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import { LAND_GRID, currentAccuracy, resetFoundationsSession, useFoundationsStore } from './store';

const state = () => useFoundationsStore.getState();

async function waitFor(condition: () => boolean, timeout = 8000) {
  const start = Date.now();
  while (!condition()) {
    if (Date.now() - start > timeout) throw new Error('waitFor timed out');
    await new Promise((resolve) => setTimeout(resolve, 10));
  }
}

describe('foundations store', () => {
  beforeEach(() => {
    resetFoundationsSession();
    state().init();
  });
  afterEach(() => state().setRunning(false));

  it('starts with the landscape, the ball at its starting corner and a neuron to play with', () => {
    expect(state().landscape).toHaveLength(LAND_GRID * LAND_GRID);
    expect(state().path).toHaveLength(1);
    expect(state().path[0]).toMatchObject({ w1: -5, w2: 5 });
    expect(state().descent).toEqual(DEFAULT_DESCENT);
    expect(state().dataset).toBe('blobs');
    expect(state().gradientNorm).toBeGreaterThan(0);
  });

  it('the neuron weights can be set one by one and the accuracy follows', () => {
    state().setDataset('or');
    state().setNeuron({ w1: 1, w2: 1, b: -0.5 });
    expect(currentAccuracy(state().dataset, state().neuron)).toBe(1);
    state().setNeuron({ b: -5 });
    expect(currentAccuracy(state().dataset, state().neuron)).toBeLessThan(1);
    state().setActivation('relu');
    expect(state().activation).toBe('relu');
  });

  it('finding the best line reaches 100% on OR and only 75% on XOR', () => {
    state().setDataset('or');
    state().findBest();
    expect(state().best?.accuracy).toBe(1);
    expect(currentAccuracy('or', state().neuron)).toBe(1);
    state().setDataset('xor');
    expect(state().best).toBeNull(); // changing the data forgets the old answer
    state().findBest();
    expect(state().best?.accuracy).toBe(0.75);
    expect(currentAccuracy('xor', state().neuron)).toBe(0.75);
  });

  it('changing the dataset or the bias redraws the landscape and restarts the path', () => {
    state().stepOnce(5);
    expect(state().path).toHaveLength(6);
    const before = Float64Array.from(state().landscape ?? []);
    state().patchDescent({ bias: -1 });
    expect(state().landscape).not.toEqual(before);
    expect(state().path).toHaveLength(1);
    state().stepOnce(3);
    state().setDataset('xor');
    expect(state().dataset).toBe('xor');
    expect(state().descent.dataset).toBe('xor');
    expect(state().path).toHaveLength(1);
  });

  it('the learning rate and the optimiser can change while the ball is rolling', async () => {
    state().patchDescent({ lr: 0.5 });
    state().setRunning(true);
    await waitFor(() => state().path.length > 20);
    state().patchDescent({ optimizer: 'momentum', lr: 0.1 });
    state().setRunning(false);
    expect(state().descent.optimizer).toBe('momentum');
    expect(state().notice).toBeNull();
  });

  it('steps the ball downhill', () => {
    state().patchDescent({ lr: 0.5 });
    const first = state().path[0]!.loss;
    state().stepOnce(50);
    expect(state().path).toHaveLength(51);
    expect(state().path.at(-1)!.loss).toBeLessThan(first);
    expect(state().running).toBe(false);
  });

  it('placing the ball moves its start; reset returns to it', () => {
    state().place(3, -2);
    expect(state().path).toHaveLength(1);
    expect(state().path[0]).toMatchObject({ w1: 3, w2: -2 });
    state().stepOnce(10);
    state().reset();
    expect(state().path).toHaveLength(1);
    expect(state().path[0]).toMatchObject({ w1: 3, w2: -2 });
    state().place(99, 99); // clamped to the window
    expect(state().path[0]!.w1).toBe(8);
  });

  it('rejects invalid settings with a notice and keeps the previous ones', () => {
    state().patchDescent({ lr: 0 });
    expect(state().notice).toMatch(/Learning rate/);
    expect(state().descent.lr).toBe(DEFAULT_DESCENT.lr);
    state().patchDescent({ lr: 0.2 });
    expect(state().notice).toBeNull();
  });

  it('the learned weights can be copied into the neuron', () => {
    state().patchDescent({ lr: 0.5 });
    state().stepOnce(120);
    state().useLearned();
    expect(state().activation).toBe('sigmoid');
    expect(currentAccuracy('blobs', state().neuron)).toBeGreaterThanOrEqual(0.9);
  });

  it('runs continuously and stops when paused', async () => {
    state().setRunning(true);
    await waitFor(() => state().path.length > 30);
    state().setRunning(false);
    const settled = state().path.length;
    await new Promise((resolve) => setTimeout(resolve, 120));
    expect(state().path.length).toBe(settled);
  });
});
