import { DEFAULT_CNN_LAB, IMAGE_SIZE } from '@sim/deep';
import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import { resetCnnSession, useCnnStore } from './store';

const state = () => useCnnStore.getState();

async function waitFor(condition: () => boolean, timeout = 8000) {
  const start = Date.now();
  while (!condition()) {
    if (Date.now() - start > timeout) throw new Error('waitFor timed out');
    await new Promise((resolve) => setTimeout(resolve, 10));
  }
}

/** A small, fast configuration so tests that train stay quick. */
const fast = { perClass: 10, batchSize: 10 } as const;

describe('cnn store', () => {
  beforeEach(() => {
    resetCnnSession();
    state().init();
  });
  afterEach(() => state().setRunning(false));

  it('starts with a sample digit already read by the network', () => {
    expect(state().source).toBe('sample');
    expect(state().sampleLabel).not.toBeNull();
    expect(state().trace?.probs).toHaveLength(10);
    expect(state().kernels?.conv1).toHaveLength(DEFAULT_CNN_LAB.filters1);
    expect(state().confusion).toHaveLength(10);
    expect(state().image).toHaveLength(IMAGE_SIZE * IMAGE_SIZE);
  });

  it('drawing paints a fresh picture; clear empties it', () => {
    state().beginStroke(4, 4);
    state().moveStroke(11, 11);
    state().endStroke();
    expect(state().source).toBe('drawn');
    expect(state().sampleLabel).toBeNull();
    const ink = state().image.reduce((a, b) => a + b, 0);
    expect(ink).toBeGreaterThan(10);
    expect(state().image[8 * IMAGE_SIZE + 8]).toBeGreaterThan(0.5); // the diagonal is continuous

    // moving without a stroke in progress does nothing
    const before = Float64Array.from(state().image);
    state().moveStroke(1, 1);
    expect(state().image).toEqual(before);

    state().clear();
    expect(state().source).toBe('blank');
    expect(state().image.every((v) => v === 0)).toBe(true);
  });

  it('starting to draw over a sample discards the sample', () => {
    state().loadSample(3);
    expect(state().source).toBe('sample');
    state().beginStroke(8, 8);
    state().endStroke();
    expect(state().source).toBe('drawn');
    expect(state().image.reduce((a, b) => a + b, 0)).toBeLessThan(12); // only the new dot
  });

  it('loads samples by index and at random', () => {
    state().loadSample(0);
    const first = Float64Array.from(state().image);
    state().loadSample(1);
    expect(state().image).not.toEqual(first);
    state().randomSample();
    expect(state().source).toBe('sample');
    expect(() => state().loadSample(99999)).toThrow();
  });

  it('trains one epoch on demand and reports it', () => {
    state().patch(fast);
    state().stepEpoch();
    expect(state().epoch).toBe(1);
    expect(state().history).toHaveLength(1);
    expect(state().running).toBe(false);
    state().confusion?.forEach((row) => expect(row.reduce((a, b) => a + b, 0)).toBe(30));
  });

  it('applies learning settings live and rebuilds on structure changes', () => {
    state().patch(fast);
    state().stepEpoch();
    const version = state().version;

    state().patch({ lr: 0.01 });
    expect(state().config.lr).toBe(0.01);
    expect(state().history).toHaveLength(1); // progress kept

    state().patch({ filters1: 4 });
    expect(state().history).toHaveLength(0); // restarted
    expect(state().kernels?.conv1).toHaveLength(4);
    expect(state().version).toBeGreaterThan(version);
  });

  it('rejects invalid settings with a notice and keeps the previous ones', () => {
    state().patch({ lr: -1 });
    expect(state().notice).toMatch(/Learning rate/);
    expect(state().config.lr).toBe(DEFAULT_CNN_LAB.lr);
    state().patch({ lr: 0.005 });
    expect(state().notice).toBeNull();
  });

  it('runs continuously, completes epochs, and stops when paused', async () => {
    state().patch(fast);
    state().setRunning(true);
    await waitFor(() => state().history.length >= 2);
    expect(state().running).toBe(true);
    state().setRunning(false);
    const settled = state().epoch;
    await new Promise((resolve) => setTimeout(resolve, 150));
    expect(state().epoch).toBe(settled);
    state().history.forEach((s, i) => expect(s.epoch).toBe(i + 1));
  });

  it('reset restarts training from the same initial weights', () => {
    state().patch(fast);
    state().stepEpoch();
    const first = state().history[0]?.trainLoss;
    state().reset();
    expect(state().history).toHaveLength(0);
    state().stepEpoch();
    expect(state().history[0]?.trainLoss).toBe(first);
  });

  it('centering the drawing changes what the network reads', () => {
    state().beginStroke(12, 12);
    state().moveStroke(14, 14);
    state().endStroke();
    state().setCenter(false);
    const raw = Float64Array.from(state().trace?.input ?? []);
    state().setCenter(true);
    const centred = Float64Array.from(state().trace?.input ?? []);
    expect(centred).not.toEqual(raw);
    expect(raw).toEqual(state().image); // without centering the network sees exactly the pad
  });

  it('computes the occlusion map only when asked and only for a real drawing', () => {
    expect(state().occlusion).toBeNull();
    state().setShowOcclusion(true);
    expect(state().occlusion).toHaveLength(IMAGE_SIZE * IMAGE_SIZE); // a sample has ink
    state().clear();
    expect(state().occlusion).toBeNull(); // blank pad: nothing to explain
    state().loadSample(2);
    expect(state().occlusion).not.toBeNull();
    state().setShowOcclusion(false);
    expect(state().occlusion).toBeNull();
  });
});
