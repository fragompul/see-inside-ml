import { DEFAULT_RNN_LAB, taskInfo } from '@sim/deep';
import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import { resetRnnSession, sweepLengths, useRnnStore } from './store';

const state = () => useRnnStore.getState();

async function waitFor(condition: () => boolean, timeout = 8000) {
  const start = Date.now();
  while (!condition()) {
    if (Date.now() - start > timeout) throw new Error('waitFor timed out');
    await new Promise((resolve) => setTimeout(resolve, 10));
  }
}

const fast = { trainSize: 64, batchSize: 16 } as const;

describe('rnn store', () => {
  beforeEach(() => {
    resetRnnSession();
    state().init();
  });
  afterEach(() => state().setRunning(false));

  it('starts with a test sequence already read by the network', () => {
    expect(state().tokens).toHaveLength(DEFAULT_RNN_LAB.seqLen);
    expect(state().source).toBe('sample');
    expect(state().label).not.toBeNull();
    expect(state().trace?.steps).toBe(DEFAULT_RNN_LAB.seqLen);
    expect(state().trace?.probs).toHaveLength(4);
    expect(state().gradientNorms).toHaveLength(DEFAULT_RNN_LAB.seqLen);
    expect(state().profile).toHaveLength(DEFAULT_RNN_LAB.seqLen);
    expect(state().confusion).toHaveLength(4);
  });

  it('cycling a token changes the sequence, its label and the trace', () => {
    const before = Int32Array.from(state().tokens);
    const labelBefore = state().label;
    state().cycleToken(0); // the task is "first", so the label follows token 0
    expect(state().source).toBe('edited');
    expect(state().tokens[0]).toBe(((before[0] as number) + 1) % 4);
    expect(state().label).not.toBe(labelBefore);
    state().cycleToken(99); // out of range: ignored
    expect(state().tokens).toHaveLength(before.length);
  });

  it('new sequences are valid and different', () => {
    const first = Int32Array.from(state().tokens);
    state().newSequence();
    expect(state().source).toBe('random');
    expect(state().tokens).not.toEqual(first);
    state().loadSample(1);
    expect(state().source).toBe('sample');
    expect(() => state().loadSample(99999)).toThrow();
  });

  it('breaking the marker rule leaves a trace but no label', () => {
    state().patch({ task: 'marker', seqLen: 6 });
    expect(state().label).not.toBeNull();
    // remove the star by cycling it to a normal symbol
    const star = Array.from(state().tokens).indexOf(4);
    state().cycleToken(star);
    expect(state().label).toBeNull();
    expect(state().gradientNorms).toBeNull();
    expect(state().trace).not.toBeNull();
  });

  it('trains an epoch on demand', () => {
    state().patch(fast);
    state().stepEpoch();
    expect(state().epoch).toBe(1);
    expect(state().history).toHaveLength(1);
    expect(state().running).toBe(false);
    const total = state()
      .confusion?.flat()
      .reduce((a, b) => a + b, 0);
    expect(total).toBe(200);
  });

  it('applies learning settings live and rebuilds on structure changes', () => {
    state().patch(fast);
    state().stepEpoch();
    state().patch({ lr: 0.02 });
    expect(state().history).toHaveLength(1);
    state().patch({ cell: 'gru' });
    expect(state().history).toHaveLength(0);
    expect(state().config.cell).toBe('gru');
  });

  it('changing the length or the task draws a sequence of the right size', () => {
    state().patch({ seqLen: 14 });
    expect(state().tokens).toHaveLength(14);
    state().patch({ task: 'parity' });
    expect(state().tokens.every((t) => t < 2)).toBe(true);
    expect(state().trace?.probs).toHaveLength(taskInfo.parity.classes);
  });

  it('rejects invalid settings with a notice', () => {
    state().patch({ hidden: 1 });
    expect(state().notice).toMatch(/Hidden size/);
    expect(state().config.hidden).toBe(DEFAULT_RNN_LAB.hidden);
    state().patch({ hidden: 12 });
    expect(state().notice).toBeNull();
  });

  it('runs continuously, completes epochs, and stops when paused', async () => {
    state().patch(fast);
    state().setRunning(true);
    await waitFor(() => state().history.length >= 2);
    state().setRunning(false);
    const settled = state().epoch;
    await new Promise((resolve) => setTimeout(resolve, 150));
    expect(state().epoch).toBe(settled);
    state().history.forEach((s, i) => expect(s.epoch).toBe(i + 1));
  });

  it('reset restarts from the same initial weights', () => {
    state().patch(fast);
    state().stepEpoch();
    const first = state().history[0]?.trainLoss;
    state().reset();
    expect(state().history).toHaveLength(0);
    state().stepEpoch();
    expect(state().history[0]?.trainLoss).toBe(first);
  });

  it('measures accuracy at several lengths and forgets it when the network changes', () => {
    state().patch(fast);
    state().measureLengths();
    const sweep = state().lengthSweep;
    expect(sweep?.length).toBeGreaterThan(3);
    sweep?.forEach((p) => expect(p.accuracy).toBeGreaterThanOrEqual(0));
    state().patch({ hidden: 8 });
    expect(state().lengthSweep).toBeNull();
  });

  it('sweep lengths stay inside what the task allows, include the trained one, and are sorted', () => {
    const lengths = sweepLengths('parity', 10);
    expect(lengths).toContain(10);
    expect(Math.max(...lengths)).toBeLessThanOrEqual(taskInfo.parity.maxLength);
    expect(Math.min(...lengths)).toBeGreaterThanOrEqual(taskInfo.parity.minLength);
    expect([...lengths].sort((a, b) => a - b)).toEqual(lengths);
  });
});
