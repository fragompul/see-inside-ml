import { DEFAULT_NLP_LAB, VOCABULARY, WORLD } from '@sim/deep';
import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import { resetNlpSession, useNlpStore } from './store';

const state = () => useNlpStore.getState();
const CONTENT = Object.entries(WORLD)
  .filter(([c]) => c !== 'function')
  .flatMap(([, w]) => w);

async function waitFor(condition: () => boolean, timeout = 8000) {
  const start = Date.now();
  while (!condition()) {
    if (Date.now() - start > timeout) throw new Error('waitFor timed out');
    await new Promise((resolve) => setTimeout(resolve, 10));
  }
}

const fast = { sentences: 300 } as const;

describe('nlp store', () => {
  beforeEach(() => {
    resetNlpSession();
    state().init();
  });
  afterEach(() => state().setRunning(false));

  it('starts with the map, neighbours, an analogy and the accuracy groups ready', () => {
    expect(state().map?.words).toHaveLength(CONTENT.length);
    expect(state().map?.coords).toHaveLength(CONTENT.length * 2);
    expect(state().neighbours).toHaveLength(8);
    expect(state().neighbours.map((n) => n.word)).not.toContain(state().selected);
    expect(state().answers).toHaveLength(3);
    expect(state().groups.map((g) => g.id)).toEqual(['gender', 'capital', 'royalty']);
    expect(state().corpus().length).toBe(DEFAULT_NLP_LAB.sentences);
    expect(state().pairCount()).toBeGreaterThan(10_000);
    expect(VOCABULARY).toContain(state().selected);
  });

  it('selecting a word and changing the analogy updates the answers', () => {
    state().select('paris');
    expect(state().selected).toBe('paris');
    expect(state().neighbours.map((n) => n.word)).not.toContain('paris');
    state().setQuery({ a: 'france', b: 'paris', c: 'italy' });
    expect(state().query).toEqual({ a: 'france', b: 'paris', c: 'italy' });
    expect(state().answers.map((n) => n.word)).not.toContain('italy');
    expect(() => state().select('banana')).toThrow();
  });

  it('browsing sentences wraps around', () => {
    const n = state().corpus().length;
    state().setSentence(-1);
    expect(state().sentence).toBe(n - 1);
    state().setSentence(n);
    expect(state().sentence).toBe(0);
  });

  it('steps training and publishes the statistics', () => {
    state().patch(fast);
    state().stepOnce(50);
    expect(state().step).toBe(50);
    expect(state().history).toHaveLength(2);
    expect(state().running).toBe(false);
  });

  it('applies live settings and rebuilds on shape or data changes', () => {
    state().patch(fast);
    state().stepOnce(25);
    state().patch({ lr: 0.02, negatives: 3 });
    expect(state().history).toHaveLength(1);
    state().patch({ dim: 8 });
    expect(state().history).toHaveLength(0);
    expect(state().step).toBe(0);
    state().patch({ window: 1 });
    expect(state().config.window).toBe(1);
  });

  it('rejects invalid settings with a notice', () => {
    state().patch({ dim: 1 });
    expect(state().notice).toMatch(/Dimension/);
    expect(state().config.dim).toBe(DEFAULT_NLP_LAB.dim);
    state().patch({ dim: 12 });
    expect(state().notice).toBeNull();
  });

  it('runs continuously and stops when paused', async () => {
    state().patch(fast);
    state().setRunning(true);
    await waitFor(() => state().history.length >= 3);
    state().setRunning(false);
    const settled = state().step;
    await new Promise((resolve) => setTimeout(resolve, 150));
    expect(state().step).toBe(settled);
  });

  it('a learning rate that is far too high stops training with an explanation, and reset recovers', async () => {
    state().patch({ ...fast, lr: 0.5 });
    state().setRunning(true);
    await waitFor(() => state().diverged);
    expect(state().running).toBe(false);
    expect(state().notice).toMatch(/learning rate is too high/);
    state().patch({ lr: 0.05 });
    state().reset();
    expect(state().diverged).toBe(false);
    expect(state().notice).toBeNull();
  });

  it('reset restarts from the same vectors', () => {
    state().patch(fast);
    const before = Float64Array.from(state().map?.coords ?? []);
    state().stepOnce(50);
    state().reset();
    expect(state().step).toBe(0);
    expect(state().map?.coords).toEqual(before);
  });
});
