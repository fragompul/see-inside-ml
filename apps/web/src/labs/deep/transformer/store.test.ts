import { readFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { createTransformer, type TokenizerData, type TransformerManifest } from '@sim/deep';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

// Load the real model from the deep package instead of fetching it over HTTP.
vi.mock('./assets', () => {
  const dir = resolve(
    dirname(fileURLToPath(import.meta.url)),
    '../../../../../../packages/deep/models/tiny-gpt',
  );
  const read = (name: string) => readFileSync(resolve(dir, name));
  const json = <T>(name: string) => JSON.parse(read(name).toString('utf-8')) as T;
  return {
    loadTinyGpt: async () => {
      const raw = read('weights.bin');
      const weights = raw.buffer.slice(
        raw.byteOffset,
        raw.byteOffset + raw.byteLength,
      ) as ArrayBuffer;
      return createTransformer(
        json<TransformerManifest>('model.json'),
        weights,
        json<TokenizerData>('tokenizer.json'),
      );
    },
  };
});

const { useTransformerStore, currentIds, rngFor, DEFAULT_PROMPT, DEFAULT_SAMPLING } =
  await import('./store');

const state = () => useTransformerStore.getState();

describe('transformer store', () => {
  beforeEach(async () => {
    state().stop();
    useTransformerStore.setState({
      status: 'idle',
      loaded: null,
      prompt: DEFAULT_PROMPT,
      generated: [],
      sampling: DEFAULT_SAMPLING,
      seed: 7,
      maxNew: 12,
      playing: false,
      selected: null,
    });
    await state().load();
  });

  afterEach(() => {
    state().stop();
    vi.useRealTimers();
  });

  it('loads the model and tokenizes the prompt', () => {
    expect(state().status).toBe('ready');
    const ids = currentIds(state());
    expect(ids.length).toBeGreaterThan(3);
    expect(state().loaded?.tokenizer.decode(ids)).toBe(DEFAULT_PROMPT);
  });

  it('reports a failed load without crashing', async () => {
    useTransformerStore.setState({ status: 'idle', loaded: null });
    await state().load(async () => {
      throw new Error('boom');
    });
    expect(state().status).toBe('error');
    expect(state().error).toBe('boom');
  });

  it('generateOne appends exactly one token and stays deterministic for a seed', () => {
    state().generateOne();
    const first = [...state().generated];
    expect(first).toHaveLength(1);

    state().clearGenerated();
    state().generateOne();
    expect(state().generated).toEqual(first);

    useTransformerStore.setState({ seed: 8 });
    state().clearGenerated();
    const outcomes = new Set<number>();
    for (const seed of [1, 2, 3, 4, 5, 6]) {
      useTransformerStore.setState({ seed });
      state().clearGenerated();
      state().generateOne();
      outcomes.add(state().generated[0] as number);
    }
    expect(outcomes.size).toBeGreaterThan(1); // different seeds can pick different tokens
  });

  it('stops at the token limit', () => {
    useTransformerStore.setState({ maxNew: 3 });
    expect(state().generateOne()).toBe(true);
    expect(state().generateOne()).toBe(true);
    expect(state().generateOne()).toBe(false); // third token reaches the cap
    expect(state().generated).toHaveLength(3);
    expect(state().generateOne()).toBe(false);
    expect(state().generated).toHaveLength(3);
  });

  it('greedy decoding is a deterministic continuation of the prompt', () => {
    useTransformerStore.setState({ sampling: { temperature: 0, topK: 0, topP: 1 }, maxNew: 6 });
    while (state().generateOne());
    const a = [...state().generated];
    state().clearGenerated();
    while (state().generateOne());
    expect(state().generated).toEqual(a);
  });

  it('editing the prompt discards the continuation and the focus', () => {
    state().generateOne();
    state().select(2);
    state().setPrompt('Tom found a key. He');
    expect(state().generated).toEqual([]);
    expect(state().selected).toBeNull();
    expect(state().playing).toBe(false);
  });

  it('lets the user choose a token by hand, and ignores invalid ids', () => {
    state().append(42);
    expect(state().generated).toEqual([42]);
    state().append(-1);
    state().append(99999);
    expect(state().generated).toEqual([42]);
  });

  it('auto-play generates on a timer and stops by itself', () => {
    vi.useFakeTimers();
    useTransformerStore.setState({ maxNew: 5 });
    state().togglePlay();
    expect(state().playing).toBe(true);
    vi.advanceTimersByTime(170 * 20);
    expect(state().playing).toBe(false);
    expect(state().generated.length).toBeGreaterThan(0);
    expect(state().generated.length).toBeLessThanOrEqual(5);
  });

  it('pausing auto-play keeps what was generated; playing again after the end starts fresh', () => {
    vi.useFakeTimers();
    useTransformerStore.setState({ maxNew: 4 });
    state().togglePlay();
    vi.advanceTimersByTime(170 * 2 + 10);
    state().togglePlay();
    expect(state().playing).toBe(false);
    const paused = state().generated.length;
    expect(paused).toBeGreaterThan(0);

    state().togglePlay();
    vi.advanceTimersByTime(170 * 20);
    expect(state().generated.length).toBeLessThanOrEqual(4);
    state().togglePlay(); // finished at the cap: starts over
    expect(state().generated.length).toBe(0);
  });

  it('each position draws from its own random stream', () => {
    expect(rngFor(1, 5).next()).toBe(rngFor(1, 5).next());
    expect(rngFor(1, 5).next()).not.toBe(rngFor(1, 6).next());
    expect(rngFor(1, 5).next()).not.toBe(rngFor(2, 5).next());
  });
});
