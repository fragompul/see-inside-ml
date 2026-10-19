import { describe, expect, it } from 'vitest';
import { cacheBytes, decodeText, generationMacs, makeDecoder } from './kvcache';

describe('key/value cache', () => {
  const m = makeDecoder(16, 4, 20, 3);
  const prompt = [1, 5, 9, 3, 7, 2];

  it('gives exactly the same text and scores with or without the cache', () => {
    const a = decodeText(m, prompt, 24, true);
    const b = decodeText(m, prompt, 24, false);
    expect(a.tokens).toEqual(b.tokens);
    for (let i = 0; i < a.lastLogits.length; i++)
      expect(a.lastLogits[i]).toBeCloseTo(b.lastLogits[i]!, 10);
  });

  it('the counted multiply-adds match the closed-form cost model', () => {
    for (const cached of [true, false]) {
      const run = decodeText(m, prompt, 20, cached);
      expect(run.macs).toEqual(generationMacs(m.shape, prompt.length, 20, cached));
    }
  });

  it('the cache saves work, and the saving grows with the length of the text', () => {
    const total = (steps: number, cached: boolean) => {
      const t = generationMacs(m.shape, 6, steps, cached);
      return t.projection + t.attention;
    };
    const speedup = (steps: number) => total(steps, false) / total(steps, true);
    expect(speedup(10)).toBeGreaterThan(1);
    expect(speedup(100)).toBeGreaterThan(speedup(10));
  });

  it('cache memory is two tensors per layer and grows linearly with the context', () => {
    expect(cacheBytes(32, 4096, 32768, 2)).toBe(2 * 32 * 4096 * 32768 * 2);
    expect(cacheBytes(2, 8, 20, 4)).toBe(2 * cacheBytes(1, 8, 20, 4));
    expect(cacheBytes(1, 8, 40, 4)).toBe(2 * cacheBytes(1, 8, 20, 4));
  });
});
