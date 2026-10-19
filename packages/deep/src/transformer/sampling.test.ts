import { describe, expect, it } from 'vitest';
import { createRng } from '@sim/core';
import {
  nextTokenDistribution,
  sampleToken,
  softmax,
  validateSampling,
  type SamplingOptions,
} from './sampling';

const logits = [2, 1, 0.5, 0, -1, -3];
const opts = (patch: Partial<SamplingOptions> = {}): SamplingOptions => ({
  temperature: 1,
  topK: 0,
  topP: 1,
  ...patch,
});

describe('softmax', () => {
  it('sums to 1 and preserves the ordering of the logits', () => {
    const p = softmax(logits);
    expect(p.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 12);
    for (let i = 1; i < p.length; i++) expect(p[i - 1]).toBeGreaterThan(p[i] as number);
  });

  it('is stable for huge logits', () => {
    const p = softmax([1000, 999, 998]);
    expect(p.every(Number.isFinite)).toBe(true);
    expect(p[0] as number).toBeGreaterThan(p[1] as number);
  });

  it('temperature sharpens (low) or flattens (high) the distribution', () => {
    const cold = softmax(logits, 0.2);
    const base = softmax(logits, 1);
    const hot = softmax(logits, 5);
    expect(cold[0] as number).toBeGreaterThan(base[0] as number);
    expect(hot[0] as number).toBeLessThan(base[0] as number);
    expect(hot[5] as number).toBeGreaterThan(base[5] as number);
  });

  it('temperature 0 is greedy decoding', () => {
    const p = softmax(logits, 0);
    expect(Array.from(p)).toEqual([1, 0, 0, 0, 0, 0]);
  });

  it('very high temperature approaches uniform', () => {
    const p = softmax(logits, 1e6);
    for (const v of p) expect(v).toBeCloseTo(1 / 6, 4);
  });
});

describe('nextTokenDistribution', () => {
  it('sorts candidates by probability and tracks the cumulative mass', () => {
    const d = nextTokenDistribution([0, 3, 1, 2]);
    expect(d.candidates.map((c) => c.id)).toEqual([1, 3, 2, 0]);
    expect(d.candidates[3]?.cumulative).toBeCloseTo(1, 12);
    expect(d.keptCount).toBe(4);
    expect(d.candidates.every((c) => c.kept)).toBe(true);
  });

  it('top-k keeps exactly the k best and renormalises', () => {
    const d = nextTokenDistribution(logits, opts({ topK: 2 }));
    expect(d.keptCount).toBe(2);
    expect(d.candidates.filter((c) => c.kept).map((c) => c.id)).toEqual([0, 1]);
    const mass = d.candidates.reduce((s, c) => s + c.adjusted, 0);
    expect(mass).toBeCloseTo(1, 12);
    expect(d.candidates[2]?.adjusted).toBe(0);
    // the ratio between the survivors is unchanged
    const [a, b] = d.candidates;
    expect((a?.adjusted as number) / (b?.adjusted as number)).toBeCloseTo(
      (a?.prob as number) / (b?.prob as number),
      10,
    );
  });

  it('top-p keeps the smallest prefix reaching the mass (and always at least one token)', () => {
    const base = nextTokenDistribution(logits);
    const p0 = base.candidates[0]?.prob as number;
    // p just above the best token's mass forces a second token in
    const d = nextTokenDistribution(logits, opts({ topP: p0 + 0.01 }));
    expect(d.keptCount).toBe(2);
    // a tiny p still keeps the best token
    expect(nextTokenDistribution(logits, opts({ topP: 0.0001 })).keptCount).toBe(1);
    // p = 1 disables the filter
    expect(nextTokenDistribution(logits, opts({ topP: 1 })).keptCount).toBe(6);
  });

  it('top-k and top-p combine by intersection', () => {
    const d = nextTokenDistribution(logits, opts({ topK: 3, topP: 0.5 }));
    expect(d.keptCount).toBeLessThanOrEqual(3);
    expect(d.keptCount).toBeGreaterThanOrEqual(1);
  });

  it('entropy is 0 when one token takes everything and log2(n) when uniform', () => {
    expect(nextTokenDistribution([5, 0, 0], opts({ temperature: 0 })).entropyBits).toBeCloseTo(
      0,
      12,
    );
    expect(nextTokenDistribution([1, 1, 1, 1]).entropyBits).toBeCloseTo(2, 12);
  });
});

describe('sampleToken', () => {
  it('matches the filtered distribution over many draws', () => {
    const d = nextTokenDistribution(logits, opts({ topK: 3 }));
    const rng = createRng(7);
    const counts = new Map<number, number>();
    const n = 20_000;
    for (let i = 0; i < n; i++) {
      const id = sampleToken(d, rng);
      counts.set(id, (counts.get(id) ?? 0) + 1);
    }
    expect([...counts.keys()].sort()).toEqual([0, 1, 2]); // dropped tokens never appear
    for (const c of d.candidates.filter((c) => c.kept)) {
      expect((counts.get(c.id) ?? 0) / n).toBeCloseTo(c.adjusted, 1);
    }
  });

  it('is deterministic for a seed and greedy at temperature 0', () => {
    const d = nextTokenDistribution(logits, opts({ temperature: 0 }));
    expect(sampleToken(d, createRng(1))).toBe(0);
    const d2 = nextTokenDistribution(logits);
    expect(sampleToken(d2, createRng(5))).toBe(sampleToken(d2, createRng(5)));
  });
});

describe('validateSampling', () => {
  it('rejects nonsense settings', () => {
    expect(() => validateSampling(opts({ temperature: -1 }))).toThrow(/temperature/);
    expect(() => validateSampling(opts({ temperature: Number.NaN }))).toThrow(/temperature/);
    expect(() => validateSampling(opts({ topK: -2 }))).toThrow(/topK/);
    expect(() => validateSampling(opts({ topK: 1.5 }))).toThrow(/topK/);
    expect(() => validateSampling(opts({ topP: 0 }))).toThrow(/topP/);
    expect(() => validateSampling(opts({ topP: 1.2 }))).toThrow(/topP/);
    expect(() => validateSampling(opts())).not.toThrow();
  });
});
