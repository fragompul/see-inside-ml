import { describe, expect, it } from 'vitest';
import { createRng } from '@sim/core';
import {
  ALL_ANALOGIES,
  ANALOGY_GROUPS,
  CATEGORY_LABEL,
  VOCABULARY,
  WORLD,
  categoryOf,
  generateCorpus,
  randomSentence,
} from './corpus';
import { SkipGram, cosine, nearestWords, noiseSampler } from './word2vec';

describe('toy-world corpus', () => {
  it('has a unique vocabulary and every word has exactly one category', () => {
    expect(new Set(VOCABULARY).size).toBe(VOCABULARY.length);
    expect(Object.keys(WORLD)).toEqual(Object.keys(CATEGORY_LABEL));
    VOCABULARY.forEach((w) => expect(() => categoryOf(w)).not.toThrow());
    expect(() => categoryOf('banana')).toThrow(RangeError);
  });

  it('only uses words of the vocabulary and is reproducible for a seed', () => {
    const corpus = generateCorpus(800, createRng(3));
    const words = new Set(VOCABULARY);
    corpus.forEach((s) => s.forEach((w) => expect(words.has(w)).toBe(true)));
    expect(generateCorpus(800, createRng(3))).toEqual(corpus);
    expect(generateCorpus(800, createRng(4))).not.toEqual(corpus);
  });

  it('uses every content word, so all of them can be learned', () => {
    const seen = new Set(generateCorpus(3000, createRng(1)).flat());
    for (const [category, words] of Object.entries(WORLD))
      if (category !== 'function') words.forEach((w) => expect(seen.has(w)).toBe(true));
  });

  it('the pronoun always agrees with the gender of the noun', () => {
    const female = new Set(['queen', 'princess', 'woman', 'girl', 'aunt']);
    const male = new Set(['king', 'prince', 'man', 'boy', 'uncle']);
    const rng = createRng(2);
    let checked = 0;
    for (let i = 0; i < 3000; i++) {
      const s = randomSentence(rng);
      const subject = s[1] as string;
      if (female.has(subject) && (s.includes('he') || s.includes('his')))
        throw new Error(s.join(' '));
      if (male.has(subject) && (s.includes('she') || s.includes('her')))
        throw new Error(s.join(' '));
      if (female.has(subject) || male.has(subject)) checked += 1;
    }
    expect(checked).toBeGreaterThan(500);
  });

  it('the capital always goes with its own country', () => {
    const rng = createRng(5);
    const pairs: Record<string, string> = {
      paris: 'france',
      rome: 'italy',
      madrid: 'spain',
      tokyo: 'japan',
      cairo: 'egypt',
      lima: 'peru',
    };
    for (let i = 0; i < 3000; i++) {
      const s = randomSentence(rng);
      const capital = s.find((w) => w in pairs);
      const country = s.find((w) => Object.values(pairs).includes(w));
      if (capital && country) expect(pairs[capital]).toBe(country);
    }
  });

  it('analogy questions are well formed', () => {
    expect(ALL_ANALOGIES.length).toBeGreaterThan(40);
    expect(ANALOGY_GROUPS.map((g) => g.id)).toEqual(['gender', 'capital', 'royalty']);
    const words = new Set(VOCABULARY);
    for (const q of ALL_ANALOGIES) {
      expect(q.every((w) => words.has(w))).toBe(true);
      expect(new Set(q).size).toBe(4);
    }
    expect(ALL_ANALOGIES.some((q) => q.join() === 'man,woman,king,queen')).toBe(true);
    expect(ALL_ANALOGIES.some((q) => q.join() === 'france,paris,italy,rome')).toBe(true);
  });
});

describe('skip-gram', () => {
  const make = (seed = 1) => new SkipGram({ vocabulary: 9, dim: 5 }, createRng(seed));
  const batch = {
    centers: [0, 3, 3, 7],
    contexts: [1, 4, 5, 8],
    negs: [2, 6, 7, 1, 0, 8, 5, 5],
    negatives: 2,
  };

  it('the update is exactly a gradient step on the summed loss (finite differences)', () => {
    const model = make();
    // random output vectors so the gradient with respect to both tables is non-trivial
    const rng = createRng(9);
    for (let i = 0; i < model.output.length; i++) model.output[i] = rng.uniform(-0.5, 0.5);
    const { centers, contexts, negs, negatives } = batch;
    const pairs = centers.length;
    const beforeIn = Float64Array.from(model.input);
    const beforeOut = Float64Array.from(model.output);
    const lr = 1e-3;
    model.step(centers, contexts, negs, negatives, lr);
    const h = 1e-6;
    let worst = 0;
    for (const [table, before] of [
      ['input', beforeIn],
      ['output', beforeOut],
    ] as const) {
      const probe = make();
      probe.input.set(beforeIn);
      probe.output.set(beforeOut);
      const data = probe[table];
      for (let i = 0; i < data.length; i++) {
        const original = data[i] as number;
        data[i] = original + h;
        const up = probe.loss(centers, contexts, negs, negatives) * pairs;
        data[i] = original - h;
        const down = probe.loss(centers, contexts, negs, negatives) * pairs;
        data[i] = original;
        const numeric = (up - down) / (2 * h);
        const applied = -((model[table][i] as number) - (before[i] as number)) / lr;
        worst = Math.max(
          worst,
          Math.abs(numeric - applied) / Math.max(1e-6, Math.abs(numeric) + Math.abs(applied)),
        );
      }
    }
    expect(worst).toBeLessThan(1e-4);
  });

  it('training on a pair raises its score and lowers the loss', () => {
    const model = make();
    const { centers, contexts, negs, negatives } = batch;
    const first = model.loss(centers, contexts, negs, negatives);
    let reported = 0;
    for (let i = 0; i < 200; i++) reported = model.step(centers, contexts, negs, negatives, 0.1);
    expect(model.loss(centers, contexts, negs, negatives)).toBeLessThan(first * 0.2);
    expect(reported).toBeLessThan(first);
  });

  it('reports the loss before the update and validates the configuration', () => {
    const model = make();
    const { centers, contexts, negs, negatives } = batch;
    const expected = model.loss(centers, contexts, negs, negatives);
    expect(model.step(centers, contexts, negs, negatives, 0.05)).toBeCloseTo(expected, 12);
    expect(() => new SkipGram({ vocabulary: 1, dim: 5 }, createRng(1))).toThrow(RangeError);
    expect(() => new SkipGram({ vocabulary: 9, dim: 1 }, createRng(1))).toThrow(RangeError);
  });
});

describe('vector helpers', () => {
  it('cosine similarity', () => {
    expect(cosine([1, 0], [1, 0])).toBeCloseTo(1, 12);
    expect(cosine([1, 0], [0, 1])).toBeCloseTo(0, 12);
    expect(cosine([1, 0], [-2, 0])).toBeCloseTo(-1, 12);
    expect(cosine([0, 0], [1, 1])).toBe(0);
  });

  it('nearestWords ranks by similarity, skips excluded ids and returns at most k', () => {
    const vectors = Float64Array.from([1, 0, 0.9, 0.1, 0, 1, -1, 0]);
    expect(nearestWords(vectors, 2, [1, 0], 2).map((n) => n.id)).toEqual([0, 1]);
    expect(nearestWords(vectors, 2, [1, 0], 2, [0]).map((n) => n.id)).toEqual([1, 2]);
    expect(nearestWords(vectors, 2, [1, 0], 10)).toHaveLength(4);
  });

  it('the noise sampler follows count^0.75 and never returns an unknown word', () => {
    const draw = noiseSampler([1, 16, 81], 0.75);
    const n = 30_000;
    const counts = [0, 0, 0];
    for (let i = 0; i < n; i++) {
      const id = draw(createRng(i));
      counts[id] = (counts[id] as number) + 1;
    }
    const weights = [1, 8, 27]; // 1^0.75, 16^0.75, 81^0.75
    const total = 36;
    counts.forEach((c, i) => expect(c / n).toBeCloseTo((weights[i] as number) / total, 1));
    expect(counts.reduce((a, b) => a + b, 0)).toBe(n);
  });
});
