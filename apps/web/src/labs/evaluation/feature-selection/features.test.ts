import { describe, expect, it } from 'vitest';
import { DEFAULT_FEATURES, GROUP, NAMES, runFeatures } from './featuresCore';
import { featuresExperiments } from './featuresSpec';

describe('feature selection lab', () => {
  for (const exp of featuresExperiments) {
    it(`experiment "${exp.title}" holds`, () => {
      expect(exp.claim(runFeatures({ ...DEFAULT_FEATURES, ...exp.scenario }))).toBe(true);
    }, 60_000);
  }

  it('the ranking is a permutation of the 20 columns and the chosen ones are its head', () => {
    const r = runFeatures(DEFAULT_FEATURES);
    expect([...r.ranking].sort((a, b) => a - b)).toEqual(Array.from({ length: 20 }, (_, i) => i));
    expect(r.chosen).toEqual(r.ranking.slice(0, DEFAULT_FEATURES.keep));
  });

  it('the column layout is four signals, four copies and twelve noise features', () => {
    expect(NAMES).toHaveLength(20);
    expect(GROUP.filter((g) => g >= 0)).toHaveLength(8);
    expect(GROUP.filter((g) => g < 0)).toHaveLength(12);
  });

  it('a model that knows the true signals is about as good as one with every column', () => {
    const r = runFeatures(DEFAULT_FEATURES);
    expect(Math.abs(r.oracleR2 - r.allR2)).toBeLessThan(0.01);
  });
});
