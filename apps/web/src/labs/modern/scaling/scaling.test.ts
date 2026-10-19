import { describe, expect, it } from 'vitest';
import { DEFAULT_SCALING, runScaling } from './scalingCore';
import { scalingExperiments } from './scalingSpec';

describe('Scaling laws lab', () => {
  for (const exp of scalingExperiments) {
    it(`experiment "${exp.title}" holds`, () => {
      expect(exp.claim(runScaling({ ...DEFAULT_SCALING, ...exp.scenario }))).toBe(true);
    });
  }

  it('the loss never goes below the noise floor', () => {
    const r = runScaling(DEFAULT_SCALING);
    expect(Math.min(...r.loss.flat())).toBeGreaterThan(r.floor);
  });
});
