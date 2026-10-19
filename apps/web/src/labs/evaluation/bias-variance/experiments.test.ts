import { describe, expect, it } from 'vitest';
import { DEFAULT_BV, simulate } from './biasVarianceCore';
import { bvExperiments } from './biasVarianceSpec';

describe('bias-variance lab', () => {
  for (const exp of bvExperiments) {
    it(`experiment "${exp.title}" holds`, () => {
      expect(exp.claim(simulate({ ...DEFAULT_BV, ...exp.scenario }))).toBe(true);
    });
  }
});
