import { describe, expect, it } from 'vitest';
import { DEFAULT_LM, runLm } from './lmCore';
import { lmExperiments } from './lmSpec';

describe('N-gram versus neural lab', () => {
  for (const exp of lmExperiments) {
    it(`experiment "${exp.title}" holds`, () => {
      expect(exp.claim(runLm({ ...DEFAULT_LM, ...exp.scenario }))).toBe(true);
    }, 300_000);
  }
});
