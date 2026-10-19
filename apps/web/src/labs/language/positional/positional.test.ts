import { describe, expect, it } from 'vitest';
import { DEFAULT_POSITIONAL, runPositional } from './positionalCore';
import { positionalExperiments } from './positionalSpec';

describe('Positional encoding lab', () => {
  for (const exp of positionalExperiments) {
    it(`experiment "${exp.title}" holds`, () => {
      expect(exp.claim(runPositional({ ...DEFAULT_POSITIONAL, ...exp.scenario }))).toBe(true);
    });
  }
});
