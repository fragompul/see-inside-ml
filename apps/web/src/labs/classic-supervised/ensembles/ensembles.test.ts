import { describe, expect, it } from 'vitest';
import { runClassifierExperiment } from '@/shared/lab-kit';
import { ensemblesSpec } from './ensemblesSpec';

describe('voting and stacking lab', () => {
  for (const exp of ensemblesSpec.experiments) {
    it(`experiment "${exp.title}" holds`, () => {
      expect(exp.claim(runClassifierExperiment(ensemblesSpec, exp))).toBe(true);
    });
  }
});
