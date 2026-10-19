import { describe, expect, it } from 'vitest';
import { runRegressorExperiment } from '@/shared/lab-kit';
import { robustSpec } from './robustSpec';

describe('robust regression and GLM lab', () => {
  for (const exp of robustSpec.experiments) {
    it(`experiment "${exp.title}" holds`, () => {
      expect(exp.claim(runRegressorExperiment(robustSpec, exp))).toBe(true);
    });
  }
});
