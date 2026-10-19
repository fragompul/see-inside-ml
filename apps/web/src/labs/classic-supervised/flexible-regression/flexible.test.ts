import { describe, expect, it } from 'vitest';
import { runRegressorExperiment } from '@/shared/lab-kit';
import { flexSpec } from './flexSpec';

describe('flexible regression lab', () => {
  for (const exp of flexSpec.experiments) {
    it(`experiment "${exp.title}" holds`, () => {
      expect(exp.claim(runRegressorExperiment(flexSpec, exp))).toBe(true);
    });
  }
});
