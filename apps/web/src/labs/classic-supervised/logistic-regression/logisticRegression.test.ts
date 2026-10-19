import { describe, expect, it } from 'vitest';
import { runClassifierExperiment } from '@/shared/lab-kit';
import { logisticSpec } from './logisticSpec';

describe('logistic regression lab', () => {
  for (const exp of logisticSpec.experiments) {
    it(`experiment "${exp.title}" holds`, () => {
      expect(exp.claim(runClassifierExperiment(logisticSpec, exp))).toBe(true);
    });
  }
});
