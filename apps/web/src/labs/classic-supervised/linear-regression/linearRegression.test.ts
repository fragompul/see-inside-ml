import { describe, expect, it } from 'vitest';
import { runRegressorExperiment } from '@/shared/lab-kit';
import { linearRegressionSpec } from './linearRegressionSpec';

describe('simple linear regression lab', () => {
  for (const exp of linearRegressionSpec.experiments) {
    it(`experiment "${exp.title}" holds`, () => {
      expect(exp.claim(runRegressorExperiment(linearRegressionSpec, exp))).toBe(true);
    });
  }
});
