import { describe, expect, it } from 'vitest';
import { runTabularExperiment } from '@/shared/lab-kit';
import { multipleRegressionSpec } from './multipleRegressionSpec';

describe('multiple regression lab', () => {
  for (const exp of multipleRegressionSpec.experiments) {
    it(`experiment "${exp.title}" holds`, () => {
      expect(exp.claim(runTabularExperiment(multipleRegressionSpec, exp))).toBe(true);
    });
  }
});
