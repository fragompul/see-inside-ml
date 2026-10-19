import { describe, expect, it } from 'vitest';
import { runProjectorExperiment } from '@/shared/lab-kit/experiments';
import { somSpec } from './somSpec';

describe('SOM lab', () => {
  for (const exp of somSpec.experiments) {
    it(`experiment "${exp.title}" holds`, () => {
      expect(exp.claim(runProjectorExperiment(somSpec, exp))).toBe(true);
    });
  }
});
