import { describe, expect, it } from 'vitest';
import { runClusterExperiment } from '@/shared/lab-kit/experiments';
import { anomalySpec } from './anomalySpec';

describe('anomaly lab', () => {
  for (const exp of anomalySpec.experiments) {
    it(`experiment "${exp.title}" holds`, () => {
      expect(exp.claim(runClusterExperiment(anomalySpec, exp))).toBe(true);
    });
  }
});
