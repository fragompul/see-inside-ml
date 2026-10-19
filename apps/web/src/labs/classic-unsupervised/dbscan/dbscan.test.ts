import { describe, expect, it } from 'vitest';
import { runClusterExperiment } from '@/shared/lab-kit';
import { dbscanSpec } from './dbscanSpec';

describe('DBSCAN lab', () => {
  for (const exp of dbscanSpec.experiments) {
    it(`experiment "${exp.title}" holds`, () => {
      expect(exp.claim(runClusterExperiment(dbscanSpec, exp))).toBe(true);
    });
  }
});
