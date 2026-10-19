import { describe, expect, it } from 'vitest';
import { runClassifierExperiment } from '@/shared/lab-kit';
import { knnSpec } from './knnSpec';

const byId = (id: string) => knnSpec.experiments.find((e) => e.id === id)!;

describe('k-NN lab', () => {
  for (const exp of knnSpec.experiments) {
    it(`experiment "${exp.title}" holds`, () => {
      expect(exp.claim(runClassifierExperiment(knnSpec, exp))).toBe(true);
    });
  }

  it('smoothing helps: a moderate k beats k = 1 on held-out points', () => {
    const one = runClassifierExperiment(knnSpec, byId('k1'));
    const some = runClassifierExperiment(knnSpec, byId('k-moderate'));
    expect(some.scores.testAccuracy).toBeGreaterThan(one.scores.testAccuracy as number);
  });
});
