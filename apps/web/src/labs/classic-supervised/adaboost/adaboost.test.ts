import { describe, expect, it } from 'vitest';
import { runClassifierExperiment } from '@/shared/lab-kit';
import { adaboostSpec } from './adaboostSpec';

describe('AdaBoost lab', () => {
  for (const exp of adaboostSpec.experiments) {
    it(`experiment "${exp.title}" holds`, () => {
      expect(exp.claim(runClassifierExperiment(adaboostSpec, exp))).toBe(true);
    });
  }

  it('boosting improves on its first weak learner', () => {
    const run = (id: string) =>
      runClassifierExperiment(
        adaboostSpec,
        adaboostSpec.experiments.find((e) => e.id === id)!,
      );
    expect(run('many-stumps').scores.trainAccuracy).toBeGreaterThan(
      run('one-stump').scores.trainAccuracy,
    );
  });
});
