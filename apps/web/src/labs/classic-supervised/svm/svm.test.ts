import { describe, expect, it } from 'vitest';
import { runClassifierExperiment } from '@/shared/lab-kit';
import { svmSpec } from './svmSpec';

const run = (id: string) =>
  runClassifierExperiment(
    svmSpec,
    svmSpec.experiments.find((e) => e.id === id)!,
  );

describe('SVM lab', () => {
  for (const exp of svmSpec.experiments) {
    it(`experiment "${exp.title}" holds`, () => {
      expect(exp.claim(runClassifierExperiment(svmSpec, exp))).toBe(true);
    });
  }

  it('the RBF kernel fits the crescents better than a line', () => {
    expect(run('rbf').scores.trainAccuracy).toBeGreaterThan(
      run('linear-fails').scores.trainAccuracy,
    );
  });
});
