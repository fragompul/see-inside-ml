import { describe, expect, it } from 'vitest';
import { runClassifierExperiment } from '@/shared/lab-kit';
import { gaussianSpec } from './gaussianSpec';

const run = (id: string) =>
  runClassifierExperiment(
    gaussianSpec,
    gaussianSpec.experiments.find((e) => e.id === id)!,
  );

describe('Gaussian classifiers lab', () => {
  for (const exp of gaussianSpec.experiments) {
    it(`experiment "${exp.title}" holds`, () => {
      expect(exp.claim(runClassifierExperiment(gaussianSpec, exp))).toBe(true);
    });
  }

  it('on correlated data LDA beats naive Bayes; on cigars QDA beats LDA', () => {
    expect(run('lda-correlated').scores.trainAccuracy).toBeGreaterThan(
      run('nb-correlated').scores.trainAccuracy,
    );
    expect(run('qda-cigars').scores.trainAccuracy).toBeGreaterThan(
      run('lda-cigars').scores.trainAccuracy,
    );
  });
});
