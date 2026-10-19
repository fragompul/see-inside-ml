import { describe, expect, it } from 'vitest';
import { runRegressorExperiment } from '@/shared/lab-kit';
import { polynomialSpec } from './polynomialSpec';

const run = (id: string) =>
  runRegressorExperiment(
    polynomialSpec,
    polynomialSpec.experiments.find((e) => e.id === id)!,
  );

describe('polynomial regression and regularisation lab', () => {
  for (const exp of polynomialSpec.experiments) {
    it(`experiment "${exp.title}" holds`, () => {
      expect(exp.claim(runRegressorExperiment(polynomialSpec, exp))).toBe(true);
    });
  }

  it('regularising degree 15 slashes the held-out error', () => {
    expect(run('ridge').scores.testRmse as number).toBeLessThan(
      (run('overfit').scores.testRmse as number) / 5,
    );
  });
});
