import { describe, expect, it } from 'vitest';
import { runRegressorExperiment } from '@/shared/lab-kit';
import { boostingSpec } from './boostingSpec';

const run = (id: string) =>
  runRegressorExperiment(
    boostingSpec,
    boostingSpec.experiments.find((e) => e.id === id)!,
  );

describe('gradient boosting lab', () => {
  for (const exp of boostingSpec.experiments) {
    it(`experiment "${exp.title}" holds`, () => {
      expect(exp.claim(runRegressorExperiment(boostingSpec, exp))).toBe(true);
    });
  }

  it('many small steps beat one big step on held-out data', () => {
    expect(run('shrunk').scores.testRmse as number).toBeLessThan(
      run('one-tree').scores.testRmse as number,
    );
  });
});
