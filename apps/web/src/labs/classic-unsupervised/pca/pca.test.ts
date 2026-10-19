import { describe, expect, it } from 'vitest';
import { runProjectorExperiment } from '@/shared/lab-kit';
import { pcaSpec } from './pcaSpec';

const run = (id: string) =>
  runProjectorExperiment(
    pcaSpec,
    pcaSpec.experiments.find((e) => e.id === id)!,
  );

describe('PCA lab', () => {
  for (const exp of pcaSpec.experiments) {
    it(`experiment "${exp.title}" holds`, () => {
      expect(exp.claim(runProjectorExperiment(pcaSpec, exp))).toBe(true);
    });
  }

  it('a nearly one-dimensional curve survives a flat view better than a rolled-up sheet', () => {
    expect(run('spring').scores.preservation).toBeGreaterThan(run('roll').scores.preservation);
  });

  it('a very sharp kernel is worse than a linear projection on the roll', () => {
    expect(run('roll').scores.trustworthiness).toBeGreaterThan(
      run('sharp-kernel').scores.trustworthiness,
    );
  });
});
