import { describe, expect, it } from 'vitest';
import { runClassifierExperiment } from '@/shared/lab-kit';
import { forestSpec } from './forestSpec';

const run = (id: string) =>
  runClassifierExperiment(
    forestSpec,
    forestSpec.experiments.find((e) => e.id === id)!,
  );

describe('random forest lab', () => {
  for (const exp of forestSpec.experiments) {
    it(`experiment "${exp.title}" holds`, () => {
      expect(exp.claim(runClassifierExperiment(forestSpec, exp))).toBe(true);
    });
  }

  it('averaging many trees beats one tree on held-out points', () => {
    expect(run('forest').scores.testAccuracy).toBeGreaterThan(
      run('one-tree').scores.testAccuracy as number,
    );
  });
});
