import { describe, expect, it } from 'vitest';
import { runClusterExperiment } from '@/shared/lab-kit';
import { kmeansOf, kmeansSpec } from './kmeansSpec';

const run = (id: string) =>
  runClusterExperiment(
    kmeansSpec,
    kmeansSpec.experiments.find((e) => e.id === id)!,
  );

describe('k-means lab', () => {
  for (const exp of kmeansSpec.experiments) {
    it(`experiment "${exp.title}" holds`, () => {
      expect(exp.claim(runClusterExperiment(kmeansSpec, exp))).toBe(true);
    });
  }

  it('restarts lower the inertia of the unlucky start', () => {
    expect(kmeansOf(run('restarts').model)!.inertia).toBeLessThan(
      kmeansOf(run('unlucky').model)!.inertia,
    );
  });

  it('too few or too many clusters match the truth worse than the right k', () => {
    expect(run('blobs').scores.ari).toBeGreaterThan(run('too-few').scores.ari);
    expect(run('blobs').scores.ari).toBeGreaterThan(run('too-many').scores.ari);
  });
});
