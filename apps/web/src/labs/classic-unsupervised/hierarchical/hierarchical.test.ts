import { describe, expect, it } from 'vitest';
import { runClusterExperiment } from '@/shared/lab-kit';
import { hierarchicalSpec } from './hierarchicalSpec';

const run = (id: string) =>
  runClusterExperiment(
    hierarchicalSpec,
    hierarchicalSpec.experiments.find((e) => e.id === id)!,
  );

describe('hierarchical clustering lab', () => {
  for (const exp of hierarchicalSpec.experiments) {
    it(`experiment "${exp.title}" holds`, () => {
      expect(exp.claim(runClusterExperiment(hierarchicalSpec, exp))).toBe(true);
    });
  }

  it('the best linkage depends on the shape of the data', () => {
    // single wins on crescents, Ward on blobs
    expect(run('single-moons').scores.ari).toBeGreaterThan(run('ward-moons').scores.ari);
    expect(run('ward-blobs').scores.ari).toBeGreaterThan(run('chaining').scores.ari);
  });
});
