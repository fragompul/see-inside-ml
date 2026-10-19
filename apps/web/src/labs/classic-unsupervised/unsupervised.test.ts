import { describe, expect, it } from 'vitest';
import { runClusterExperiment, type ClusterLabSpec } from '@/shared/lab-kit';
import { gmmSpec } from './gmm/gmmSpec';
import { meanshiftSpec } from './meanshift/meanshiftSpec';
import { spectralSpec } from './spectral/spectralSpec';

const specs: [string, ClusterLabSpec][] = [
  ['mean shift', meanshiftSpec],
  ['Gaussian mixture', gmmSpec],
  ['spectral clustering', spectralSpec],
];

for (const [name, spec] of specs) {
  describe(`${name} lab`, () => {
    for (const exp of spec.experiments) {
      it(`experiment "${exp.title}" holds`, () => {
        expect(exp.claim(runClusterExperiment(spec, exp))).toBe(true);
      });
    }
  });
}

describe('the contrasts the labs rely on', () => {
  const run = (spec: ClusterLabSpec, id: string) =>
    runClusterExperiment(
      spec,
      spec.experiments.find((e) => e.id === id)!,
    );

  it('a tilting ellipse beats a circle on tilted data; spectral beats the density peaks on crescents', () => {
    expect(run(gmmSpec, 'tilted-full').scores.ari).toBeGreaterThan(
      run(gmmSpec, 'tilted-spherical').scores.ari,
    );
    expect(run(spectralSpec, 'moons').scores.ari).toBeGreaterThan(
      run(meanshiftSpec, 'moons').scores.ari,
    );
  });
});
