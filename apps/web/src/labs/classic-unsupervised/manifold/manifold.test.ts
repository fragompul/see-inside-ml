import { describe, expect, it } from 'vitest';
import { runProjector, runProjectorExperiment, buildProjectionData } from '@/shared/lab-kit';
import { makeEmbedder, manifoldSpec } from './manifoldSpec';

const run = (id: string) =>
  runProjectorExperiment(
    manifoldSpec,
    manifoldSpec.experiments.find((e) => e.id === id)!,
  );

describe('manifold learning lab', () => {
  for (const exp of manifoldSpec.experiments) {
    it(`experiment "${exp.title}" holds`, () => {
      expect(exp.claim(runProjectorExperiment(manifoldSpec, exp))).toBe(true);
    });
  }

  it('the right method for the shape wins: Isomap on the S-curve, t-SNE on the roll', () => {
    expect(run('scurve-isomap').scores.trustworthiness).toBeGreaterThan(
      run('roll-shortcut').scores.trustworthiness,
    );
    expect(run('roll-tsne').scores.trustworthiness).toBeGreaterThan(
      run('roll-shortcut').scores.trustworthiness,
    );
  });

  it('Isomap refuses clusters that are not connected, and says so', () => {
    const data = buildProjectionData({ dataset: 'clusters', n: 150, noise: 0.1, seed: 4 });
    const model = makeEmbedder({
      method: 'isomap',
      neighbors: 5,
      perplexity: 20,
      minDist: 0.1,
      seed: 3,
    });
    expect(() => runProjector(model, data)).toThrow(/disconnected/);
  });

  it('every method produces a 2-D layout for every kind of data', () => {
    for (const method of ['mds', 'smacof', 'lle', 'tsne', 'umap'] as const) {
      const data = buildProjectionData({ dataset: 'scurve', n: 60, noise: 0.1, seed: 2 });
      const r = runProjector(
        makeEmbedder({ method, neighbors: 8, perplexity: 10, minDist: 0.1, seed: 2 }),
        data,
      );
      expect(r.embedding.cols).toBe(2);
      expect(r.embedding.rows).toBe(60);
      expect(r.embedding.data.every(Number.isFinite)).toBe(true);
    }
  });
});
