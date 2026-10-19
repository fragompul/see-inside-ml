import { Matrix, createRng } from '@sim/core';
import { describe, expect, it } from 'vitest';
import fixture from '../fixtures/manifold.json';
import {
  classicalMds,
  continuity,
  geodesicDistances,
  graphComponents,
  isomap,
  lle,
  lleWeights,
  neighbourPreservation,
  smacofSteps,
  stress,
  trustworthiness,
} from './manifold';
import { pairwiseDistances } from './hierarchical';
import { PCA } from './pca';

const x = Matrix.from(fixture.x);
const n = x.rows;
const close = (a: number, b: number, tol: number) => expect(Math.abs(a - b)).toBeLessThan(tol);

describe('Isomap agrees with scikit-learn', () => {
  it('geodesic distances and the embedding (up to the sign of each axis)', () => {
    const g = geodesicDistances(x, fixture.isomap.neighbors);
    fixture.isomap.geodesic_sample.forEach((row, i) =>
      row.forEach((v, j) => close(g[i * n + j] as number, v, 1e-9)),
    );
    const { embedding } = isomap(x, fixture.isomap.neighbors, 2);
    fixture.isomap.embedding.forEach((row, i) =>
      row.forEach((v, c) => close(embedding.get(i, c), v, 1e-6)),
    );
  });
});

describe('locally linear embedding agrees with scikit-learn', () => {
  it('embedding eigenvectors (up to sign) and reconstruction error', () => {
    const r = lle(x, fixture.lle.neighbors, 2, 1e-3);
    close(r.reconstructionError, fixture.lle.reconstruction_error, 1e-9);
    for (let c = 0; c < 2; c++) {
      // eigenvectors are only defined up to a sign: compare the better of the two
      let same = 0;
      let flipped = 0;
      fixture.lle.embedding.forEach((row, i) => {
        same += Math.abs(r.embedding.get(i, c) - (row[c] as number));
        flipped += Math.abs(r.embedding.get(i, c) + (row[c] as number));
      });
      expect(Math.min(same, flipped)).toBeLessThan(1e-4);
    }
  });

  it('each point is reconstructed from its neighbours with weights that sum to one', () => {
    const { weights, neighborIndex } = lleWeights(x, 10);
    for (let i = 0; i < n; i++) {
      let sum = 0;
      for (let j = 0; j < n; j++) sum += weights.get(i, j);
      close(sum, 1, 1e-12);
      for (let j = 0; j < n; j++)
        if (weights.get(i, j) !== 0) expect(neighborIndex[i]).toContain(j);
    }
  });
});

describe('metric MDS (SMACOF) agrees with scikit-learn', () => {
  it('final embedding, stress and iteration count from the same start', () => {
    const d = pairwiseDistances(x);
    const states = Array.from(
      smacofSteps(d, n, Matrix.from(fixture.smacof.init), { tolerance: 1e-3, maxIterations: 300 }),
    );
    const last = states[states.length - 1]!;
    expect(last.iteration).toBe(fixture.smacof.n_iter);
    close(last.stress, fixture.smacof.stress, 1e-6 * fixture.smacof.stress);
    fixture.smacof.embedding.forEach((row, i) =>
      row.forEach((v, c) => close(last.embedding.get(i, c), v, 1e-6)),
    );
  });

  it('stress falls from one iteration to the next', () => {
    const d = pairwiseDistances(x);
    const states = Array.from(
      smacofSteps(d, n, Matrix.from(fixture.smacof.init), { maxIterations: 60 }),
    );
    for (let i = 1; i < states.length - 1; i++)
      expect(states[i]!.stress).toBeLessThanOrEqual(states[i - 1]!.stress + 1e-9);
  });
});

describe('trustworthiness agrees with scikit-learn', () => {
  const z = Matrix.from(fixture.trust.embedded);
  for (const k of [3, 5, 10]) {
    it(`k = ${k}: a random embedding and the Isomap one`, () => {
      close(
        trustworthiness(x, z, k),
        (fixture.trust.values as Record<string, number>)[String(k)] as number,
        1e-12,
      );
      const iso = isomap(x, 10, 2).embedding;
      close(
        trustworthiness(x, iso, k),
        (fixture.trust.iso_values as Record<string, number>)[String(k)] as number,
        1e-12,
      );
    });
  }
});

describe('properties', () => {
  it('classical MDS of Euclidean distances is PCA (same scores up to sign)', () => {
    const d = pairwiseDistances(x);
    const mds = classicalMds(d, n, 2).embedding;
    const pca = new PCA({ components: 2 }).fit(x).transform(x);
    for (let c = 0; c < 2; c++)
      for (let i = 0; i < n; i++) close(Math.abs(mds.get(i, c)), Math.abs(pca.get(i, c)), 1e-6);
  });

  it('a perfect copy has trustworthiness, continuity and neighbour preservation of 1', () => {
    expect(trustworthiness(x, x, 5)).toBeCloseTo(1, 12);
    expect(continuity(x, x, 5)).toBeCloseTo(1, 12);
    expect(neighbourPreservation(x, x, 5)).toBe(1);
  });

  it('a random embedding scores far lower than a faithful one', () => {
    const rng = createRng(2);
    const noise = new Matrix(
      n,
      2,
      Float64Array.from({ length: n * 2 }, () => rng.normal()),
    );
    const iso = isomap(x, 10, 2).embedding;
    expect(neighbourPreservation(x, iso, 5)).toBeGreaterThan(
      neighbourPreservation(x, noise, 5) + 0.3,
    );
  });

  it('Isomap unrolls the swiss roll better than PCA, which flattens it onto itself', () => {
    const iso = isomap(x, 10, 2).embedding;
    const pca = new PCA({ components: 2 }).fit(x).transform(x);
    expect(trustworthiness(x, iso, 5)).toBeGreaterThan(trustworthiness(x, pca, 5));
  });

  it('too few neighbours disconnect the graph and Isomap refuses', () => {
    expect(graphComponents(x, 10)).toBe(1);
    expect(graphComponents(x, 1)).toBeGreaterThan(1);
    expect(() => isomap(x, 1, 2)).toThrow(/disconnected/);
  });

  it('stress is zero for a perfect embedding of a planar point set', () => {
    const flat = Matrix.from(
      Array.from({ length: 30 }, (_, i) => [Math.cos(i), Math.sin(i * 0.7)]),
    );
    const d = pairwiseDistances(flat);
    expect(stress(d, flat)).toBeCloseTo(0, 18);
  });
});
