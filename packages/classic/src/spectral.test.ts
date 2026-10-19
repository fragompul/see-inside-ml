import { Matrix, adjustedRandIndex, createRng, makeClassification } from '@sim/core';
import { describe, expect, it } from 'vitest';
import fixture from '../fixtures/spectral.json';
import { KMeans } from './kmeans';
import {
  SpectralClustering,
  affinityMatrix,
  nearZeroEigenvalues,
  spectralEmbedding,
} from './spectral';

describe('the spectrum of the normalised Laplacian agrees with NumPy', () => {
  const x = Matrix.from(fixture.x);
  const n = x.rows;

  it('RBF affinity', () => {
    const w = affinityMatrix(x, { kind: 'rbf', gamma: fixture.rbf.gamma });
    const { eigenvalues } = spectralEmbedding(w, n, 3);
    const mine = Array.from(eigenvalues).sort((a, b) => a - b);
    fixture.rbf.eigenvalues.forEach((v, i) =>
      expect(Math.abs((mine[i] as number) - v)).toBeLessThan(1e-9),
    );
  });

  it('k-nearest-neighbour affinity (symmetrised)', () => {
    const w = affinityMatrix(x, { kind: 'knn', neighbors: fixture.knn.neighbors });
    expect(w.reduce((a, b) => a + b, 0)).toBeCloseTo(fixture.knn.affinity_sum, 9);
    const mine = Array.from(spectralEmbedding(w, n, 3).eigenvalues).sort((a, b) => a - b);
    fixture.knn.eigenvalues.forEach((v, i) =>
      expect(Math.abs((mine[i] as number) - v)).toBeLessThan(1e-9),
    );
  });
});

describe('spectral clustering', () => {
  it('separates the two crescents that k-means cannot', () => {
    const moons = makeClassification('moons', 160, 0.08, createRng(4));
    const spectral = new SpectralClustering({ k: 2, affinity: { kind: 'knn', neighbors: 10 } }).fit(
      moons.x,
    );
    const plain = new KMeans({ k: 2, restarts: 5 }).fit(moons.x);
    expect(adjustedRandIndex(moons.y, spectral.labels)).toBeGreaterThan(0.95);
    expect(adjustedRandIndex(moons.y, plain.labels)).toBeLessThan(0.6);
  });

  it('separates a ring from its centre', () => {
    const circles = makeClassification('circles', 200, 0.05, createRng(5));
    const m = new SpectralClustering({ k: 2, affinity: { kind: 'knn', neighbors: 8 } }).fit(
      circles.x,
    );
    expect(adjustedRandIndex(circles.y, m.labels)).toBeGreaterThan(0.9);
  });

  it('the number of near-zero eigenvalues counts the disconnected pieces of the graph', () => {
    const blobs = makeClassification('blobs', 150, 0.0, createRng(2));
    const w = affinityMatrix(blobs.x, { kind: 'rbf', gamma: 6 });
    const { eigenvalues } = spectralEmbedding(w, blobs.x.rows, 3);
    expect(nearZeroEigenvalues(eigenvalues, 0.05)).toBe(3);
  });

  it('eigenvalues lie in [0, 2] and the smallest is zero', () => {
    const blobs = makeClassification('blobs', 90, 0.1, createRng(2));
    const { eigenvalues } = spectralEmbedding(
      affinityMatrix(blobs.x, { kind: 'rbf', gamma: 2 }),
      90,
      3,
    );
    expect(Math.min(...eigenvalues)).toBeGreaterThan(-1e-9);
    expect(Math.max(...eigenvalues)).toBeLessThan(2 + 1e-9);
    expect(Math.abs(eigenvalues[0] as number)).toBeLessThan(1e-9);
  });

  it('a too narrow RBF kernel fragments the graph and the clustering falls apart', () => {
    const moons = makeClassification('moons', 160, 0.08, createRng(4));
    const good = new SpectralClustering({ k: 2, affinity: { kind: 'rbf', gamma: 8 } }).fit(moons.x);
    const bad = new SpectralClustering({ k: 2, affinity: { kind: 'rbf', gamma: 5000 } }).fit(
      moons.x,
    );
    expect(adjustedRandIndex(moons.y, good.labels)).toBeGreaterThan(
      adjustedRandIndex(moons.y, bad.labels),
    );
  });
});
