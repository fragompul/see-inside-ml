import { Matrix, createRng } from '@sim/core';
import { describe, expect, it } from 'vitest';
import fixture from '../fixtures/tsne.json';
import { neighbourPreservation, trustworthiness } from './manifold';
import {
  conditionalProbabilities,
  jointProbabilities,
  klAndGradient,
  tsne,
  tsneSteps,
} from './tsne';
import { findAB, fuzzyGraph, umap, umapSteps } from './umap';
import { squaredDistances } from './manifold';
import { KMeans } from './kmeans';
import { adjustedRandIndex } from '@sim/core';

const x = Matrix.from(fixture.x);
const n = x.rows;
const close = (a: number, b: number, tol: number) => expect(Math.abs(a - b)).toBeLessThan(tol);

describe('t-SNE agrees with scikit-learn', () => {
  for (const ref of fixture.cases) {
    it(`perplexity ${ref.perplexity}: joint probabilities, KL divergence and its gradient`, () => {
      const p = jointProbabilities(x, ref.perplexity);
      ref.p_full.forEach((row, i) => row.forEach((v, j) => close(p[i * n + j] as number, v, 1e-6)));
      const y = Matrix.from(ref.y);
      const { kl, grad } = klAndGradient(p, y);
      close(kl, ref.kl, 1e-5);
      ref.grad.forEach((row, i) =>
        row.forEach((v, c) => close(grad[i * 2 + c] as number, v, 1e-5)),
      );
    });
  }

  it('every conditional distribution has the requested perplexity', () => {
    const cond = conditionalProbabilities(squaredDistances(x), n, 12);
    for (let i = 0; i < n; i++) {
      let h = 0;
      let sum = 0;
      for (let j = 0; j < n; j++) {
        const p = cond[i * n + j] as number;
        sum += p;
        if (p > 0) h -= p * Math.log(p);
      }
      close(sum, 1, 1e-9);
      close(Math.exp(h), 12, 0.05);
    }
  });
});

describe('t-SNE as an optimiser', () => {
  it('lowers the KL divergence, and its final value is close to scikit-learn’s exact method', () => {
    const steps = Array.from(tsneSteps(x, { perplexity: 15, iterations: 600 }));
    const afterEarly = steps.filter((s) => !s.exaggerating);
    expect(afterEarly[afterEarly.length - 1]!.kl).toBeLessThan(afterEarly[0]!.kl);
    // local optima differ between implementations, but both land in the same neighbourhood of values
    expect(afterEarly[afterEarly.length - 1]!.kl).toBeLessThan(fixture.sklearn_final_kl * 1.6);
  });

  it('keeps neighbours together on clustered data and is reproducible for a seed', () => {
    const a = tsne(x, { perplexity: 10, iterations: 400, init: 'random', seed: 3 });
    const b = tsne(x, { perplexity: 10, iterations: 400, init: 'random', seed: 3 });
    expect(Array.from(a.embedding.data)).toEqual(Array.from(b.embedding.data));
    expect(trustworthiness(x, a.embedding, 5)).toBeGreaterThan(0.9);
    expect(neighbourPreservation(x, a.embedding, 5)).toBeGreaterThan(0.4);
  });
});

describe('UMAP', () => {
  it('fits the low-dimensional kernel to umap-learn’s known constants', () => {
    const { a, b } = findAB(1, 0.1);
    close(a, 1.577, 0.03);
    close(b, 0.895, 0.02);
  });

  it('builds a symmetric fuzzy graph with strengths in [0, 1] and a full-strength link at every point’s nearest neighbour', () => {
    const g = fuzzyGraph(x, 10);
    for (let i = 0; i < n; i++)
      for (let j = 0; j < n; j++) {
        const w = g.weights[i * n + j] as number;
        expect(w).toBeGreaterThanOrEqual(0);
        expect(w).toBeLessThanOrEqual(1 + 1e-12);
        close(w, g.weights[j * n + i] as number, 1e-12);
      }
    for (let i = 0; i < n; i++) {
      const nearest = (g.neighbors[i] as number[])[0] as number;
      close(g.weights[i * n + nearest] as number, 1, 1e-9);
    }
  });

  it('each point’s kernel is calibrated so its memberships sum to log2(k)', () => {
    const k = 10;
    const g = fuzzyGraph(x, k);
    // memberships sum over the k neighbours of the point (before symmetrisation) to log2(k)
    // (the sigma search leaves 1 for the nearest neighbour, so check the sum of the others)
    for (let i = 0; i < n; i += 7) {
      const nb = g.neighbors[i] as number[];
      let sum = 0;
      nb.forEach((j) => {
        let d = 0;
        for (let c = 0; c < x.cols; c++) d += (x.get(i, c) - x.get(j, c)) ** 2;
        sum += Math.exp(-Math.max(0, Math.sqrt(d) - (g.rho[i] as number)) / (g.sigma[i] as number));
      });
      close(sum, Math.log2(k), 0.01);
    }
  });

  it('keeps clusters apart and neighbours near, reproducibly', () => {
    const e1 = umap(x, { neighbors: 10, epochs: 150, seed: 2 });
    const e2 = umap(x, { neighbors: 10, epochs: 150, seed: 2 });
    expect(Array.from(e1.data)).toEqual(Array.from(e2.data));
    expect(trustworthiness(x, e1, 5)).toBeGreaterThan(0.9);
    const labels = new KMeans({ k: 3, restarts: 5, seed: 1 }).fit(e1).labels;
    expect(adjustedRandIndex(fixture.labels, labels)).toBeGreaterThan(0.5);
  });

  it('streams intermediate layouts and a different seed gives a different layout', () => {
    const states = Array.from(umapSteps(x, { neighbors: 10, epochs: 40, seed: 2 }, 10));
    expect(states.map((s) => s.epoch)).toEqual([0, 10, 20, 30, 40]);
    const other = umap(x, { neighbors: 10, epochs: 40, seed: 5 });
    expect(Array.from(other.data)).not.toEqual(Array.from(states[4]!.embedding.data));
    void createRng;
  });
});
