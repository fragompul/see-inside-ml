import { createRng } from '@sim/core';
import { describe, expect, it } from 'vitest';
import { Gcn, normalisedAdjacency, plantedPartition, trainGcn } from './gnn';

const small = plantedPartition(3, 8, 0.6, 0.05, 4, 1, createRng(2));

describe('graph convolution', () => {
  it('the normalised adjacency is symmetric and keeps the square root of the degrees fixed', () => {
    const a = normalisedAdjacency(small.graph);
    const n = small.graph.nodes;
    for (let i = 0; i < n; i++)
      for (let j = 0; j < n; j++) expect(a.data[i * n + j]).toBeCloseTo(a.data[j * n + i]!, 12);
    // (A + I) 1 = d, so A_hat sqrt(d) = sqrt(d)
    const d = Float64Array.from({ length: n }, (_, i) => {
      let s = 1;
      for (const [u, v] of small.graph.edges) if (u === i || v === i) s++;
      return Math.sqrt(s);
    });
    for (let i = 0; i < n; i++) {
      let s = 0;
      for (let j = 0; j < n; j++) s += a.data[i * n + j]! * d[j]!;
      expect(s).toBeCloseTo(d[i]!, 10);
    }
  });

  it.each([true, false])(
    'backpropagation matches finite differences (graph used: %s)',
    (useGraph) => {
      const model = new Gcn(
        { layers: 3, hidden: 5, useGraph },
        4,
        3,
        normalisedAdjacency(small.graph),
        createRng(3),
      );
      const labelled = [0, 3, 9, 12, 17, 20];
      model.loss(small.features, small.labels, labelled, true);
      const h = 1e-6;
      for (const p of model.parameters) {
        const grad = Float64Array.from(p.grad);
        for (const i of [0, Math.floor(p.data.length / 2), p.data.length - 1]) {
          const saved = p.data[i]!;
          p.data[i] = saved + h;
          const up = model.loss(small.features, small.labels, labelled, false);
          p.data[i] = saved - h;
          const down = model.loss(small.features, small.labels, labelled, false);
          p.data[i] = saved;
          expect(grad[i]).toBeCloseTo((up - down) / (2 * h), 5);
        }
      }
    },
  );

  it('a node that is not connected to anything is unaffected by the graph', () => {
    const lonely = { nodes: 3, edges: [] as [number, number][] };
    const a = normalisedAdjacency(lonely);
    expect(Array.from(a.data)).toEqual([1, 0, 0, 0, 1, 0, 0, 0, 1]);
  });

  it('with a few labels the graph network beats one that ignores the graph', () => {
    const data = plantedPartition(3, 40, 0.15, 0.01, 6, 0.7, createRng(5));
    const options = { epochs: 150, learningRate: 0.05, labelledPerClass: 4, seed: 1 };
    const withGraph = trainGcn(data, { layers: 2, hidden: 16, useGraph: true }, options);
    const without = trainGcn(data, { layers: 2, hidden: 16, useGraph: false }, options);
    expect(withGraph.testAccuracy).toBeGreaterThan(without.testAccuracy + 0.1);
  });
});
