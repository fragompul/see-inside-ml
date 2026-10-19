import { describe, expect, it } from 'vitest';
import { DEFAULT_GNN, DEPTHS, runGnnLab } from './gnnCore';
import { gnnExperiments } from './gnnSpec';

describe('Graph neural network lab', () => {
  for (const exp of gnnExperiments) {
    it(`experiment "${exp.title}" holds`, () => {
      expect(exp.claim(runGnnLab({ ...DEFAULT_GNN, ...exp.scenario }))).toBe(true);
    }, 200_000);
  }

  it('has a result for every depth and a picture of a graph of 120 nodes', () => {
    const r = runGnnLab(DEFAULT_GNN);
    expect(r.graph).toHaveLength(DEPTHS.length);
    expect(r.nodes).toHaveLength(120);
    expect(r.nodes.filter((n) => n.labelled)).toHaveLength(12);
    expect(r.edges.length).toBeGreaterThan(100);
  });
});
