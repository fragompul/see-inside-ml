import { describe, expect, it } from 'vitest';
import { DEFAULT_FLOW_LAB, GRID, HALF, runFlowLab } from './flowCore';
import { flowExperiments } from './flowSpec';

describe('Normalising flows lab', () => {
  for (const exp of flowExperiments) {
    it(`experiment "${exp.title}" holds`, () => {
      expect(exp.claim(runFlowLab({ ...DEFAULT_FLOW_LAB, ...exp.scenario }))).toBe(true);
    }, 300_000);
  }

  it('the density on the grid is a probability density that integrates to about one', () => {
    const r = runFlowLab({ ...DEFAULT_FLOW_LAB, dataset: 'moons', layers: 4 });
    const cell = ((2 * HALF) / GRID) ** 2;
    const total = r.density.reduce((a, d) => a + d * cell, 0);
    expect(total).toBeGreaterThan(0.9);
    expect(total).toBeLessThan(1.05);
  });

  it('keeps the data at every stage, from data to latent space', () => {
    const r = runFlowLab({ ...DEFAULT_FLOW_LAB, layers: 3, steps: 100 });
    expect(r.stages).toHaveLength(4);
    expect(r.stages[0]).toHaveLength(600);
  });
});
