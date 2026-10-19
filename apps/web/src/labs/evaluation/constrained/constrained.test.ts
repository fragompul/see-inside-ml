import { describe, expect, it } from 'vitest';
import { DEFAULT_CONSTRAINED, GRID, runConstrained } from './constrainedCore';
import { constrainedExperiments } from './constrainedSpec';

describe('Constrained optimisation lab', () => {
  for (const exp of constrainedExperiments) {
    it(`experiment "${exp.title}" holds`, () => {
      expect(exp.claim(runConstrained({ ...DEFAULT_CONSTRAINED, ...exp.scenario }))).toBe(true);
    }, 60_000);
  }

  it('draws the objective and the feasible region on a grid and compares the three methods', () => {
    const r = runConstrained(DEFAULT_CONSTRAINED);
    expect(r.surface).toHaveLength(GRID * GRID);
    expect(r.feasible.some((v) => v === 1)).toBe(true);
    expect(r.compare.penalty).toHaveLength(8);
    expect(r.compare.barrier).toHaveLength(8);
  });

  it('the barrier does not apply to a problem with an equality constraint', () => {
    const r = runConstrained({ ...DEFAULT_CONSTRAINED, problem: 'line', method: 'barrier' });
    expect(r.steps).toHaveLength(0);
    expect(r.compare.barrier).toBeNull();
  });
});
