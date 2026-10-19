import { describe, expect, it } from 'vitest';
import { DEFAULT_NMF, hoyer, runNmf } from './nmfCore';
import { nmfExperiments } from './nmfSpec';

describe('NMF lab', () => {
  for (const exp of nmfExperiments) {
    it(`experiment "${exp.title}" holds`, () => {
      expect(exp.claim(runNmf({ ...DEFAULT_NMF, ...exp.scenario }))).toBe(true);
    });
  }

  it('Hoyer sparsity is 0 for a flat row and 1 for a single spike', () => {
    expect(hoyer([1, 1, 1, 1])).toBeCloseTo(0, 9);
    expect(hoyer([0, 0, 5, 0])).toBeCloseTo(1, 9);
  });

  it('the parts and weights are non-negative and the error never increases', () => {
    const r = runNmf({ ...DEFAULT_NMF, components: 6, iterations: 40 });
    for (const v of r.model.h.data) expect(v).toBeGreaterThanOrEqual(0);
    for (const v of r.model.w.data) expect(v).toBeGreaterThanOrEqual(0);
    for (let i = 1; i < r.model.history.length; i++)
      expect(r.model.history[i]!).toBeLessThanOrEqual(r.model.history[i - 1]! * (1 + 1e-9));
  });
});
