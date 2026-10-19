import { describe, expect, it } from 'vitest';
import { DEFAULT_BAYES, runBayes } from './bayesCore';
import { bayesExperiments } from './bayesSpec';

describe('Bayes lab', () => {
  for (const exp of bayesExperiments) {
    it(`experiment "${exp.title}" holds`, () => {
      expect(exp.claim(runBayes(exp.scenario))).toBe(true);
    });
  }

  it('the plotted curves have unit area', () => {
    for (const model of ['coin', 'mean', 'rate'] as const) {
      const r = runBayes({ ...DEFAULT_BAYES, model, n: 20 });
      const dx = r.grid[1]! - r.grid[0]!;
      for (const curve of [r.priorCurve, r.likelihoodCurve, r.posteriorCurve])
        expect(curve.reduce((a, b) => a + b, 0) * dx).toBeGreaterThan(0.9);
    }
  });

  it('the predictive distribution of the next ten flips sums to 1', () => {
    const r = runBayes(DEFAULT_BAYES);
    expect(r.predictive!.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 10);
  });
});
