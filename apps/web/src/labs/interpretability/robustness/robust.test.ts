import { describe, expect, it } from 'vitest';
import { BUDGETS, DEFAULT_ROBUSTNESS, runRobustness } from './robustCore';
import { robustExperiments } from './robustSpec';

describe('Adversarial robustness lab', () => {
  for (const exp of robustExperiments) {
    it(`experiment "${exp.title}" holds`, () => {
      expect(exp.claim(runRobustness({ ...DEFAULT_ROBUSTNESS, ...exp.scenario }))).toBe(true);
    }, 200_000);
  }

  it('has a curve point for every budget and the pictures differ by at most the budget', () => {
    const r = runRobustness(DEFAULT_ROBUSTNESS);
    expect(r.chosen.fgsm).toHaveLength(BUDGETS.length);
    for (const e of r.examples) {
      const biggest = Math.max(
        ...e.original.map((v, i) => Math.abs(v - (e.attacked[i] as number))),
      );
      expect(biggest).toBeLessThanOrEqual(DEFAULT_ROBUSTNESS.shownEpsilon + 1e-9);
    }
  });
});
