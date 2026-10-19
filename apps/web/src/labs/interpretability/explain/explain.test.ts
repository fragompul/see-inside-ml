import { describe, expect, it } from 'vitest';
import { DEFAULT_EXPLAIN, FEATURES, runExplain } from './explainCore';
import { explainExperiments } from './explainSpec';

describe('explanation lab', () => {
  for (const exp of explainExperiments) {
    it(`experiment "${exp.title}" holds`, () => {
      expect(exp.claim(runExplain({ ...DEFAULT_EXPLAIN, ...exp.scenario }))).toBe(true);
    }, 300_000);
  }

  it('explains ten features of a held-out patient, and the models reach plausible scores', () => {
    const r = runExplain(DEFAULT_EXPLAIN);
    expect(FEATURES).toHaveLength(10);
    expect(r.shap.values).toHaveLength(10);
    expect(r.testR2).toBeGreaterThan(0.3);
    expect(r.ice.curves.length).toBeGreaterThan(10);
  });
});
