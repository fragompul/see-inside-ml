import { describe, expect, it } from 'vitest';
import { DEFAULT_POLICY_LAB, SEEDS, runPolicy } from './policyCore';
import { policyExperiments } from './policySpec';

describe('policy-gradient lab', () => {
  for (const exp of policyExperiments) {
    it(`experiment "${exp.title}" holds`, () => {
      expect(exp.claim(runPolicy({ ...DEFAULT_POLICY_LAB, ...exp.scenario }))).toBe(true);
    }, 60_000);
  }

  it('curves have one point per episode and the comparison uses the stated number of seeds', () => {
    const r = runPolicy({ ...DEFAULT_POLICY_LAB, episodes: 50 });
    for (const c of Object.values(r.curves)) expect(c).toHaveLength(50);
    expect(r.training.returns).toHaveLength(50);
    expect(SEEDS).toBeGreaterThan(3);
  });

  it('no policy can beat the optimal return from the start', () => {
    const r = runPolicy({ ...DEFAULT_POLICY_LAB, learningRate: 0.3 });
    expect(r.finalReturn).toBeLessThanOrEqual(r.optimal + 1e-9);
  });
});
