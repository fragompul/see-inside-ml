import { describe, expect, it } from 'vitest';
import { BETAS, BEST_OF, DEFAULT_RLHF, runRlhf } from './rlhfCore';
import { rlhfExperiments } from './rlhfSpec';

describe('Learning from human feedback lab', () => {
  for (const exp of rlhfExperiments) {
    it(`experiment "${exp.title}" holds`, () => {
      expect(exp.claim(runRlhf({ ...DEFAULT_RLHF, ...exp.scenario }))).toBe(true);
    }, 300_000);
  }

  it('sweeps the penalty and the number of draws, with the distance from the original policy rising as the penalty falls', () => {
    const r = runRlhf(DEFAULT_RLHF);
    expect(r.kl).toHaveLength(BETAS.length);
    expect(r.bestOfTruth).toHaveLength(BEST_OF.length);
    for (let i = 1; i < r.kl.length; i++)
      expect(r.kl[i]!).toBeGreaterThanOrEqual(r.kl[i - 1]! - 1e-12);
  });
});
