import { describe, expect, it } from 'vitest';
import { CONFIGS, DEFAULT_DQN_LAB, runDqnLab } from './dqnCore';
import { dqnExperiments } from './dqnSpec';

describe('DQN lab', () => {
  for (const exp of dqnExperiments) {
    it(`experiment "${exp.title}" holds`, () => {
      expect(exp.claim(runDqnLab({ ...DEFAULT_DQN_LAB, ...exp.scenario }))).toBe(true);
    }, 200_000);
  }

  it('has a result and a full-length curve for every configuration', () => {
    const r = runDqnLab(DEFAULT_DQN_LAB);
    for (const c of CONFIGS) expect(r.results[c.id]?.curve).toHaveLength(DEFAULT_DQN_LAB.episodes);
  });
});
