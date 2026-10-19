import { describe, expect, it } from 'vitest';
import { DEFAULT_DYNA_LAB, runDyna } from './dynaCore';
import { dynaExperiments } from './dynaSpec';

describe('Dyna lab', () => {
  for (const exp of dynaExperiments) {
    it(`experiment "${exp.title}" holds`, () => {
      expect(exp.claim(runDyna({ ...DEFAULT_DYNA_LAB, ...exp.scenario }))).toBe(true);
    });
  }

  it('the curves have one point per episode and the shortest path is positive', () => {
    const r = runDyna({ ...DEFAULT_DYNA_LAB, episodes: 15 });
    expect(r.curve).toHaveLength(15);
    expect(r.baseline).toHaveLength(15);
    expect(r.shortest).toBeGreaterThan(5);
  });
});
