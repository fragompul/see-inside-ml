import { describe, expect, it } from 'vitest';
import { DEFAULT_AC, LAMBDAS, runAc } from './acCore';
import { acExperiments } from './acSpec';

describe('Actor-critic lab', () => {
  for (const exp of acExperiments) {
    it(`experiment "${exp.title}" holds`, () => {
      expect(exp.claim(runAc({ ...DEFAULT_AC, ...exp.scenario }))).toBe(true);
    }, 60_000);
  }

  it('has a curve for every lambda', () => {
    const r = runAc({ ...DEFAULT_AC, episodes: 60 });
    for (const l of LAMBDAS) expect(r.curves[String(l)]).toHaveLength(60);
  });
});
