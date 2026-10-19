import { describe, expect, it } from 'vitest';
import { DEFAULT_BANDITS, PROBLEMS, RUNS, runBandits } from './banditsCore';
import { banditExperiments } from './banditsSpec';

describe('bandit lab', () => {
  for (const exp of banditExperiments) {
    it(`experiment "${exp.title}" holds`, () => {
      expect(exp.claim(runBandits({ ...DEFAULT_BANDITS, ...exp.scenario }, 'ucb'))).toBe(true);
    }, 60_000);
  }

  it('averages over the stated number of runs and every curve is non-decreasing', () => {
    const r = runBandits({ ...DEFAULT_BANDITS, steps: 100 }, 'thompson');
    expect(RUNS).toBeGreaterThan(10);
    for (const e of r.race)
      for (let i = 1; i < e.curve.length; i++)
        expect(e.curve[i]!).toBeGreaterThanOrEqual(e.curve[i - 1]! - 1e-9);
    expect(r.single.pulls).toHaveLength(100);
  });

  it('every preset has a unique best arm', () => {
    for (const p of Object.values(PROBLEMS)) {
      const best = Math.max(...p.problem.means);
      expect(p.problem.means.filter((m) => m === best)).toHaveLength(1);
    }
  });
});
