import { describe, expect, it } from 'vitest';
import { DEFAULT_SURVIVAL, runSurvival, simulate, trueMedian } from './survivalCore';
import { survivalExperiments } from './survivalSpec';

describe('Survival analysis lab', () => {
  for (const exp of survivalExperiments) {
    it(`experiment "${exp.title}" holds`, () => {
      expect(exp.claim(runSurvival({ ...DEFAULT_SURVIVAL, ...exp.scenario }))).toBe(true);
    });
  }

  it('simulates alternating groups and censors by the follow-up', () => {
    const d = simulate({ ...DEFAULT_SURVIVAL, n: 100 });
    expect(d.group.filter((g) => g === 1)).toHaveLength(50);
    expect(Math.max(...d.time)).toBeLessThanOrEqual(DEFAULT_SURVIVAL.followup);
    expect(trueMedian(0.1)).toBeCloseTo(6.931, 3);
  });

  it('curves cover a grid and the true curve starts at one', () => {
    const r = runSurvival(DEFAULT_SURVIVAL);
    expect(r.grid).toHaveLength(81);
    expect(r.curves[0]?.truth[0]).toBe(1);
    expect(r.curves[0]?.estimate[0]).toBe(1);
  });
});
