import { describe, expect, it } from 'vitest';
import { DEFAULT_TUNING, RACE_REPEATS, runTuning } from './tuningCore';
import { tuningExperiments } from './tuningSpec';

describe('tuning lab', () => {
  for (const exp of tuningExperiments) {
    it(`experiment "${exp.title}" holds`, () => {
      expect(exp.claim(runTuning({ ...DEFAULT_TUNING, ...exp.scenario }))).toBe(true);
    }, 300_000);
  }

  it('the race covers the four methods with the stated number of repetitions', () => {
    const r = runTuning({ ...DEFAULT_TUNING, budget: 6 });
    expect(r.race.map((e) => e.method)).toEqual(['grid', 'random', 'halving', 'bayesian']);
    for (const e of r.race) {
      expect(e.meanRegret).toBeGreaterThanOrEqual(0);
      expect(e.success * RACE_REPEATS).toBeCloseTo(Math.round(e.success * RACE_REPEATS), 9);
    }
    expect(r.single.incumbent).toHaveLength(r.single.evaluations.length);
  });
});
