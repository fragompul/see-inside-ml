import { describe, expect, it } from 'vitest';
import { DEFAULT_GUIDANCE, STEP_COUNTS, WEIGHTS, runGuidance } from './guidanceCore';
import { guidanceExperiments } from './guidanceSpec';

describe('Guidance and flow matching lab', () => {
  for (const exp of guidanceExperiments) {
    it(`experiment "${exp.title}" holds`, () => {
      expect(exp.claim(runGuidance({ ...DEFAULT_GUIDANCE, ...exp.scenario }))).toBe(true);
    }, 280_000);
  }

  it('has a result for every weight and every number of steps', () => {
    const r = runGuidance(DEFAULT_GUIDANCE);
    expect(r.accuracy).toHaveLength(WEIGHTS.length);
    expect(r.flowDistance).toHaveLength(STEP_COUNTS.length);
    expect(r.drawn).toHaveLength(400);
  });
});
