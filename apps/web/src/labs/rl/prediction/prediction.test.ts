import { describe, expect, it } from 'vitest';
import { DEFAULT_PREDICTION, runPrediction } from './predictionCore';
import { predictionExperiments } from './predictionSpec';

describe('prediction lab', () => {
  for (const exp of predictionExperiments) {
    it(`experiment "${exp.title}" holds`, () => {
      expect(exp.claim(runPrediction({ ...DEFAULT_PREDICTION, ...exp.scenario }))).toBe(true);
    });
  }

  it('every method reports a value per state and an error curve with one point per episode plus the start', () => {
    const r = runPrediction({ ...DEFAULT_PREDICTION, episodes: 12 });
    for (const m of ['mc', 'td', 'nstep', 'lambda'] as const) {
      expect(r.values[m]).toHaveLength(19);
      expect(r.curves[m]).toHaveLength(13);
    }
    expect(r.truth[9]).toBeCloseTo(0.5, 12);
  });
});
