import { describe, expect, it } from 'vitest';
import { DEFAULT_HMM, casinoModel, runHmm } from './hmmCore';
import { hmmExperiments } from './hmmSpec';

describe('HMM lab', () => {
  for (const exp of hmmExperiments) {
    it(`experiment "${exp.title}" holds`, () => {
      expect(exp.claim(runHmm({ ...DEFAULT_HMM, ...exp.scenario }))).toBe(true);
    });
  }

  it('the casino model is a proper probability model', () => {
    const m = casinoModel(0.9, 0.4);
    for (const row of [...m.transition, ...m.emission])
      expect(row.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 12);
    expect(m.emission[1]![5]).toBeCloseTo(0.4, 12);
  });

  it('is deterministic for a seed', () => {
    expect(runHmm(DEFAULT_HMM).rolls).toEqual(runHmm(DEFAULT_HMM).rolls);
  });
});
