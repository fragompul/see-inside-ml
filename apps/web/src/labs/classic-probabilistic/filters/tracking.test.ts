import { describe, expect, it } from 'vitest';
import { DEFAULT_TRACKING, runTracking } from './trackingCore';
import { trackingExperiments } from './trackingSpec';

describe('tracking lab', () => {
  for (const exp of trackingExperiments) {
    it(`experiment "${exp.title}" holds`, () => {
      expect(exp.claim(runTracking({ ...DEFAULT_TRACKING, ...exp.scenario }))).toBe(true);
    });
  }

  it('is deterministic for a seed and the first reading is never lost', () => {
    const a = runTracking({ ...DEFAULT_TRACKING, dropout: 0.9, outliers: 0.3 });
    const b = runTracking({ ...DEFAULT_TRACKING, dropout: 0.9, outliers: 0.3 });
    expect(a.readings).toEqual(b.readings);
    expect(a.readings[0]).not.toBeNull();
  });
});
