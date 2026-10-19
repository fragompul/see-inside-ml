import { describe, expect, it } from 'vitest';
import { DEFAULT_ATTRIBUTION, EXAMPLES, runAttribution } from './attributionCore';
import { attributionExperiments } from './attributionSpec';

describe('Attribution lab', () => {
  for (const exp of attributionExperiments) {
    it(`experiment "${exp.title}" holds`, () => {
      expect(exp.claim(runAttribution({ ...DEFAULT_ATTRIBUTION, ...exp.scenario }))).toBe(true);
    }, 300_000);
  }

  it('shows the examples with their labels and one map per method', () => {
    const r = runAttribution(DEFAULT_ATTRIBUTION);
    expect(r.images).toHaveLength(EXAMPLES);
    expect(Object.keys(r.maps)).toHaveLength(4);
    expect(r.maps.integrated).toHaveLength(256);
    expect(r.accuracy).toBeGreaterThan(0.8);
  });
});
