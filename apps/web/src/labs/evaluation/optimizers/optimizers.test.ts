import { describe, expect, it } from 'vitest';
import { DEFAULT_OPTIMIZERS, runOptimizers } from './optimizersCore';
import { optimizerExperiments } from './optimizersSpec';

describe('optimizer race', () => {
  for (const exp of optimizerExperiments) {
    it(`experiment "${exp.title}" holds`, () => {
      expect(exp.claim(runOptimizers({ ...DEFAULT_OPTIMIZERS, ...exp.scenario }))).toBe(true);
    });
  }

  it('every path starts at the same point and has one value per point', () => {
    const r = runOptimizers(DEFAULT_OPTIMIZERS);
    expect(r.paths).toHaveLength(7);
    for (const p of r.paths) {
      expect(p.points[0]).toEqual(r.start);
      expect(p.values).toHaveLength(p.points.length);
    }
  });
});
