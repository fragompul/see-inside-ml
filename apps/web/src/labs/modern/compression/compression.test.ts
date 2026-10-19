import { describe, expect, it } from 'vitest';
import { DEFAULT_COMPRESSION, runCompression } from './compressionCore';
import { compressionExperiments } from './compressionSpec';

describe('model compression lab', () => {
  for (const exp of compressionExperiments) {
    it(`experiment "${exp.title}" holds`, () => {
      expect(exp.claim(runCompression({ ...DEFAULT_COMPRESSION, ...exp.scenario }))).toBe(true);
    }, 300_000);
  }

  it('the teacher is accurate and the full-precision "compression" changes nothing', () => {
    const r = runCompression({ ...DEFAULT_COMPRESSION, bits: 32 });
    expect(r.teacherAccuracy).toBeGreaterThan(0.97);
    expect(r.accuracy).toBe(r.teacherAccuracy);
    expect(r.bytes).toBe(r.fullBytes);
  });
});
