import { describe, expect, it } from 'vitest';
import { DEFAULT_ICA, runIca } from './icaCore';
import { icaExperiments } from './icaSpec';

describe('ICA lab', () => {
  for (const exp of icaExperiments) {
    it(`experiment "${exp.title}" holds`, () => {
      expect(exp.claim(runIca({ ...DEFAULT_ICA, ...exp.scenario }))).toBe(true);
    });
  }

  it('the sources have unit variance and are really independent of the mixing', () => {
    const r = runIca(DEFAULT_ICA);
    for (let c = 0; c < 3; c++) {
      let m = 0;
      let v = 0;
      for (let i = 0; i < r.sources.rows; i++) m += r.sources.get(i, c) / r.sources.rows;
      for (let i = 0; i < r.sources.rows; i++) v += (r.sources.get(i, c) - m) ** 2 / r.sources.rows;
      expect(m).toBeCloseTo(0, 9);
      expect(v).toBeCloseTo(1, 9);
    }
  });
});
