import { describe, expect, it } from 'vitest';
import { DEFAULT_GAN_LAB, runGanLab } from './ganCore';
import { ganExperiments } from './ganSpec';

describe('GAN variants lab', () => {
  for (const exp of ganExperiments) {
    it(`experiment "${exp.title}" holds`, () => {
      expect(exp.claim(runGanLab({ ...DEFAULT_GAN_LAB, ...exp.scenario }))).toBe(true);
    }, 280_000);
  }

  it('records a history point every 100 updates and plain numbers only', () => {
    const r = runGanLab({ kind: 'gan', steps: 200, seed: 1 });
    expect(r.history.map((h) => h.step)).toEqual([100, 200]);
    expect(r.samples).toHaveLength(800);
    expect(r.classAccuracy).toBeNull();
  });
});
