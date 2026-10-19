import { describe, expect, it } from 'vitest';
import { DEFAULT_MOE_LAB, GRID, regionOf, runMoeLab } from './moeCore';
import { moeExperiments } from './moeSpec';

describe('Mixture of experts lab', () => {
  for (const exp of moeExperiments) {
    it(`experiment "${exp.title}" holds`, () => {
      expect(exp.claim(runMoeLab({ ...DEFAULT_MOE_LAB, ...exp.scenario }))).toBe(true);
    }, 300_000);
  }

  it('assigns the four quadrants to four regions and draws a full map of the routing', () => {
    expect([
      regionOf(-0.5, -0.5),
      regionOf(0.5, -0.5),
      regionOf(-0.5, 0.5),
      regionOf(0.5, 0.5),
    ]).toEqual([0, 1, 2, 3]);
    const r = runMoeLab(DEFAULT_MOE_LAB);
    expect(r.map).toHaveLength(GRID * GRID);
    expect(r.share.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 9);
  });
});
