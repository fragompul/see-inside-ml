import { describe, expect, it } from 'vitest';
import { DEFAULT_TOOLBOX_LAB, runToolbox } from './toolboxCore';
import { toolboxExperiments } from './toolboxSpec';

describe('training toolbox lab', () => {
  for (const exp of toolboxExperiments) {
    it(`experiment "${exp.title}" holds`, () => {
      expect(exp.claim(runToolbox({ ...DEFAULT_TOOLBOX_LAB, ...exp.scenario }))).toBe(true);
    }, 60_000);
  }

  it('the default run learns the moons and reports a decision grid', () => {
    const r = runToolbox(DEFAULT_TOOLBOX_LAB);
    expect(r.testAccuracy).toBeGreaterThan(0.9);
    expect(r.grid.labels).toHaveLength(r.grid.size * r.grid.size);
    expect(new Set(r.grid.labels).size).toBe(2);
  });

  it('is deterministic for a seed', () => {
    const a = runToolbox({ ...DEFAULT_TOOLBOX_LAB, epochs: 20 });
    const b = runToolbox({ ...DEFAULT_TOOLBOX_LAB, epochs: 20 });
    expect(a.testLoss).toBe(b.testLoss);
  });
});
