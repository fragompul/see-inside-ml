import { describe, expect, it } from 'vitest';
import { DEFAULT_CONTROL, runControl } from './controlCore';
import { controlExperiments } from './controlSpec';

describe('Latent control lab', () => {
  for (const exp of controlExperiments) {
    it(`experiment "${exp.title}" holds`, () => {
      expect(exp.claim(runControl({ ...DEFAULT_CONTROL, ...exp.scenario }))).toBe(true);
    }, 280_000);
  }

  it('the conditional grid has ten rows of eight images', () => {
    const r = runControl({ ...DEFAULT_CONTROL, mode: 'conditional' });
    if (r.kind !== 'conditional') throw new Error('wrong kind');
    expect(r.grid).toHaveLength(10);
    expect(r.grid[0]).toHaveLength(8);
    expect(r.grid[0]![0]).toHaveLength(256);
  });
});
