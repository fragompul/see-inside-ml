import { describe, expect, it } from 'vitest';
import { DEFAULT_ODE, GRID, runOde } from './odeCore';
import { odeExperiments } from './odeSpec';

describe('Neural ODE lab', () => {
  for (const exp of odeExperiments) {
    it(`experiment "${exp.title}" holds`, () => {
      expect(exp.claim(runOde({ ...DEFAULT_ODE, ...exp.scenario }))).toBe(true);
    }, 200_000);
  }

  it('draws a probability map, a velocity field and the paths of 36 points', () => {
    const r = runOde({ ...DEFAULT_ODE, steps: 2 });
    expect(r.map).toHaveLength(GRID * GRID);
    expect(r.field).toHaveLength(81);
    expect(r.paths).toHaveLength(36);
    expect(r.paths[0]?.points).toHaveLength(3);
  });
});
