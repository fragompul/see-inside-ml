import { describe, expect, it } from 'vitest';
import { DEFAULT_MCMC, TARGETS, runMcmc, type TargetId } from './mcmcCore';
import { mcmcExperiments } from './mcmcSpec';

describe('MCMC lab', () => {
  for (const exp of mcmcExperiments) {
    it(`experiment "${exp.title}" holds`, () => {
      expect(exp.claim(runMcmc({ ...DEFAULT_MCMC, ...exp.scenario }))).toBe(true);
    });
  }

  it.each(['gaussian', 'banana', 'modes'] as TargetId[])(
    'the gradient of %s matches finite differences',
    (id) => {
      const t = TARGETS[id];
      const x = [0.7, 0.4];
      const g = t.gradient(x);
      for (let d = 0; d < 2; d++) {
        const h = 1e-6;
        const up = [...x];
        const down = [...x];
        up[d] = (up[d] as number) + h;
        down[d] = (down[d] as number) - h;
        expect(g[d]).toBeCloseTo((t.logDensity(up) - t.logDensity(down)) / (2 * h), 5);
      }
    },
  );
});
