import { describe, expect, it } from 'vitest';
import { DEFAULT_CONV_ARITH, buildStack, runConvArith } from './convArithCore';
import { convArithExperiments } from './convArithSpec';

describe('Convolution arithmetic lab', () => {
  for (const exp of convArithExperiments) {
    it(`experiment "${exp.title}" holds`, () => {
      expect(exp.claim(runConvArith({ ...DEFAULT_CONV_ARITH, ...exp.scenario }))).toBe(true);
    });
  }

  it('every design keeps at least one value at the end for every setting', () => {
    for (const design of ['plain', 'pooled', 'strided', 'dilated'] as const)
      for (const depth of [1, 2, 3, 4, 6, 8])
        for (const input of [64, 128, 224]) {
          const r = runConvArith({ design, depth, kernel: 3, input });
          expect(r.finalSize).toBeGreaterThanOrEqual(2);
          expect(buildStack({ design, depth, kernel: 3, input }).length).toBeGreaterThanOrEqual(
            depth,
          );
        }
  });
});
