import { describe, expect, it } from 'vitest';
import { twoInputs } from './twoInputs';

describe('two inputs: additive and interacting truths', () => {
  it('an additive truth is learned by an additive model and by hinges, not by a line', () => {
    const r = twoInputs('additive');
    expect(r.r2.additive).toBeGreaterThan(0.95);
    expect(r.r2.marsAdditive).toBeGreaterThan(0.9);
    expect(r.r2.linear).toBeLessThan(0.2);
    // the recovered shape of each input follows the true one
    const corr = (a: number[], b: number[]) => {
      const ma = a.reduce((s, v) => s + v, 0) / a.length;
      const mb = b.reduce((s, v) => s + v, 0) / b.length;
      let s = 0;
      let sa = 0;
      let sb = 0;
      a.forEach((v, i) => {
        s += (v - ma) * ((b[i] as number) - mb);
        sa += (v - ma) ** 2;
        sb += ((b[i] as number) - mb) ** 2;
      });
      return s / Math.sqrt(sa * sb);
    };
    expect(corr(r.shapes.recovered[0] as number[], r.shapes.truth[0] as number[])).toBeGreaterThan(
      0.98,
    );
    expect(corr(r.shapes.recovered[1] as number[], r.shapes.truth[1] as number[])).toBeGreaterThan(
      0.98,
    );
  });

  it('an interaction is beyond any additive model, but products of hinges can express it', () => {
    const r = twoInputs('interaction');
    expect(r.r2.additive).toBeLessThan(0.6);
    expect(r.r2.marsAdditive).toBeLessThan(0.6);
    expect(r.r2.marsInteracting).toBeGreaterThan(0.75);
    expect(r.r2.marsInteracting).toBeGreaterThan(r.r2.additive + 0.2);
  });
});
