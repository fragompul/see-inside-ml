import { describe, expect, it } from 'vitest';
import { fitPowerLaw } from './scaling';

const xs = [10, 20, 40, 80, 160, 320, 640];

describe('power-law fit', () => {
  it('recovers the exponent exactly when the irreducible part is known', () => {
    const y = xs.map((x) => 0.4 + 3 * x ** -0.35);
    const fit = fitPowerLaw(xs, y, 0.4);
    expect(fit.alpha).toBeCloseTo(0.35, 10);
    expect(fit.a).toBeCloseTo(3, 8);
    expect(fit.r2).toBeCloseTo(1, 10);
    expect(fit.predict(5000)).toBeCloseTo(0.4 + 3 * 5000 ** -0.35, 8);
  });

  it('finds the irreducible part and the exponent when neither is known', () => {
    const y = xs.map((x) => 0.4 + 3 * x ** -0.35);
    const fit = fitPowerLaw(xs, y);
    expect(fit.c).toBeGreaterThan(0.36);
    expect(fit.c).toBeLessThan(0.44);
    expect(fit.alpha).toBeGreaterThan(0.3);
    expect(fit.alpha).toBeLessThan(0.4);
    expect(fit.r2).toBeGreaterThan(0.9999);
  });

  it('a curve that is not a power law fits worse', () => {
    const y = xs.map((x) => 1 / (1 + Math.exp((Math.log(x) - 4) * 3)) + 0.1);
    expect(fitPowerLaw(xs, y, 0.1).r2).toBeLessThan(0.97);
  });
});
