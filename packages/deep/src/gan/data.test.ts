import { describe, expect, it } from 'vitest';
import { createRng } from '@sim/core';
import { GAN_DATASETS, MODE_CENTERS, MODE_STD, hasModes, sampleReal } from './datasets';
import { fractionNearData, modesCovered, slicedWasserstein } from './metrics';

describe('datasets', () => {
  it.each(GAN_DATASETS)('%s: reproducible, finite and inside the plot window', (dataset) => {
    const a = sampleReal(dataset, 500, createRng(3));
    const b = sampleReal(dataset, 500, createRng(3));
    expect(a).toEqual(b);
    expect(a).toHaveLength(1000);
    expect(a.every((v) => Number.isFinite(v) && Math.abs(v) < 3.5)).toBe(true);
    expect(sampleReal(dataset, 500, createRng(4))).not.toEqual(a);
  });

  it('eight Gaussians sit on a circle of radius 2 and only those have modes', () => {
    MODE_CENTERS.forEach((c) => expect(Math.hypot(c[0], c[1])).toBeCloseTo(2, 12));
    const points = sampleReal('gaussians8', 800, createRng(1));
    expect(
      fractionNearData(points, Float64Array.from(MODE_CENTERS.flat()), 5 * MODE_STD),
    ).toBeGreaterThan(0.99);
    expect(hasModes('gaussians8')).toBe(true);
    expect(hasModes('ring')).toBe(false);
  });

  it('the ring is thin and the two moons are separated', () => {
    const ring = sampleReal('ring', 500, createRng(2));
    for (let i = 0; i < 500; i++)
      expect(
        Math.abs(Math.hypot(ring[2 * i] as number, ring[2 * i + 1] as number) - 2),
      ).toBeLessThan(0.5);
    const moons = sampleReal('moons', 2000, createRng(2));
    // both halves occur: some points above and some below the middle line
    let above = 0;
    for (let i = 0; i < 2000; i++) if ((moons[2 * i + 1] as number) > 0) above += 1;
    expect(above).toBeGreaterThan(600);
    expect(above).toBeLessThan(1400);
  });
});

describe('metrics', () => {
  it('sliced Wasserstein is 0 for equal clouds, small for same-distribution samples and large for different ones', () => {
    const a = sampleReal('gaussian', 400, createRng(1));
    expect(slicedWasserstein(a, a)).toBe(0);
    const same = slicedWasserstein(a, sampleReal('gaussian', 400, createRng(2)));
    const different = slicedWasserstein(a, sampleReal('ring', 400, createRng(2)));
    expect(same).toBeLessThan(0.1);
    expect(different).toBeGreaterThan(3 * same);
    // shifting a cloud by (1, 0) moves it by about E|cos θ| = 2/π per projection on average
    const shifted = a.map((v, i) => (i % 2 === 0 ? v + 1 : v));
    expect(slicedWasserstein(a, shifted)).toBeCloseTo(2 / Math.PI, 1);
  });

  it('a collapsed cloud is far from the real one even if every point is on the data', () => {
    const real = sampleReal('gaussians8', 400, createRng(1));
    const collapsed = new Float64Array(800);
    for (let i = 0; i < 400; i++) {
      collapsed[2 * i] = (MODE_CENTERS[0] as readonly number[])[0] as number;
      collapsed[2 * i + 1] = (MODE_CENTERS[0] as readonly number[])[1] as number;
    }
    expect(fractionNearData(collapsed, real)).toBe(1);
    expect(modesCovered(collapsed).count).toBe(1);
    expect(slicedWasserstein(collapsed, real)).toBeGreaterThan(0.5);
  });

  it('counts modes only above the share threshold', () => {
    const samples = new Float64Array(400);
    for (let i = 0; i < 100; i++) {
      samples[2 * i] = (MODE_CENTERS[0] as readonly number[])[0] as number;
      samples[2 * i + 1] = (MODE_CENTERS[0] as readonly number[])[1] as number;
    }
    for (let i = 100; i < 199; i++) {
      samples[2 * i] = (MODE_CENTERS[4] as readonly number[])[0] as number;
      samples[2 * i + 1] = (MODE_CENTERS[4] as readonly number[])[1] as number;
    }
    // one stray point in a third mode: 0.5% of the samples, below the 2% default
    samples[398] = (MODE_CENTERS[2] as readonly number[])[0] as number;
    samples[399] = (MODE_CENTERS[2] as readonly number[])[1] as number;
    const result = modesCovered(samples);
    expect(result.count).toBe(2);
    expect(result.covered[0]).toBe(true);
    expect(result.covered[4]).toBe(true);
    expect(result.covered[2]).toBe(false);
    expect(modesCovered(samples, 0.001).count).toBe(3);
  });

  it('rejects mismatched clouds', () => {
    expect(() => slicedWasserstein(new Float64Array(4), new Float64Array(6))).toThrow(RangeError);
    expect(() => slicedWasserstein(new Float64Array(0), new Float64Array(0))).toThrow(RangeError);
  });
});
