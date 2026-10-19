import { describe, expect, it } from 'vitest';
import { digitColor, fromPixel, gridCodes, plotExtent, toPixel } from './latent';

describe('plotExtent', () => {
  it('covers the data, ignores extreme outliers and never gets tiny', () => {
    // 100 points in [-1, 1] on both of two dims, plus one wild outlier
    const mu: number[] = [];
    for (let i = 0; i < 100; i++) mu.push((i / 99) * 2 - 1, 1 - (i / 99) * 2);
    mu.push(500, 500);
    const e = plotExtent(mu, 101, 2, [0, 1]);
    expect(e).toBeGreaterThanOrEqual(2);
    expect(e).toBeLessThan(5);
    const wide = Array.from({ length: 200 }, (_, i) => (i % 2 ? 6 : -6) * (1 - i / 400));
    expect(plotExtent(wide, 100, 2, [0, 1])).toBeGreaterThan(6);
  });

  it('is at least 2 for empty data and rounds to half units', () => {
    expect(plotExtent([], 0, 2, [0, 1])).toBe(2);
    const e = plotExtent([2.2, 0, -2.2, 0, 2.1, 0, -2.1, 0], 4, 2, [0, 1]);
    expect(e * 2).toBe(Math.round(e * 2));
  });

  it('only looks at the two plotted dimensions', () => {
    // dim 2 is huge but not plotted
    const mu = [0.5, 0.5, 100, -0.5, -0.5, -100];
    expect(plotExtent(mu, 2, 3, [0, 1])).toBe(2);
  });
});

describe('pixel mapping', () => {
  it('maps the window edges and centre, and inverts itself', () => {
    expect(toPixel(-3, 3, 300)).toBe(0);
    expect(toPixel(3, 3, 300)).toBe(300);
    expect(toPixel(0, 3, 300)).toBe(150);
    for (const v of [-2.5, -0.3, 0, 1.7])
      expect(fromPixel(toPixel(v, 3, 300), 3, 300)).toBeCloseTo(v, 12);
  });

  it('clamps pointers that leave the window', () => {
    expect(fromPixel(-50, 3, 300)).toBe(-3);
    expect(fromPixel(900, 3, 300)).toBe(3);
  });
});

describe('gridCodes', () => {
  it('lays an n × n grid of cell centres, top row first, holding other dims at the probe', () => {
    const codes = gridCodes(2, 4, [0, 0, 0.7], 0, 1);
    expect(codes).toHaveLength(4 * 4 * 3);
    // first cell: left column, top row
    expect(codes[0]).toBeCloseTo(-1.5, 12);
    expect(codes[1]).toBeCloseTo(1.5, 12);
    expect(codes[2]).toBe(0.7);
    // last cell: right column, bottom row
    const last = (4 * 4 - 1) * 3;
    expect(codes[last]).toBeCloseTo(1.5, 12);
    expect(codes[last + 1]).toBeCloseTo(-1.5, 12);
    expect(codes[last + 2]).toBe(0.7);
  });

  it('works for any pair of dimensions', () => {
    const codes = gridCodes(1, 2, [9, 9, 9, 9], 3, 1);
    expect(codes[0]).toBe(9);
    expect(codes[3]).toBeCloseTo(-0.5, 12);
    expect(codes[1]).toBeCloseTo(0.5, 12);
  });
});

describe('digitColor', () => {
  it('gives ten distinct colours per theme', () => {
    for (const dark of [true, false])
      expect(new Set(Array.from({ length: 10 }, (_, d) => digitColor(d, dark))).size).toBe(10);
    expect(digitColor(3, true)).not.toBe(digitColor(3, false));
  });
});
