import { describe, expect, it } from 'vitest';
import { linearRange, logRange, position, thin } from './chartScale';

describe('thin', () => {
  it('keeps short series whole and long ones to the limit, with both ends', () => {
    expect(thin(5)).toEqual([0, 1, 2, 3, 4]);
    const long = thin(10_000, 100);
    expect(long).toHaveLength(100);
    expect(long[0]).toBe(0);
    expect(long[99]).toBe(9_999);
    expect([...long].sort((a, b) => a - b)).toEqual(long);
  });
});

describe('linearRange', () => {
  it('starts at zero for non-negative data and rounds the top up to a nice number', () => {
    expect(linearRange([0.2, 1.3, 0.7])).toEqual({ lo: 0, hi: 2 });
    expect(linearRange([3, 41])).toEqual({ lo: 0, hi: 50 });
  });

  it('handles negative data, fixed bounds, empty and flat series', () => {
    const r = linearRange([-0.3, 0.8]);
    expect(r.lo).toBeLessThan(-0.29);
    expect(r.hi).toBeGreaterThanOrEqual(0.8);
    expect(linearRange([0.1, 0.4], { lo: 0, hi: 1 })).toEqual({ lo: 0, hi: 1 });
    expect(linearRange([])).toEqual({ lo: 0, hi: 1 });
    expect(linearRange([NaN])).toEqual({ lo: 0, hi: 1 });
    const flat = linearRange([0, 0]);
    expect(flat.hi).toBeGreaterThan(flat.lo);
  });
});

describe('logRange', () => {
  it('covers positive values in whole decades, at least one tall', () => {
    expect(logRange([0.003, 0.5, 20])).toEqual({ lo: -3, hi: 2 });
    expect(logRange([0.5, 0.6])).toEqual({ lo: -1, hi: 0 });
    expect(logRange([0, -1])).toEqual({ lo: -1, hi: 0 });
    const exact = logRange([1, 1]);
    expect(exact.hi).toBeGreaterThan(exact.lo);
  });
});

describe('position', () => {
  it('maps into 0..1 on both kinds of axis and clamps', () => {
    expect(position(5, { lo: 0, hi: 10 }, false)).toBe(0.5);
    expect(position(-5, { lo: 0, hi: 10 }, false)).toBe(0);
    expect(position(1e-2, { lo: -4, hi: 0 }, true)).toBeCloseTo(0.5, 12);
    expect(position(0, { lo: -4, hi: 0 }, true)).toBe(0);
    expect(position(100, { lo: -4, hi: 0 }, true)).toBe(1);
  });
});
