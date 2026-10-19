import { describe, expect, it } from 'vitest';
import { decadeTicks, downsample, niceMax } from '@/shared/components/LossChart';

describe('chart helpers', () => {
  it('downsample keeps short series intact and long ones bounded, ending on the last point', () => {
    expect(downsample(5)).toEqual([0, 1, 2, 3, 4]);
    const idx = downsample(10_000, 300);
    expect(idx).toHaveLength(300);
    expect(idx[0]).toBe(0);
    expect(idx[299]).toBe(9_999);
    expect(idx.every((v, i) => i === 0 || v >= (idx[i - 1] as number))).toBe(true);
  });

  it('niceMax rounds up to 1, 2, 5 × 10ⁿ', () => {
    expect(niceMax(0.73)).toBe(1);
    expect(niceMax(1.3)).toBe(2);
    expect(niceMax(3.2)).toBe(5);
    expect(niceMax(7)).toBe(10);
    expect(niceMax(0)).toBe(1);
  });

  it('decadeTicks lists powers of ten and thins wide ranges', () => {
    expect(decadeTicks(1e-3, 1)).toEqual([1e-3, 1e-2, 1e-1, 1]);
    expect(decadeTicks(1e-9, 1).length).toBeLessThanOrEqual(5);
    expect(decadeTicks(0.1, 0.1).length).toBeGreaterThanOrEqual(2);
  });
});
