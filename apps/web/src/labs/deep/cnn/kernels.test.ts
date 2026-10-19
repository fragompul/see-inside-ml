import { IMAGE_SIZE, convolveSingle } from '@sim/deep';
import { describe, expect, it } from 'vitest';
import { KERNEL_PRESETS, patchAt, windowOrigin } from './kernels';

const N = IMAGE_SIZE;
const preset = (id: string) => KERNEL_PRESETS.find((p) => p.id === id)?.values as number[];

/** Left half dark, right half bright: one vertical edge. */
const verticalEdge = Float64Array.from({ length: N * N }, (_, i) => (i % N >= N / 2 ? 1 : 0));

describe('kernel presets', () => {
  it('have nine values each and unique ids', () => {
    expect(new Set(KERNEL_PRESETS.map((p) => p.id)).size).toBe(KERNEL_PRESETS.length);
    KERNEL_PRESETS.forEach((p) => expect(p.values).toHaveLength(9));
  });

  it('vertical-edge kernel lights up only along a vertical edge', () => {
    const out = convolveSingle(verticalEdge, N, preset('edge-v'), 3);
    const side = N - 2;
    const at = (x: number, y: number) => out[y * side + x] as number;
    expect(at(N / 2 - 2, 5)).toBe(4); // window straddles the edge: 4 = (1+2+1) * 1
    expect(at(2, 5)).toBe(0);
    expect(at(side - 2, 5)).toBe(0);
  });

  it('horizontal-edge kernel ignores a vertical edge', () => {
    const out = convolveSingle(verticalEdge, N, preset('edge-h'), 3);
    expect(out.every((v) => v === 0)).toBe(true);
  });

  it('blur keeps flat areas and identity copies the image', () => {
    const flat = new Float64Array(N * N).fill(0.5);
    convolveSingle(flat, N, preset('blur'), 3).forEach((v) => expect(v).toBeCloseTo(0.5, 12));
    const copy = convolveSingle(verticalEdge, N, preset('identity'), 3);
    expect(copy[0]).toBe(verticalEdge[N + 1]);
  });

  it('sharpen leaves flat areas unchanged (weights sum to one)', () => {
    const sum = preset('sharpen').reduce((a, b) => a + b, 0);
    expect(sum).toBe(1);
  });
});

describe('window helpers', () => {
  it('centres the window on the pointer and never leaves the image', () => {
    expect(windowOrigin(8, 3, N)).toBe(7);
    expect(windowOrigin(-5, 3, N)).toBe(0);
    expect(windowOrigin(99, 3, N)).toBe(N - 3);
  });

  it('extracts the patch under the window in row-major order', () => {
    const image = Float64Array.from({ length: N * N }, (_, i) => i);
    expect(patchAt(image, N, 1, 2, 2)).toEqual([2 * N + 1, 2 * N + 2, 3 * N + 1, 3 * N + 2]);
  });
});
