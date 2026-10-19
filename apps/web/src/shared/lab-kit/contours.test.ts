import { describe, expect, it } from 'vitest';
import { contourSegments } from './contours';

describe('marching squares', () => {
  const n = 40;
  const field = (f: (x: number, y: number) => number) =>
    Float64Array.from({ length: n * n }, (_, i) =>
      f(((i % n) + 0.5) / n, (Math.floor(i / n) + 0.5) / n),
    );

  it('finds nothing where the field never crosses the level', () => {
    expect(
      contourSegments(
        field(() => 1),
        n,
        0,
      ),
    ).toHaveLength(0);
  });

  it('traces a circle: every segment lies on the circle of radius 0.3', () => {
    const f = field((x, y) => Math.hypot(x - 0.5, y - 0.5) - 0.3);
    const segs = contourSegments(f, n, 0);
    expect(segs.length).toBeGreaterThan(40);
    for (const [x0, y0, x1, y1] of segs) {
      for (const [x, y] of [
        [x0, y0],
        [x1, y1],
      ] as const)
        expect(Math.abs(Math.hypot(x / n - 0.5, y / n - 0.5) - 0.3)).toBeLessThan(0.02);
    }
  });

  it('a straight line x = 0.5 gives vertical segments only', () => {
    for (const [x0, , x1] of contourSegments(
      field((x) => x - 0.5),
      n,
      0,
    ))
      expect(Math.abs(x0 - x1)).toBeLessThan(1e-9);
  });
});
