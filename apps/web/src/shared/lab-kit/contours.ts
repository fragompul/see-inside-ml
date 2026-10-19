import type { Painter } from './DataPlane';

/**
 * Marching squares: the line segments where a scalar field sampled on an `n × n` grid crosses
 * `level`. Coordinates are in cell units (0 … n); the field is `values[row * n + col]`.
 */
export function contourSegments(
  values: ArrayLike<number>,
  n: number,
  level: number,
): [number, number, number, number][] {
  const out: [number, number, number, number][] = [];
  const v = (r: number, c: number) => values[r * n + c] as number;
  const lerp = (a: number, b: number) => (a === b ? 0.5 : (level - a) / (b - a));
  for (let r = 0; r < n - 1; r++)
    for (let c = 0; c < n - 1; c++) {
      const tl = v(r, c);
      const tr = v(r, c + 1);
      const br = v(r + 1, c + 1);
      const bl = v(r + 1, c);
      const code =
        (tl > level ? 8 : 0) | (tr > level ? 4 : 0) | (br > level ? 2 : 0) | (bl > level ? 1 : 0);
      if (code === 0 || code === 15) continue;
      // crossing points on the four edges (cell centres sit at +0.5)
      const top: [number, number] = [c + 0.5 + lerp(tl, tr), r + 0.5];
      const right: [number, number] = [c + 1.5, r + 0.5 + lerp(tr, br)];
      const bottom: [number, number] = [c + 0.5 + lerp(bl, br), r + 1.5];
      const left: [number, number] = [c + 0.5, r + 0.5 + lerp(tl, bl)];
      const seg = (a: [number, number], b: [number, number]) => out.push([a[0], a[1], b[0], b[1]]);
      switch (code) {
        case 1:
        case 14:
          seg(left, bottom);
          break;
        case 2:
        case 13:
          seg(bottom, right);
          break;
        case 3:
        case 12:
          seg(left, right);
          break;
        case 4:
        case 11:
          seg(top, right);
          break;
        case 6:
        case 9:
          seg(top, bottom);
          break;
        case 7:
        case 8:
          seg(left, top);
          break;
        case 5:
          seg(left, top);
          seg(bottom, right);
          break;
        case 10:
          seg(top, right);
          seg(left, bottom);
          break;
      }
    }
  return out;
}

/** Draws the `level` contour of a grid field laid over `bounds`, in data coordinates. */
export function drawContour(
  p: Painter,
  values: ArrayLike<number>,
  n: number,
  bounds: { x0: number; x1: number; y0: number; y1: number },
  level: number,
  style: { color: string; width?: number; dash?: number[] },
): void {
  const toX = (cx: number) => bounds.x0 + (cx / n) * (bounds.x1 - bounds.x0);
  const toY = (cy: number) => bounds.y1 - (cy / n) * (bounds.y1 - bounds.y0);
  p.ctx.save();
  p.ctx.strokeStyle = style.color;
  p.ctx.lineWidth = style.width ?? 1.6;
  p.ctx.setLineDash(style.dash ?? []);
  p.ctx.beginPath();
  for (const [x0, y0, x1, y1] of contourSegments(values, n, level)) {
    p.ctx.moveTo(p.px(toX(x0)), p.py(toY(y0)));
    p.ctx.lineTo(p.px(toX(x1)), p.py(toY(y1)));
  }
  p.ctx.stroke();
  p.ctx.restore();
}
