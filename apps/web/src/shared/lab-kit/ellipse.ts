import type { Matrix } from '@sim/core';
import type { Painter } from './DataPlane';

/** The `n`-sigma contour of a 2-D Gaussian with the given mean and covariance, drawn in data coordinates. */
export function drawEllipse(
  p: Painter,
  mean: ArrayLike<number>,
  cov: Matrix,
  sigmas: number,
  style: { color: string; width?: number; alpha?: number; fill?: boolean },
): void {
  const a = cov.get(0, 0);
  const b = cov.get(0, 1);
  const d = cov.get(1, 1);
  const half = (a + d) / 2;
  const disc = Math.sqrt(Math.max(0, ((a - d) / 2) ** 2 + b * b));
  const l1 = half + disc;
  const l2 = Math.max(half - disc, 1e-12);
  // direction of the major axis
  const angle = Math.abs(b) < 1e-12 ? (a >= d ? 0 : Math.PI / 2) : Math.atan2(l1 - a, b);
  const steps = 64;
  p.ctx.save();
  p.ctx.beginPath();
  for (let s = 0; s <= steps; s++) {
    const t = (s / steps) * 2 * Math.PI;
    const u = sigmas * Math.sqrt(l1) * Math.cos(t);
    const v = sigmas * Math.sqrt(l2) * Math.sin(t);
    const x = (mean[0] as number) + u * Math.cos(angle) - v * Math.sin(angle);
    const y = (mean[1] as number) + u * Math.sin(angle) + v * Math.cos(angle);
    if (s === 0) p.ctx.moveTo(p.px(x), p.py(y));
    else p.ctx.lineTo(p.px(x), p.py(y));
  }
  p.ctx.globalAlpha = style.alpha ?? 1;
  if (style.fill) {
    p.ctx.fillStyle = style.color;
    p.ctx.globalAlpha = (style.alpha ?? 1) * 0.18;
    p.ctx.fill();
    p.ctx.globalAlpha = style.alpha ?? 1;
  }
  p.ctx.lineWidth = style.width ?? 2;
  p.ctx.strokeStyle = style.color;
  p.ctx.stroke();
  p.ctx.restore();
}
