import { useEffect, useRef } from 'react';
import { usePalette } from '@/shared/canvas/colors';
import { toPixel } from '@/shared/canvas/plane';
import { EXTENT } from '../store';

const SIZE = 480;

export type CloudColor = 'brand' | 'neg' | 'pos' | 'fg';

export interface Cloud {
  points: Float64Array;
  color: CloudColor;
  /** Draw hollow rings instead of filled dots. */
  hollow?: boolean;
  alpha?: number;
  radius?: number;
}

interface Props {
  clouds: Cloud[];
  /** Short arrows at `points` pointing along `vectors` (both `n × 2`). */
  arrows?: { points: Float64Array; vectors: Float64Array } | null;
  /** Paths of the first `count` points of a trajectory up to step `upto`. */
  trails?: { data: Float64Array; points: number; upto: number; count: number } | null;
  ariaLabel: string;
}

/** A 2-D plot window for point clouds, a vector field and the paths points take through time. */
export function PlaneCanvas({ clouds, arrows, trails, ariaLabel }: Props) {
  const palette = usePalette();
  const ref = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = ref.current;
    const ctx = canvas?.getContext('2d');
    if (!canvas || !ctx) return;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    canvas.width = SIZE * dpr;
    canvas.height = SIZE * dpr;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.fillStyle = palette.css.surface;
    ctx.fillRect(0, 0, SIZE, SIZE);

    const X = (v: number) => toPixel(v, EXTENT, SIZE);
    const Y = (v: number) => SIZE - toPixel(v, EXTENT, SIZE);
    const color = (c: CloudColor) =>
      ({
        brand: palette.css.brand,
        neg: palette.css.neg,
        pos: palette.css.pos,
        fg: palette.css.fg,
      })[c];

    ctx.strokeStyle = palette.line;
    ctx.globalAlpha = 0.45;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(X(0), 0);
    ctx.lineTo(X(0), SIZE);
    ctx.moveTo(0, Y(0));
    ctx.lineTo(SIZE, Y(0));
    ctx.stroke();
    ctx.globalAlpha = 1;

    if (trails && trails.upto > 0) {
      ctx.strokeStyle = palette.css.brand;
      ctx.globalAlpha = 0.3;
      ctx.lineWidth = 1.1;
      const n = trails.points;
      for (let p = 0; p < Math.min(trails.count, n); p++) {
        ctx.beginPath();
        for (let k = 0; k <= trails.upto; k++) {
          const x = X(trails.data[(k * n + p) * 2] as number);
          const y = Y(trails.data[(k * n + p) * 2 + 1] as number);
          if (k === 0) ctx.moveTo(x, y);
          else ctx.lineTo(x, y);
        }
        ctx.stroke();
      }
      ctx.globalAlpha = 1;
    }

    if (arrows) {
      ctx.strokeStyle = palette.css.pos;
      ctx.fillStyle = palette.css.pos;
      ctx.globalAlpha = 0.8;
      ctx.lineWidth = 1.5;
      const count = arrows.points.length / 2;
      // scale so the longest arrow is about one cell long
      let longest = 1e-9;
      for (let i = 0; i < count; i++)
        longest = Math.max(
          longest,
          Math.hypot(arrows.vectors[2 * i] as number, arrows.vectors[2 * i + 1] as number),
        );
      const cell = SIZE / Math.sqrt(count);
      for (let i = 0; i < count; i++) {
        const x = X(arrows.points[2 * i] as number);
        const y = Y(arrows.points[2 * i + 1] as number);
        const dx = ((arrows.vectors[2 * i] as number) / longest) * cell * 0.9;
        const dy = (-(arrows.vectors[2 * i + 1] as number) / longest) * cell * 0.9;
        const len = Math.hypot(dx, dy);
        if (len < 2) continue;
        ctx.beginPath();
        ctx.moveTo(x - dx / 2, y - dy / 2);
        ctx.lineTo(x + dx / 2, y + dy / 2);
        ctx.stroke();
        const ang = Math.atan2(dy, dx);
        const head = Math.min(6, len * 0.4);
        ctx.beginPath();
        ctx.moveTo(x + dx / 2, y + dy / 2);
        ctx.lineTo(
          x + dx / 2 - head * Math.cos(ang - 0.5),
          y + dy / 2 - head * Math.sin(ang - 0.5),
        );
        ctx.lineTo(
          x + dx / 2 - head * Math.cos(ang + 0.5),
          y + dy / 2 - head * Math.sin(ang + 0.5),
        );
        ctx.closePath();
        ctx.fill();
      }
      ctx.globalAlpha = 1;
    }

    for (const cloud of clouds) {
      ctx.globalAlpha = cloud.alpha ?? 0.9;
      const r = cloud.radius ?? 3.2;
      if (cloud.hollow) {
        ctx.strokeStyle = color(cloud.color);
        ctx.lineWidth = 1.3;
      } else {
        ctx.fillStyle = color(cloud.color);
      }
      for (let i = 0; i < cloud.points.length / 2; i++) {
        ctx.beginPath();
        ctx.arc(
          X(cloud.points[2 * i] as number),
          Y(cloud.points[2 * i + 1] as number),
          r,
          0,
          Math.PI * 2,
        );
        if (cloud.hollow) ctx.stroke();
        else ctx.fill();
      }
      ctx.globalAlpha = 1;
    }
  }, [clouds, arrows, trails, palette]);

  return (
    <canvas
      ref={ref}
      role="img"
      aria-label={ariaLabel}
      style={{ width: SIZE, height: 'auto', aspectRatio: '1' }}
      className="block max-w-full rounded-xl border border-line-strong"
    />
  );
}
