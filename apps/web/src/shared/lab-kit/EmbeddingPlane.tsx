import type { Matrix } from '@sim/core';
import { useEffect, useMemo, useRef, type PointerEvent } from 'react';
import { cn } from '@/design-system';
import { mix, usePalette } from '@/shared/canvas/colors';

/**
 * A scatter of 2-D coordinates, coloured by class or by a continuous value, with a hover that names
 * the nearest point and rings a set of "related" points (e.g. its neighbours in the original space).
 */
export function EmbeddingPlane({
  points,
  values,
  colour,
  related,
  focus,
  onFocus,
  width = 520,
  height = 400,
  ariaLabel,
  className,
}: {
  points: Matrix;
  /** Class index or position in [0, 1], one per point. */
  values: ArrayLike<number>;
  colour: 'continuous' | 'classes';
  /** Points to ring (e.g. the original-space neighbours of the focused point). */
  related?: ReadonlySet<number> | null;
  /** The point being looked at. */
  focus?: number | null;
  onFocus?: (index: number | null) => void;
  width?: number;
  height?: number;
  ariaLabel: string;
  className?: string;
}) {
  const palette = usePalette();
  const ref = useRef<HTMLCanvasElement>(null);

  // data → pixel mapping that fits the whole cloud with a margin
  const frame = useMemo(() => {
    let x0 = Infinity;
    let x1 = -Infinity;
    let y0 = Infinity;
    let y1 = -Infinity;
    for (let i = 0; i < points.rows; i++) {
      x0 = Math.min(x0, points.get(i, 0));
      x1 = Math.max(x1, points.get(i, 0));
      y0 = Math.min(y0, points.get(i, 1));
      y1 = Math.max(y1, points.get(i, 1));
    }
    if (!Number.isFinite(x0)) [x0, x1, y0, y1] = [-1, 1, -1, 1];
    const span = Math.max(x1 - x0, y1 - y0, 1e-9);
    const pad = 22;
    const scale = Math.min((width - 2 * pad) / span, (height - 2 * pad) / span);
    const cx = (x0 + x1) / 2;
    const cy = (y0 + y1) / 2;
    return {
      px: (v: number) => width / 2 + (v - cx) * scale,
      py: (v: number) => height / 2 - (v - cy) * scale,
    };
  }, [points, width, height]);

  const colorOf = (i: number): string => {
    const v = values[i] as number;
    if (colour === 'classes') return palette.css.classes[v % palette.css.classes.length] as string;
    const t = Math.min(1, Math.max(0, v));
    const c =
      t < 0.5
        ? mix(palette.neg, palette.brand, t * 2)
        : mix(palette.brand, palette.pos, (t - 0.5) * 2);
    return `rgb(${c[0]}, ${c[1]}, ${c[2]})`;
  };

  useEffect(() => {
    const canvas = ref.current;
    const ctx = canvas?.getContext('2d');
    if (!canvas || !ctx) return;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    canvas.width = width * dpr;
    canvas.height = height * dpr;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.fillStyle = palette.css.surface;
    ctx.fillRect(0, 0, width, height);
    const dim = focus != null && focus >= 0;
    for (let i = 0; i < points.rows; i++) {
      const x = frame.px(points.get(i, 0));
      const y = frame.py(points.get(i, 1));
      ctx.beginPath();
      ctx.arc(x, y, 3.6, 0, Math.PI * 2);
      ctx.globalAlpha = dim && !related?.has(i) && i !== focus ? 0.35 : 1;
      ctx.fillStyle = colorOf(i);
      ctx.fill();
      ctx.lineWidth = 0.8;
      ctx.strokeStyle = palette.css.surface;
      ctx.stroke();
    }
    ctx.globalAlpha = 1;
    if (related) {
      ctx.lineWidth = 2;
      ctx.strokeStyle = palette.css.fg;
      for (const i of related) {
        ctx.beginPath();
        ctx.arc(frame.px(points.get(i, 0)), frame.py(points.get(i, 1)), 7, 0, Math.PI * 2);
        ctx.stroke();
      }
    }
    if (dim) {
      ctx.lineWidth = 2.5;
      ctx.strokeStyle = palette.css.brand;
      ctx.beginPath();
      ctx.arc(frame.px(points.get(focus, 0)), frame.py(points.get(focus, 1)), 9, 0, Math.PI * 2);
      ctx.stroke();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [points, values, colour, related, focus, frame, palette, width, height]);

  const nearest = (event: PointerEvent<HTMLCanvasElement>): number | null => {
    const rect = event.currentTarget.getBoundingClientRect();
    const mx = ((event.clientX - rect.left) / rect.width) * width;
    const my = ((event.clientY - rect.top) / rect.height) * height;
    let best: number | null = null;
    let bestD = 14;
    for (let i = 0; i < points.rows; i++) {
      const d = Math.hypot(frame.px(points.get(i, 0)) - mx, frame.py(points.get(i, 1)) - my);
      if (d < bestD) {
        bestD = d;
        best = i;
      }
    }
    return best;
  };

  return (
    <canvas
      ref={ref}
      role="img"
      aria-label={ariaLabel}
      style={{ width, height: 'auto', aspectRatio: `${width} / ${height}` }}
      className={cn(
        'block max-w-full cursor-crosshair rounded-xl border border-line-strong',
        className,
      )}
      onPointerMove={(e) => onFocus?.(nearest(e))}
      onPointerLeave={() => onFocus?.(null)}
    />
  );
}
