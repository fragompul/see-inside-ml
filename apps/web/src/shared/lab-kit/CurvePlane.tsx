import type { Table } from '@sim/core';
import { useEffect, useMemo, useRef, type PointerEvent } from 'react';
import { cn } from '@/design-system';
import { usePalette } from '@/shared/canvas/colors';
import type { EditTool, Painter } from './DataPlane';
import { nearestPoint } from './tableEdit';

export interface Curve {
  xs: ArrayLike<number>;
  ys: ArrayLike<number>;
  /** CSS colour; defaults to the brand colour. */
  color?: string;
  dashed?: boolean;
  width?: number;
}

export interface Band {
  xs: ArrayLike<number>;
  lo: ArrayLike<number>;
  hi: ArrayLike<number>;
  color?: string;
  alpha?: number;
}

export interface CurvePlaneProps {
  /** One-feature table. */
  all: Table;
  testMask: Uint8Array;
  curves?: Curve[];
  bands?: Band[];
  /** Predictions at the rows of `all`: draws each error as a line, or as a square of that size. */
  residuals?: { pred: ArrayLike<number>; mode: 'lines' | 'squares' } | null;
  xRange?: [number, number];
  yRange?: [number, number];
  highlight?: ReadonlySet<number> | null;
  tool?: EditTool;
  onAdd?: (x: number, y: number) => void;
  onMove?: (index: number, x: number, y: number) => void;
  onErase?: (index: number) => void;
  underlay?: (p: Painter) => void;
  overlay?: (p: Painter) => void;
  width?: number;
  height?: number;
  ariaLabel: string;
  className?: string;
}

/** A 1-D regression plot: points, fitted curves, uncertainty bands and the errors, editable with the pointer. */
export function CurvePlane({
  all,
  testMask,
  curves = [],
  bands = [],
  residuals,
  xRange,
  yRange,
  highlight,
  tool = 'none',
  onAdd,
  onMove,
  onErase,
  underlay,
  overlay,
  width = 560,
  height = 360,
  ariaLabel,
  className,
}: CurvePlaneProps) {
  const palette = usePalette();
  const ref = useRef<HTMLCanvasElement>(null);
  const dragging = useRef<number | null>(null);

  const view = useMemo(() => {
    let x0 = Infinity;
    let x1 = -Infinity;
    let y0 = Infinity;
    let y1 = -Infinity;
    for (let i = 0; i < all.x.rows; i++) {
      x0 = Math.min(x0, all.x.get(i, 0));
      x1 = Math.max(x1, all.x.get(i, 0));
      y0 = Math.min(y0, all.y[i] as number);
      y1 = Math.max(y1, all.y[i] as number);
    }
    if (!Number.isFinite(x0)) [x0, x1, y0, y1] = [-1, 1, -1, 1];
    const padX = (x1 - x0 || 1) * 0.06;
    const padY = (y1 - y0 || 1) * 0.15;
    const [ax0, ax1] = xRange ?? [x0 - padX, x1 + padX];
    const [ay0, ay1] = yRange ?? [y0 - padY, y1 + padY];
    return {
      x0: ax0,
      x1: ax1,
      y0: ay0,
      y1: ay1,
      sx: width / (ax1 - ax0),
      sy: height / (ay1 - ay0),
    };
  }, [all, xRange, yRange, width, height]);

  const pixels = useMemo(() => {
    const px = new Float64Array(all.x.rows);
    const py = new Float64Array(all.x.rows);
    for (let i = 0; i < all.x.rows; i++) {
      px[i] = (all.x.get(i, 0) - view.x0) * view.sx;
      py[i] = height - ((all.y[i] as number) - view.y0) * view.sy;
    }
    return { px, py };
  }, [all, view, height]);

  useEffect(() => {
    const canvas = ref.current;
    const ctx = canvas?.getContext('2d');
    if (!canvas || !ctx) return;
    const toX = (v: number) => (v - view.x0) * view.sx;
    const toY = (v: number) => height - (v - view.y0) * view.sy;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    canvas.width = width * dpr;
    canvas.height = height * dpr;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.fillStyle = palette.css.surface;
    ctx.fillRect(0, 0, width, height);

    // faint axes through zero
    ctx.strokeStyle = palette.line;
    ctx.globalAlpha = 0.4;
    ctx.lineWidth = 1;
    ctx.beginPath();
    if (view.x0 < 0 && view.x1 > 0) {
      ctx.moveTo(toX(0), 0);
      ctx.lineTo(toX(0), height);
    }
    if (view.y0 < 0 && view.y1 > 0) {
      ctx.moveTo(0, toY(0));
      ctx.lineTo(width, toY(0));
    }
    ctx.stroke();
    ctx.globalAlpha = 1;

    const painter: Painter = { ctx, px: toX, py: toY, palette, width, height };
    for (const b of bands) {
      ctx.fillStyle = b.color ?? palette.css.brand;
      ctx.globalAlpha = b.alpha ?? 0.18;
      ctx.beginPath();
      for (let i = 0; i < b.xs.length; i++)
        ctx.lineTo(toX(b.xs[i] as number), toY(b.hi[i] as number));
      for (let i = b.xs.length - 1; i >= 0; i--)
        ctx.lineTo(toX(b.xs[i] as number), toY(b.lo[i] as number));
      ctx.closePath();
      ctx.fill();
      ctx.globalAlpha = 1;
    }
    underlay?.(painter);

    if (residuals) {
      ctx.strokeStyle = palette.css.neg;
      ctx.fillStyle = palette.css.neg;
      for (let i = 0; i < all.x.rows; i++) {
        const x = pixels.px[i] as number;
        const yTrue = pixels.py[i] as number;
        const yHat = toY(residuals.pred[i] as number);
        if (residuals.mode === 'lines') {
          ctx.globalAlpha = 0.55;
          ctx.lineWidth = 1.2;
          ctx.beginPath();
          ctx.moveTo(x, yTrue);
          ctx.lineTo(x, yHat);
          ctx.stroke();
        } else {
          // a square whose side is the error: its area is the squared error being minimised
          const side = Math.abs(yTrue - yHat);
          ctx.globalAlpha = 0.14;
          ctx.fillRect(x, Math.min(yTrue, yHat), side, side);
          ctx.globalAlpha = 0.5;
          ctx.lineWidth = 1;
          ctx.strokeRect(x, Math.min(yTrue, yHat), side, side);
        }
        ctx.globalAlpha = 1;
      }
    }

    for (const c of curves) {
      ctx.strokeStyle = c.color ?? palette.css.brand;
      ctx.lineWidth = c.width ?? 2.6;
      ctx.lineJoin = 'round';
      ctx.setLineDash(c.dashed ? [7, 6] : []);
      ctx.beginPath();
      for (let i = 0; i < c.xs.length; i++) {
        const x = toX(c.xs[i] as number);
        const y = toY(c.ys[i] as number);
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.stroke();
      ctx.setLineDash([]);
    }

    for (let i = 0; i < all.x.rows; i++) {
      const x = pixels.px[i] as number;
      const y = pixels.py[i] as number;
      ctx.beginPath();
      ctx.arc(x, y, 4.6, 0, Math.PI * 2);
      if (testMask[i] === 1) {
        ctx.fillStyle = palette.css.surface;
        ctx.fill();
        ctx.lineWidth = 2;
        ctx.strokeStyle = palette.css.fg;
        ctx.stroke();
      } else {
        ctx.fillStyle = palette.css.fg;
        ctx.globalAlpha = 0.85;
        ctx.fill();
        ctx.globalAlpha = 1;
      }
      if (highlight?.has(i)) {
        ctx.beginPath();
        ctx.arc(x, y, 10, 0, Math.PI * 2);
        ctx.lineWidth = 2;
        ctx.strokeStyle = palette.css.brand;
        ctx.stroke();
      }
    }
    overlay?.(painter);
  }, [
    all,
    testMask,
    curves,
    bands,
    residuals,
    highlight,
    pixels,
    view,
    palette,
    underlay,
    overlay,
    width,
    height,
  ]);

  const locate = (event: PointerEvent<HTMLCanvasElement>) => {
    const rect = event.currentTarget.getBoundingClientRect();
    const cx = ((event.clientX - rect.left) / rect.width) * width;
    const cy = ((event.clientY - rect.top) / rect.height) * height;
    return { cx, cy, x: view.x0 + cx / view.sx, y: view.y0 + (height - cy) / view.sy };
  };

  return (
    <canvas
      ref={ref}
      role="img"
      aria-label={ariaLabel}
      style={{
        width,
        height: 'auto',
        aspectRatio: `${width} / ${height}`,
        touchAction: tool === 'none' ? undefined : 'none',
      }}
      className={cn(
        'block max-w-full rounded-xl border border-line-strong',
        tool === 'add' && 'cursor-crosshair',
        tool === 'move' && 'cursor-grab',
        tool === 'erase' && 'cursor-pointer',
        className,
      )}
      onPointerDown={(event) => {
        if (tool === 'none') return;
        const p = locate(event);
        const near = nearestPoint(pixels.px, pixels.py, p.cx, p.cy, 14);
        if (tool === 'add') onAdd?.(p.x, p.y);
        else if (tool === 'erase' && near !== null) onErase?.(near);
        else if (tool === 'move' && near !== null) {
          dragging.current = near;
          event.currentTarget.setPointerCapture(event.pointerId);
        }
      }}
      onPointerMove={(event) => {
        if (dragging.current === null) return;
        const p = locate(event);
        onMove?.(dragging.current, p.x, p.y);
      }}
      onPointerUp={() => (dragging.current = null)}
      onPointerCancel={() => (dragging.current = null)}
    />
  );
}
