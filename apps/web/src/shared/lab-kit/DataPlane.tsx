import type { Table } from '@sim/core';
import { useEffect, useMemo, useRef, type PointerEvent } from 'react';
import { cn } from '@/design-system';
import { usePalette, type Palette } from '@/shared/canvas/colors';
import type { Regions } from './pipeline';
import { nearestPoint } from './tableEdit';

export type EditTool = 'add' | 'move' | 'erase' | 'none';

/** What an overlay needs to draw in data coordinates. */
export interface Painter {
  ctx: CanvasRenderingContext2D;
  px: (x: number) => number;
  py: (y: number) => number;
  palette: Palette;
  width: number;
  height: number;
}

export interface DataPlaneProps {
  all: Table;
  /** 1 for the rows held out for testing. */
  testMask: Uint8Array;
  regions?: Regions | null;
  bounds: { x0: number; x1: number; y0: number; y1: number };
  /** 1 for the rows the model gets wrong. */
  wrong?: Uint8Array | null;
  /** Rows to ring (support vectors, selected points…). */
  highlight?: ReadonlySet<number> | null;
  tool?: EditTool;
  activeClass?: number;
  onAdd?: (x: number, y: number, cls: number) => void;
  onMove?: (index: number, x: number, y: number) => void;
  onErase?: (index: number) => void;
  /** Drawn after the regions, before the points (e.g. margins, contours). */
  underlay?: (p: Painter) => void;
  /** Drawn on top of the points (e.g. tree splits, centroids). */
  overlay?: (p: Painter) => void;
  width?: number;
  height?: number;
  ariaLabel: string;
  className?: string;
}

/** A 2-D scatter of labelled points over the model's decision regions, editable with the pointer. */
export function DataPlane({
  all,
  testMask,
  regions,
  bounds,
  wrong,
  highlight,
  tool = 'none',
  activeClass = 0,
  onAdd,
  onMove,
  onErase,
  underlay,
  overlay,
  width = 560,
  height = 420,
  ariaLabel,
  className,
}: DataPlaneProps) {
  const palette = usePalette();
  const ref = useRef<HTMLCanvasElement>(null);
  const dragging = useRef<number | null>(null);

  const toPx = useMemo(() => {
    const sx = width / (bounds.x1 - bounds.x0);
    const sy = height / (bounds.y1 - bounds.y0);
    return {
      x: (v: number) => (v - bounds.x0) * sx,
      y: (v: number) => height - (v - bounds.y0) * sy,
      sx,
      sy,
    };
  }, [bounds, width, height]);

  const pixels = useMemo(() => {
    const px = new Float64Array(all.x.rows);
    const py = new Float64Array(all.x.rows);
    for (let i = 0; i < all.x.rows; i++) {
      px[i] = toPx.x(all.x.get(i, 0));
      py[i] = toPx.y(all.x.get(i, 1));
    }
    return { px, py };
  }, [all, toPx]);

  // the decision regions as one small image, scaled up
  const regionImage = useMemo(() => {
    if (!regions) return null;
    const canvas = document.createElement('canvas');
    canvas.width = regions.n;
    canvas.height = regions.n;
    const ctx = canvas.getContext('2d');
    if (!ctx) return null;
    const img = ctx.createImageData(regions.n, regions.n);
    for (let i = 0; i < regions.n * regions.n; i++) {
      const c = palette.classes[
        (regions.labels[i] as number) % palette.classes.length
      ] as readonly [number, number, number];
      const conf = regions.confidence[i] as number;
      // pull towards the background when the model is unsure
      const a = 0.16 + 0.44 * Math.max(0, Math.min(1, (conf - 0.3) / 0.7));
      img.data[i * 4] = c[0] * a + palette.surface[0] * (1 - a);
      img.data[i * 4 + 1] = c[1] * a + palette.surface[1] * (1 - a);
      img.data[i * 4 + 2] = c[2] * a + palette.surface[2] * (1 - a);
      img.data[i * 4 + 3] = 255;
    }
    ctx.putImageData(img, 0, 0);
    return canvas;
  }, [regions, palette]);

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
    if (regionImage) {
      ctx.imageSmoothingEnabled = false;
      ctx.drawImage(regionImage, 0, 0, width, height);
    }
    const painter: Painter = { ctx, px: toPx.x, py: toPx.y, palette, width, height };
    underlay?.(painter);

    for (let i = 0; i < all.x.rows; i++) {
      const label = all.y[i] as number;
      // a negative label (noise, unassigned) is drawn in a neutral colour
      const color =
        label < 0
          ? palette.css.muted
          : (palette.css.classes[label % palette.css.classes.length] as string);
      const x = pixels.px[i] as number;
      const y = pixels.py[i] as number;
      ctx.beginPath();
      ctx.arc(x, y, 4.6, 0, Math.PI * 2);
      if (testMask[i] === 1) {
        ctx.fillStyle = palette.css.surface;
        ctx.fill();
        ctx.lineWidth = 2;
        ctx.strokeStyle = color;
        ctx.stroke();
      } else {
        ctx.fillStyle = color;
        ctx.fill();
        ctx.lineWidth = 1;
        ctx.strokeStyle = palette.css.surface;
        ctx.stroke();
      }
      if (wrong?.[i] === 1) {
        ctx.beginPath();
        ctx.arc(x, y, 8, 0, Math.PI * 2);
        ctx.lineWidth = 1.6;
        ctx.strokeStyle = palette.css.fg;
        ctx.stroke();
      }
      if (highlight?.has(i)) {
        ctx.beginPath();
        ctx.arc(x, y, 10.5, 0, Math.PI * 2);
        ctx.lineWidth = 2;
        ctx.strokeStyle = palette.css.brand;
        ctx.stroke();
      }
    }
    overlay?.(painter);
  }, [
    all,
    testMask,
    regionImage,
    wrong,
    highlight,
    pixels,
    toPx,
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
    return { cx, cy, x: bounds.x0 + cx / toPx.sx, y: bounds.y0 + (height - cy) / toPx.sy };
  };

  const onDown = (event: PointerEvent<HTMLCanvasElement>) => {
    if (tool === 'none') return;
    const p = locate(event);
    const near = nearestPoint(pixels.px, pixels.py, p.cx, p.cy, 14);
    if (tool === 'add') onAdd?.(p.x, p.y, activeClass);
    else if (tool === 'erase' && near !== null) onErase?.(near);
    else if (tool === 'move' && near !== null) {
      dragging.current = near;
      event.currentTarget.setPointerCapture(event.pointerId);
    }
  };
  const onMoveEvent = (event: PointerEvent<HTMLCanvasElement>) => {
    if (dragging.current === null) return;
    const p = locate(event);
    onMove?.(dragging.current, p.x, p.y);
  };
  const release = () => {
    dragging.current = null;
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
      onPointerDown={onDown}
      onPointerMove={onMoveEvent}
      onPointerUp={release}
      onPointerCancel={release}
    />
  );
}
