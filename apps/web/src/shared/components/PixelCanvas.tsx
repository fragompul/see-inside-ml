import { useEffect, useRef, type PointerEvent } from 'react';
import { cn } from '@/design-system';
import { mix, usePalette } from '@/shared/canvas/colors';
import { divergingRgb } from '@/shared/components/Heat';

export interface PixelBox {
  x: number;
  y: number;
  w: number;
  h: number;
}

interface Props {
  /** `side × side` values. */
  values: ArrayLike<number>;
  side: number;
  /** CSS size in pixels (square). */
  size: number;
  /** `ink`: 0..1 from paper to ink; `diverging`: frost ← 0 → ember around ±scale. */
  mode?: 'ink' | 'diverging';
  scale?: number;
  /** Translucent ember wash on top, e.g. an occlusion map (normalised by its own maximum). */
  overlay?: ArrayLike<number> | null;
  /** Outlined rectangle in pixel units (the convolution window). */
  box?: PixelBox | null;
  grid?: boolean;
  /** Pointer callbacks receive positions in pixel units (0..side, fractional). */
  onDown?: (x: number, y: number) => void;
  onMove?: (x: number, y: number) => void;
  onUp?: () => void;
  onLeave?: () => void;
  ariaLabel: string;
  className?: string;
}

/** A small image drawn pixel by pixel, optionally interactive (drawing pad, window dragging). */
export function PixelCanvas({
  values,
  side,
  size,
  mode = 'ink',
  scale = 1,
  overlay,
  box,
  grid = false,
  onDown,
  onMove,
  onUp,
  onLeave,
  ariaLabel,
  className,
}: Props) {
  const palette = usePalette();
  const ref = useRef<HTMLCanvasElement>(null);
  const interactive = Boolean(onDown || onMove);

  useEffect(() => {
    const canvas = ref.current;
    const ctx = canvas?.getContext('2d');
    if (!canvas || !ctx) return;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    canvas.width = Math.round(size * dpr);
    canvas.height = Math.round(size * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    const cell = size / side;

    for (let i = 0; i < side * side; i++) {
      const v = values[i] as number;
      const c =
        mode === 'ink'
          ? mix(palette.surface, palette.fg, Math.max(0, Math.min(1, v)))
          : divergingRgb(v, scale, palette);
      ctx.fillStyle = `rgb(${c[0]}, ${c[1]}, ${c[2]})`;
      const x = (i % side) * cell;
      const y = Math.floor(i / side) * cell;
      // overdraw by a hair so antialiasing never leaves seams between cells
      ctx.fillRect(x, y, cell + 0.5, cell + 0.5);
    }

    if (overlay) {
      let max = 1e-9;
      for (let i = 0; i < overlay.length; i++) max = Math.max(max, overlay[i] as number);
      const [r, g, b] = palette.pos;
      for (let i = 0; i < side * side; i++) {
        const a = Math.max(0, (overlay[i] as number) / max);
        if (a < 0.04) continue;
        ctx.fillStyle = `rgba(${r}, ${g}, ${b}, ${(a * 0.78).toFixed(3)})`;
        ctx.fillRect((i % side) * cell, Math.floor(i / side) * cell, cell + 0.5, cell + 0.5);
      }
    }

    if (grid) {
      ctx.strokeStyle = palette.line;
      ctx.globalAlpha = 0.22;
      ctx.lineWidth = 1;
      ctx.beginPath();
      for (let i = 1; i < side; i++) {
        ctx.moveTo(i * cell, 0);
        ctx.lineTo(i * cell, size);
        ctx.moveTo(0, i * cell);
        ctx.lineTo(size, i * cell);
      }
      ctx.stroke();
      ctx.globalAlpha = 1;
    }

    if (box) {
      ctx.lineWidth = 2.5;
      ctx.strokeStyle = palette.css.brand;
      ctx.shadowColor = palette.css.brand;
      ctx.shadowBlur = 10;
      ctx.strokeRect(box.x * cell + 1, box.y * cell + 1, box.w * cell - 2, box.h * cell - 2);
      ctx.shadowBlur = 0;
    }
  }, [values, side, size, mode, scale, overlay, box, grid, palette]);

  const locate = (event: PointerEvent<HTMLCanvasElement>): [number, number] => {
    const rect = event.currentTarget.getBoundingClientRect();
    return [
      ((event.clientX - rect.left) / rect.width) * side,
      ((event.clientY - rect.top) / rect.height) * side,
    ];
  };

  return (
    <canvas
      ref={ref}
      role="img"
      aria-label={ariaLabel}
      style={{
        width: size,
        height: 'auto',
        aspectRatio: '1',
        touchAction: interactive ? 'none' : undefined,
      }}
      className={cn(
        'block max-w-full rounded-lg border border-line-strong',
        interactive && 'cursor-crosshair',
        className,
      )}
      onPointerDown={
        onDown
          ? (e) => {
              e.currentTarget.setPointerCapture(e.pointerId);
              onDown(...locate(e));
            }
          : undefined
      }
      onPointerMove={onMove ? (e) => onMove(...locate(e)) : undefined}
      onPointerUp={onUp}
      onPointerCancel={onUp}
      onPointerLeave={onLeave}
    />
  );
}
