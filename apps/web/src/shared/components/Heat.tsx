import { useEffect, useRef, useState, type MouseEvent, type PointerEvent } from 'react';
import { cn } from '@/design-system';
import { mix, usePalette, type Palette, type Rgb } from '@/shared/canvas/colors';

interface Props {
  /** Row-major values, `rows × cols`. */
  values: ArrayLike<number>;
  cols: number;
  /** Symmetric colour range ±scale (0 = neutral). Share one scale across strips to compare them. */
  scale: number;
  cellW?: number;
  cellH?: number;
  /** Called with the flat index under the pointer (or null). */
  onHover?: (index: number | null) => void;
  /** Called with the flat index of the clicked cell. */
  onSelect?: (index: number) => void;
  className?: string;
  ariaLabel: string;
}

/** Diverging colour: frost (negative) ← surface (0) → ember (positive). */
export function divergingRgb(value: number, scale: number, palette: Palette): Rgb {
  const t = scale > 0 ? Math.max(-1, Math.min(1, value / scale)) : 0;
  return t >= 0 ? mix(palette.surface, palette.pos, t) : mix(palette.surface, palette.neg, -t);
}

/** A grid of numbers drawn as colours: one cell per value. */
export function Heat({
  values,
  cols,
  scale,
  cellW = 6,
  cellH = 14,
  onHover,
  onSelect,
  className,
  ariaLabel,
}: Props) {
  const palette = usePalette();
  const ref = useRef<HTMLCanvasElement>(null);
  const [hover, setHover] = useState<number | null>(null);
  const rows = Math.ceil(values.length / cols);
  const width = cols * cellW;
  const height = rows * cellH;

  useEffect(() => {
    const canvas = ref.current;
    const ctx = canvas?.getContext('2d');
    if (!canvas || !ctx) return;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    canvas.width = width * dpr;
    canvas.height = height * dpr;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    const gap = cellW >= 8 && cellH >= 8 ? 1 : 0;
    for (let i = 0; i < values.length; i++) {
      const c = divergingRgb(values[i] as number, scale, palette);
      ctx.fillStyle = `rgb(${c[0]}, ${c[1]}, ${c[2]})`;
      ctx.fillRect(
        (i % cols) * cellW + gap / 2,
        Math.floor(i / cols) * cellH + gap / 2,
        cellW - gap,
        cellH - gap,
      );
    }
  }, [values, cols, scale, palette, width, height, cellW, cellH]);

  const locate = (event: PointerEvent<HTMLDivElement> | MouseEvent<HTMLDivElement>) => {
    const rect = event.currentTarget.getBoundingClientRect();
    const col = Math.floor(((event.clientX - rect.left) / rect.width) * cols);
    const row = Math.floor(((event.clientY - rect.top) / rect.height) * rows);
    const index = row * cols + col;
    return col >= 0 && col < cols && row >= 0 && row < rows && index < values.length ? index : null;
  };

  return (
    <div
      className={cn(
        'relative inline-block',
        (onHover || onSelect) && 'cursor-crosshair',
        className,
      )}
      style={{ width, height }}
      role="img"
      aria-label={ariaLabel}
      onClick={
        onSelect
          ? (e) => {
              const i = locate(e);
              if (i !== null) onSelect(i);
            }
          : undefined
      }
      onPointerMove={
        onHover || onSelect
          ? (e) => {
              const i = locate(e);
              setHover(i);
              onHover?.(i);
            }
          : undefined
      }
      onPointerLeave={
        onHover || onSelect
          ? () => {
              setHover(null);
              onHover?.(null);
            }
          : undefined
      }
    >
      <canvas ref={ref} style={{ width, height }} className="block rounded-[3px]" />
      {hover !== null && (
        <span
          aria-hidden
          className="pointer-events-none absolute rounded-[2px] border-2 border-fg"
          style={{
            left: (hover % cols) * cellW,
            top: Math.floor(hover / cols) * cellH,
            width: cellW,
            height: cellH,
          }}
        />
      )}
    </div>
  );
}

/** Largest absolute value in one or more arrays (used to share a colour scale). */
export function maxAbs(...arrays: ArrayLike<number>[]): number {
  let max = 1e-9;
  for (const a of arrays)
    for (let i = 0; i < a.length; i++) max = Math.max(max, Math.abs(a[i] as number));
  return max;
}
