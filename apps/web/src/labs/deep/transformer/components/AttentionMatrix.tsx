import { type Matrix } from '@sim/core';
import { useEffect, useRef, useState, type MouseEvent } from 'react';
import { cn } from '@/design-system';
import { mix, usePalette } from '@/shared/canvas/colors';
import { visible } from '../tokens';

export interface CellInfo {
  query: number;
  key: number;
}

interface Props {
  /** Attention weights (T × T), zero above the diagonal. */
  probs: Matrix;
  /** Display string per token. */
  labels: string[];
  /** Query position currently in focus (its row is outlined). */
  selected: number;
  onSelect: (position: number) => void;
  onHover?: (cell: CellInfo | null) => void;
  /** Largest cell size in px. */
  maxCell?: number;
  /** Small-multiple mode: no labels, no interaction. */
  compact?: boolean;
  className?: string;
}

const LABEL_W = 78;
const LABEL_H = 74;

/**
 * Rows are the token that is "looking" (query), columns are the tokens it can look at (keys).
 * The empty upper triangle is the causal mask: a token can never see the future.
 */
export function AttentionMatrix({
  probs,
  labels,
  selected,
  onSelect,
  onHover,
  maxCell = 30,
  compact = false,
  className,
}: Props) {
  const palette = usePalette();
  const wrapRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [available, setAvailable] = useState(420);
  const [hover, setHover] = useState<CellInfo | null>(null);

  const T = probs.rows;
  const reserveX = compact ? 0 : LABEL_W;
  const cell = Math.max(6, Math.min(maxCell, Math.floor((available - reserveX) / Math.max(T, 1))));
  const grid = cell * T;

  useEffect(() => {
    const el = wrapRef.current;
    if (!el) return;
    const observer = new ResizeObserver(([entry]) => {
      if (entry) setAvailable(Math.floor(entry.contentRect.width));
    });
    observer.observe(el);
    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    const canvas = canvasRef.current;
    const ctx = canvas?.getContext('2d');
    if (!canvas || !ctx) return;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    canvas.width = grid * dpr;
    canvas.height = grid * dpr;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, grid, grid);
    const gap = cell >= 14 ? 1 : 0;
    for (let i = 0; i < T; i++) {
      for (let j = 0; j < T; j++) {
        const x = j * cell;
        const y = i * cell;
        if (j > i) {
          // masked: a faint hatch so the triangle reads as "not allowed", not "zero attention"
          ctx.fillStyle = `rgba(${palette.fg[0]}, ${palette.fg[1]}, ${palette.fg[2]}, 0.035)`;
          ctx.fillRect(x + gap, y + gap, cell - gap * 2, cell - gap * 2);
          continue;
        }
        const w = probs.get(i, j);
        const c = mix(palette.surface, palette.brand, Math.sqrt(Math.min(1, w)));
        ctx.fillStyle = `rgb(${c[0]}, ${c[1]}, ${c[2]})`;
        ctx.fillRect(x + gap, y + gap, cell - gap * 2, cell - gap * 2);
      }
    }
  }, [probs, palette, cell, grid, T]);

  const locate = (event: MouseEvent<HTMLDivElement>): CellInfo | null => {
    const rect = event.currentTarget.getBoundingClientRect();
    const key = Math.floor((event.clientX - rect.left) / cell);
    const query = Math.floor((event.clientY - rect.top) / cell);
    if (key < 0 || query < 0 || key >= T || query >= T || key > query) return null;
    return { query, key };
  };

  const labelStyle = { fontSize: cell >= 20 ? 11 : 9 } as const;

  return (
    <div ref={wrapRef} className={cn('w-full', className)}>
      <div className="flex">
        {!compact && <div style={{ width: LABEL_W }} className="shrink-0" />}
        {!compact && (
          <div className="relative flex" style={{ width: grid, height: LABEL_H }} aria-hidden>
            {labels.map((label, j) => (
              <span
                key={j}
                className="flex items-end justify-start overflow-hidden pb-1 font-mono whitespace-pre text-muted"
                style={{
                  width: cell,
                  height: LABEL_H,
                  writingMode: 'vertical-rl',
                  transform: 'rotate(180deg)',
                  ...labelStyle,
                  ...(hover?.key === j ? { color: 'var(--fg)' } : {}),
                }}
              >
                {visible(label)}
              </span>
            ))}
          </div>
        )}
      </div>
      <div className="flex">
        {!compact && (
          <div className="shrink-0" style={{ width: LABEL_W }} aria-hidden>
            {labels.map((label, i) => (
              <button
                key={i}
                type="button"
                tabIndex={-1}
                onClick={() => onSelect(i)}
                className={cn(
                  'block w-full cursor-pointer overflow-hidden pr-2 text-right font-mono whitespace-pre text-ellipsis text-muted transition-colors hover:text-fg',
                  (selected === i || hover?.query === i) && 'text-fg',
                )}
                style={{ height: cell, lineHeight: `${cell}px`, ...labelStyle }}
              >
                {visible(label)}
              </button>
            ))}
          </div>
        )}
        <div
          className={cn('relative', !compact && 'cursor-crosshair')}
          style={{ width: grid, height: grid }}
          onPointerMove={
            compact
              ? undefined
              : (e) => {
                  const next = locate(e);
                  setHover(next);
                  onHover?.(next);
                }
          }
          onPointerLeave={
            compact
              ? undefined
              : () => {
                  setHover(null);
                  onHover?.(null);
                }
          }
          onClick={
            compact
              ? undefined
              : (e) => {
                  const c = locate(e);
                  if (c) onSelect(c.query);
                }
          }
          role={compact ? 'img' : 'grid'}
          aria-label="Attention weights: rows are the tokens looking, columns are the tokens they look at"
        >
          <canvas
            ref={canvasRef}
            style={{ width: grid, height: grid }}
            className="block rounded-sm"
          />
          {!compact && (
            <>
              {/* row focus */}
              <span
                aria-hidden
                className="pointer-events-none absolute inset-x-0 rounded-sm border border-brand shadow-[0_0_0_2px_var(--brand-soft)]"
                style={{ top: selected * cell, height: cell }}
              />
              {hover && (
                <span
                  aria-hidden
                  className="pointer-events-none absolute rounded-sm border-2 border-fg"
                  style={{
                    left: hover.key * cell,
                    top: hover.query * cell,
                    width: cell,
                    height: cell,
                  }}
                />
              )}
            </>
          )}
        </div>
      </div>
    </div>
  );
}
