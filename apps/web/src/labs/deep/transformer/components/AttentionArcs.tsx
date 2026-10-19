import { type Matrix } from '@sim/core';
import { useLayoutEffect, useRef, useState } from 'react';
import { cn } from '@/design-system';
import { tokenHue, tokenLabel } from '../tokens';

interface Props {
  /** Attention weights (T × T) of the head being shown. */
  probs: Matrix;
  /** Display string per token. */
  labels: string[];
  ids: readonly number[];
  /** Query token: the arcs start here. */
  query: number;
  onSelect: (position: number) => void;
}

const ARC_H = 96;
const MIN_WEIGHT = 0.03;

/**
 * The same numbers as one row of the attention matrix, drawn over the text itself: an arc from the
 * token that is reading to every token it reads from, as thick as the attention weight.
 */
export function AttentionArcs({ probs, labels, ids, query, onSelect }: Props) {
  const rowRef = useRef<HTMLUListElement>(null);
  const [layout, setLayout] = useState<{ centers: number[]; width: number }>({
    centers: [],
    width: 0,
  });

  useLayoutEffect(() => {
    const el = rowRef.current;
    if (!el) return;
    const measure = () =>
      setLayout({
        centers: Array.from(
          el.children,
          (li) => (li as HTMLElement).offsetLeft + (li as HTMLElement).offsetWidth / 2,
        ),
        width: el.scrollWidth,
      });
    measure();
    const observer = new ResizeObserver(measure);
    observer.observe(el);
    return () => observer.disconnect();
  }, [labels]);

  const { centers, width } = layout;
  const xq = centers[query];

  return (
    <div className="overflow-x-auto pb-1" aria-label="Attention arcs">
      <div className="relative" style={{ width: Math.max(width, 1) }}>
        <svg width={Math.max(width, 1)} height={ARC_H} className="block" aria-hidden>
          {xq !== undefined &&
            centers.map((xk, j) => {
              const w = j <= query ? probs.get(query, j) : 0;
              if (w < MIN_WEIGHT || j === query) return null;
              const lift = Math.min(ARC_H - 8, 14 + Math.abs(xq - xk) * 0.32);
              return (
                <path
                  key={j}
                  d={`M${xq},${ARC_H} Q${(xq + xk) / 2},${ARC_H - lift * 2} ${xk},${ARC_H}`}
                  fill="none"
                  stroke="var(--brand)"
                  strokeWidth={1.2 + 7 * w}
                  strokeOpacity={0.25 + 0.7 * w}
                  strokeLinecap="round"
                />
              );
            })}
          {xq !== undefined && probs.get(query, query) >= MIN_WEIGHT && (
            <circle
              cx={xq}
              cy={ARC_H - 4}
              r={3 + 8 * probs.get(query, query)}
              fill="var(--brand)"
              fillOpacity={0.5}
            />
          )}
        </svg>
        <ul ref={rowRef} className="flex gap-1.5" aria-label="Tokens, attention view">
          {labels.map((label, i) => {
            const w = i <= query ? probs.get(query, i) : 0;
            return (
              <li key={i} className="flex flex-col items-center gap-1">
                <button
                  type="button"
                  onClick={() => onSelect(i)}
                  aria-pressed={i === query}
                  style={{ ['--h' as string]: tokenHue(i) }}
                  className={cn(
                    'tok cursor-pointer rounded-md px-1.5 py-0.5 font-mono text-[0.84rem] whitespace-pre transition-all duration-200',
                    i === query && 'tok-solid text-bg shadow-[0_0_0_3px_var(--brand-soft)]',
                    i > query && 'opacity-40',
                  )}
                >
                  {tokenLabel(label, ids[i] as number)}
                </button>
                <span className="font-mono text-[0.62rem] text-muted tabular-nums">
                  {i <= query && w >= 0.005 ? `${Math.round(w * 100)}%` : '·'}
                </span>
              </li>
            );
          })}
        </ul>
      </div>
    </div>
  );
}
