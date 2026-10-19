import { CATEGORY_LABEL, categoryOf, type WordCategory } from '@sim/deep';
import { useMemo } from 'react';
import { cn } from '@/design-system';
import { CATEGORY_COLOR } from '../colors';
import type { MapData } from '../store';

const W = 640;
const H = 460;
const PAD = 34;

interface Arrow {
  from: string;
  to: string;
  dashed?: boolean;
}

/** Every content word on the plane of the two main axes of its vector, coloured by family. */
export function EmbeddingMap({
  map,
  selected,
  onSelect,
  arrows = [],
  marked = [],
}: {
  map: MapData;
  selected?: string;
  onSelect?: (word: string) => void;
  arrows?: Arrow[];
  /** Words to circle (e.g. the three words of an analogy and its answer). */
  marked?: string[];
}) {
  const view = useMemo(() => {
    const n = map.words.length;
    let minX = Infinity;
    let maxX = -Infinity;
    let minY = Infinity;
    let maxY = -Infinity;
    for (let i = 0; i < n; i++) {
      const x = map.coords[2 * i] as number;
      const y = map.coords[2 * i + 1] as number;
      minX = Math.min(minX, x);
      maxX = Math.max(maxX, x);
      minY = Math.min(minY, y);
      maxY = Math.max(maxY, y);
    }
    const spanX = Math.max(1e-9, maxX - minX);
    const spanY = Math.max(1e-9, maxY - minY);
    const at = new Map<string, [number, number]>();
    map.words.forEach((word, i) => {
      at.set(word, [
        PAD + (((map.coords[2 * i] as number) - minX) / spanX) * (W - 2 * PAD),
        H - PAD - (((map.coords[2 * i + 1] as number) - minY) / spanY) * (H - 2 * PAD),
      ]);
    });
    return at;
  }, [map]);

  const categories = useMemo(
    () => [...new Set(map.words.map((w) => categoryOf(w)))] as WordCategory[],
    [map.words],
  );

  return (
    <div>
      <svg
        viewBox={`0 0 ${W} ${H}`}
        className="block w-full max-w-[640px] rounded-xl border border-line-strong bg-surface"
        role="group"
        aria-label="Map of the word vectors, projected on their two main axes"
      >
        <defs>
          <marker
            id="arrowhead"
            viewBox="0 0 10 10"
            refX="8"
            refY="5"
            markerWidth="7"
            markerHeight="7"
            orient="auto"
          >
            <path d="M0 0 L10 5 L0 10 z" fill="var(--fg)" />
          </marker>
        </defs>
        {arrows.map((a, i) => {
          const from = view.get(a.from);
          const to = view.get(a.to);
          if (!from || !to) return null;
          return (
            <line
              key={i}
              x1={from[0]}
              y1={from[1]}
              x2={to[0]}
              y2={to[1]}
              stroke="var(--fg)"
              strokeWidth={1.6}
              strokeDasharray={a.dashed ? '5 4' : undefined}
              markerEnd="url(#arrowhead)"
              opacity={0.75}
            />
          );
        })}
        {map.words.map((word) => {
          const p = view.get(word);
          if (!p) return null;
          const category = categoryOf(word);
          const active = word === selected;
          const isMarked = marked.includes(word);
          return (
            <g
              key={word}
              role={onSelect ? 'button' : undefined}
              tabIndex={onSelect ? 0 : undefined}
              aria-label={onSelect ? `Select ${word}` : undefined}
              aria-pressed={onSelect ? active : undefined}
              onClick={() => onSelect?.(word)}
              onKeyDown={(e) => {
                if (onSelect && (e.key === 'Enter' || e.key === ' ')) {
                  e.preventDefault();
                  onSelect(word);
                }
              }}
              className={cn(onSelect && 'cursor-pointer outline-none')}
            >
              {(active || isMarked) && (
                <circle
                  cx={p[0]}
                  cy={p[1]}
                  r={11}
                  fill="none"
                  stroke={CATEGORY_COLOR[category]}
                  strokeWidth={2}
                />
              )}
              <circle
                cx={p[0]}
                cy={p[1]}
                r={5.2}
                fill={CATEGORY_COLOR[category]}
                stroke="var(--surface)"
                strokeWidth={1}
              />
              <text
                x={p[0] + 8}
                y={p[1] - 7}
                className={cn(
                  'font-mono text-[11px]',
                  active || isMarked ? 'fill-fg font-semibold' : 'fill-muted',
                )}
                paintOrder="stroke"
                stroke="var(--surface)"
                strokeWidth={3}
                strokeLinejoin="round"
              >
                {word}
              </text>
            </g>
          );
        })}
      </svg>
      <ul
        className="mt-3 flex flex-wrap gap-x-4 gap-y-1.5 text-[0.76rem] text-muted"
        aria-label="Word families"
      >
        {categories.map((c) => (
          <li key={c} className="flex items-center gap-1.5">
            <span className="size-2.5 rounded-full" style={{ background: CATEGORY_COLOR[c] }} />
            {CATEGORY_LABEL[c]}
          </li>
        ))}
      </ul>
    </div>
  );
}
