import { useEffect, useMemo, useRef, useState, type PointerEvent } from 'react';
import { linearRange, logRange, position, thin, type Range } from './chartScale';

export interface Series {
  name: string;
  color: string;
  values: ArrayLike<number>;
  dashed?: boolean;
}

interface Props {
  title: string;
  /** One x value per sample (shared by all series). */
  x: ArrayLike<number>;
  series: Series[];
  /** Log y axis (positive values only). */
  log?: boolean;
  /** Fix the y range (linear axes only). */
  yRange?: Partial<Range>;
  /** A horizontal guide, e.g. 0.5 for a discriminator in equilibrium. */
  guide?: number;
  /** Text under the chart. */
  note?: string;
  xLabel?: string;
  height?: number;
}

const PAD = { left: 40, right: 10, top: 10, bottom: 22 };

const fmt = (v: number) => {
  const a = Math.abs(v);
  if (a >= 100) return v.toFixed(0);
  if (a >= 1) return v.toFixed(2);
  if (a !== 0 && a < 0.001) return v.toExponential(0);
  return v.toFixed(3);
};

/** A compact line chart with a hover read-out, used for training statistics. */
export function MiniChart({
  title,
  x,
  series,
  log = false,
  yRange,
  guide,
  note,
  xLabel = 'iteration',
  height = 150,
}: Props) {
  const wrapRef = useRef<HTMLDivElement>(null);
  const [width, setWidth] = useState(320);
  const [hover, setHover] = useState<number | null>(null);

  useEffect(() => {
    const el = wrapRef.current;
    if (!el) return;
    const observer = new ResizeObserver(([entry]) => {
      if (entry) setWidth(Math.max(120, Math.floor(entry.contentRect.width)));
    });
    observer.observe(el);
    return () => observer.disconnect();
  }, []);

  const chart = useMemo(() => {
    const idx = thin(x.length);
    const all = series.flatMap((s) => idx.map((i) => s.values[i] as number));
    if (guide !== undefined) all.push(guide);
    const range: Range = log ? logRange(all) : linearRange(all, yRange);
    const innerW = Math.max(10, width - PAD.left - PAD.right);
    const innerH = height - PAD.top - PAD.bottom;
    const first = (x[idx[0] ?? 0] as number | undefined) ?? 0;
    const last = (x[idx[idx.length - 1] ?? 0] as number | undefined) ?? 1;
    const span = Math.max(1e-9, last - first);
    const px = (v: number) => PAD.left + ((v - first) / span) * innerW;
    const py = (v: number) => PAD.top + innerH - position(v, range, log) * innerH;
    const paths = series.map((s) =>
      idx
        .map(
          (i, k) =>
            `${k === 0 ? 'M' : 'L'}${px(x[i] as number).toFixed(1)},${py(s.values[i] as number).toFixed(1)}`,
        )
        .join(' '),
    );
    const ticks = log
      ? Array.from({ length: range.hi - range.lo + 1 }, (_, k) => 10 ** (range.lo + k)).filter(
          (_, k, a) => a.length <= 5 || k % Math.ceil(a.length / 4) === 0,
        )
      : [range.lo, (range.lo + range.hi) / 2, range.hi];
    return { idx, range, innerW, innerH, first, last, px, py, paths, ticks };
  }, [x, series, log, yRange, guide, width, height]);

  const onMove = (event: PointerEvent<SVGSVGElement>) => {
    if (chart.idx.length === 0) return;
    const rect = event.currentTarget.getBoundingClientRect();
    const rel = (event.clientX - rect.left - PAD.left) / chart.innerW;
    setHover(Math.round(Math.min(1, Math.max(0, rel)) * (chart.idx.length - 1)));
  };

  const at = chart.idx.length
    ? chart.idx[Math.min(hover ?? chart.idx.length - 1, chart.idx.length - 1)]
    : undefined;
  const empty = x.length === 0;

  return (
    <figure ref={wrapRef} className="min-w-0">
      <figcaption className="mb-1 flex flex-wrap items-baseline justify-between gap-x-3">
        <span className="eyebrow">{title}</span>
        <span className="font-mono text-[0.68rem] text-muted tabular-nums">
          {at !== undefined && !empty
            ? `${xLabel} ${x[at]} · ${series.map((s) => `${s.name} ${fmt(s.values[at] as number)}`).join(' · ')}`
            : '–'}
        </span>
      </figcaption>
      <svg
        width="100%"
        height={height}
        viewBox={`0 0 ${width} ${height}`}
        role="img"
        aria-label={`${title}: ${series.map((s) => s.name).join(', ')}`}
        onPointerMove={onMove}
        onPointerLeave={() => setHover(null)}
      >
        {chart.ticks.map((t, i) => {
          const y = chart.py(t);
          return (
            <g key={i}>
              <line
                x1={PAD.left}
                x2={width - PAD.right}
                y1={y}
                y2={y}
                stroke="var(--line)"
                strokeDasharray={i === 0 ? undefined : '3 5'}
              />
              <text
                x={PAD.left - 6}
                y={y + 3.5}
                textAnchor="end"
                className="fill-muted font-mono text-[9px]"
              >
                {log ? `1e${Math.round(Math.log10(t))}` : fmt(t)}
              </text>
            </g>
          );
        })}
        {guide !== undefined && (
          <line
            x1={PAD.left}
            x2={width - PAD.right}
            y1={chart.py(guide)}
            y2={chart.py(guide)}
            stroke="var(--line-strong)"
            strokeDasharray="2 4"
          />
        )}
        {!empty && x.length > 1 && (
          <>
            <text x={PAD.left} y={height - 6} className="fill-faint font-mono text-[9px]">
              {chart.first}
            </text>
            <text
              x={width - PAD.right}
              y={height - 6}
              textAnchor="end"
              className="fill-faint font-mono text-[9px]"
            >
              {chart.last}
            </text>
            {series.map((s, k) => (
              <path
                key={s.name}
                d={chart.paths[k]}
                fill="none"
                stroke={s.color}
                strokeWidth={2}
                strokeDasharray={s.dashed ? '4 4' : undefined}
                strokeLinejoin="round"
              />
            ))}
          </>
        )}
        {hover !== null && at !== undefined && !empty && (
          <line
            x1={chart.px(x[at] as number)}
            x2={chart.px(x[at] as number)}
            y1={PAD.top}
            y2={height - PAD.bottom}
            stroke="var(--line-strong)"
          />
        )}
      </svg>
      <div className="mt-1 flex flex-wrap gap-x-4 font-mono text-[0.66rem] text-muted">
        {series.map((s) => (
          <span key={s.name} className="flex items-center gap-1.5">
            <span className="h-0.5 w-4 rounded" style={{ background: s.color }} />
            {s.name}
          </span>
        ))}
      </div>
      {note && <p className="mt-1.5 text-[0.72rem] leading-snug text-faint">{note}</p>}
    </figure>
  );
}
