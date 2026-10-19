import { type EpochStats } from '@sim/deep';
import { useEffect, useMemo, useRef, useState, type PointerEvent } from 'react';
import { Segmented } from '@/design-system';
import { usePalette } from '@/shared/canvas/colors';

type View = 'loss' | 'metric';

const HEIGHT = 190;
const PAD = { left: 42, right: 12, top: 14, bottom: 26 };
const MAX_POINTS = 360;

/** Evenly thins a series so SVG paths stay light on very long runs (always keeps the last point). */
export function downsample(length: number, max = MAX_POINTS): number[] {
  if (length <= max) return Array.from({ length }, (_, i) => i);
  const out: number[] = [];
  const step = (length - 1) / (max - 1);
  for (let i = 0; i < max; i++) out.push(Math.round(i * step));
  return out;
}

/** Powers of ten between `min` and `max` (inclusive), thinned to at most 5 ticks. */
export function decadeTicks(min: number, max: number): number[] {
  const lo = Math.round(Math.log10(min));
  const hi = Math.round(Math.log10(max));
  const every = Math.max(1, Math.ceil((hi - lo) / 4));
  const ticks: number[] = [];
  for (let e = lo; e <= hi; e += every) ticks.push(10 ** e);
  if (ticks.length < 2) ticks.push(max);
  return ticks;
}

export function niceMax(value: number): number {
  if (!(value > 0)) return 1;
  const exp = Math.floor(Math.log10(value));
  const base = 10 ** exp;
  const f = value / base;
  const nice = f <= 1 ? 1 : f <= 2 ? 2 : f <= 5 ? 5 : 10;
  return nice * base;
}

const format = (v: number) => {
  const a = Math.abs(v);
  if (a >= 100) return v.toFixed(0);
  if (a >= 1) return v.toFixed(2);
  if (a !== 0 && a < 0.001) return v.toExponential(1);
  return v.toFixed(3);
};

export function LossChart({
  history,
  problem,
  metricName,
}: {
  history: EpochStats[];
  problem: 'classification' | 'regression';
  /** Overrides the second view's name ("accuracy" / "MSE"), e.g. "reconstruction". */
  metricName?: string;
}) {
  const palette = usePalette();
  const wrapRef = useRef<HTMLDivElement>(null);
  const [width, setWidth] = useState(520);
  const [view, setView] = useState<View>('loss');
  const [hoverIndex, setHoverIndex] = useState<number | null>(null);

  useEffect(() => {
    const el = wrapRef.current;
    if (!el) return;
    const observer = new ResizeObserver(([entry]) => {
      if (entry) setWidth(Math.floor(entry.contentRect.width));
    });
    observer.observe(el);
    return () => observer.disconnect();
  }, []);

  const isAccuracy = view === 'metric' && problem === 'classification';
  const pick = (s: EpochStats, which: 'train' | 'test') =>
    view === 'loss'
      ? which === 'train'
        ? s.trainLoss
        : s.testLoss
      : which === 'train'
        ? s.trainMetric
        : s.testMetric;

  const chart = useMemo(() => {
    const indices = downsample(history.length);
    const points = indices.map((i) => history[i] as EpochStats);
    const innerW = Math.max(10, width - PAD.left - PAD.right);
    const innerH = HEIGHT - PAD.top - PAD.bottom;
    const first = points[0]?.epoch ?? 0;
    const last = points[points.length - 1]?.epoch ?? 1;
    const span = Math.max(1, last - first);
    const all = points.flatMap((p) => [pick(p, 'train'), pick(p, 'test')]).filter(Number.isFinite);
    // Losses span orders of magnitude, so they get a log axis (decades); accuracy and MSE are linear.
    const log = view === 'loss';
    const positives = all.filter((v) => v > 0);
    const yMax = isAccuracy
      ? 1
      : log
        ? 10 ** Math.ceil(Math.log10(Math.max(...positives, 1e-9)))
        : niceMax(Math.max(...all, 1e-6));
    const yMin = log
      ? 10 ** Math.floor(Math.log10(Math.max(Math.min(...positives, yMax), 1e-9)))
      : 0;
    const scale = (v: number) =>
      log
        ? (Math.log10(Math.max(v, yMin)) - Math.log10(yMin)) /
          Math.max(1e-9, Math.log10(yMax) - Math.log10(yMin))
        : (v - yMin) / (yMax - yMin);
    const x = (epoch: number) => PAD.left + ((epoch - first) / span) * innerW;
    const y = (v: number) => PAD.top + innerH - Math.min(1, Math.max(0, scale(v))) * innerH;
    const path = (which: 'train' | 'test') =>
      points
        .map(
          (p, i) =>
            `${i === 0 ? 'M' : 'L'}${x(p.epoch).toFixed(1)},${y(pick(p, which)).toFixed(1)}`,
        )
        .join(' ');
    return {
      points,
      x,
      y,
      yMax,
      yMin,
      log,
      first,
      last,
      innerW,
      innerH,
      train: path('train'),
      test: path('test'),
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [history, width, view, problem]);

  const hovered =
    hoverIndex === null ? null : chart.points[Math.min(hoverIndex, chart.points.length - 1)];
  const shown = hovered ?? history[history.length - 1];
  const empty = chart.points.length === 0;
  const yTicks = empty
    ? []
    : chart.log
      ? decadeTicks(chart.yMin, chart.yMax)
      : [0, 0.5, 1].map((t) => t * chart.yMax);

  const onMove = (event: PointerEvent<SVGSVGElement>) => {
    if (chart.points.length === 0) return;
    const rect = event.currentTarget.getBoundingClientRect();
    const rel = (event.clientX - rect.left - PAD.left) / chart.innerW;
    setHoverIndex(Math.round(Math.min(1, Math.max(0, rel)) * (chart.points.length - 1)));
  };

  const metricLabel = metricName ?? (problem === 'classification' ? 'accuracy' : 'MSE');
  const fmtMetric = (v: number) => (isAccuracy ? `${(v * 100).toFixed(1)}%` : format(v));

  return (
    <div ref={wrapRef}>
      <div className="mb-3 flex flex-wrap items-center justify-between gap-3">
        <Segmented<View>
          label="Chart metric"
          size="sm"
          fit
          value={view}
          onValueChange={setView}
          options={[
            { value: 'loss', label: 'Loss' },
            {
              value: 'metric',
              label: metricName
                ? metricName.charAt(0).toUpperCase() + metricName.slice(1)
                : problem === 'classification'
                  ? 'Accuracy'
                  : 'MSE',
            },
          ]}
        />
        <p className="font-mono text-[0.72rem] text-muted tabular-nums">
          {shown ? (
            <>
              epoch {shown.epoch} ·{' '}
              <span style={{ color: palette.css.brand }}>
                train {view === 'loss' ? format(shown.trainLoss) : fmtMetric(shown.trainMetric)}
              </span>{' '}
              ·{' '}
              <span style={{ color: palette.css.classes[2] }}>
                test {view === 'loss' ? format(shown.testLoss) : fmtMetric(shown.testMetric)}
              </span>
            </>
          ) : (
            `Press play to start training (${metricLabel} appears here)`
          )}
        </p>
      </div>
      <svg
        width="100%"
        height={HEIGHT}
        viewBox={`0 0 ${width} ${HEIGHT}`}
        onPointerMove={onMove}
        onPointerLeave={() => setHoverIndex(null)}
        role="img"
        aria-label={`Training ${view === 'loss' ? 'loss' : metricLabel} curves for train and test sets`}
      >
        {yTicks.map((tick, i) => (
          <g key={i}>
            <line
              x1={PAD.left}
              x2={width - PAD.right}
              y1={chart.y(tick)}
              y2={chart.y(tick)}
              stroke="var(--line)"
              strokeDasharray={i === 0 ? undefined : '3 5'}
            />
            <text
              x={PAD.left - 8}
              y={chart.y(tick) + 3.5}
              textAnchor="end"
              className="fill-muted font-mono text-[10px]"
            >
              {isAccuracy
                ? `${Math.round(tick * 100)}%`
                : chart.log
                  ? tick.toExponential(0)
                  : format(tick)}
            </text>
          </g>
        ))}
        {chart.points.length > 1 && (
          <>
            <text x={PAD.left} y={HEIGHT - 6} className="fill-faint font-mono text-[10px]">
              {chart.first}
            </text>
            <text
              x={width - PAD.right}
              y={HEIGHT - 6}
              textAnchor="end"
              className="fill-faint font-mono text-[10px]"
            >
              epoch {chart.last}
            </text>
            <path
              d={chart.test}
              fill="none"
              stroke={palette.css.classes[2]}
              strokeWidth={1.8}
              strokeDasharray="5 4"
              strokeLinejoin="round"
            />
            <path
              d={chart.train}
              fill="none"
              stroke={palette.css.brand}
              strokeWidth={2.2}
              strokeLinejoin="round"
            />
          </>
        )}
        {hovered && (
          <g>
            <line
              x1={chart.x(hovered.epoch)}
              x2={chart.x(hovered.epoch)}
              y1={PAD.top}
              y2={HEIGHT - PAD.bottom}
              stroke="var(--line-strong)"
            />
            <circle
              cx={chart.x(hovered.epoch)}
              cy={chart.y(pick(hovered, 'train'))}
              r={3.6}
              fill={palette.css.brand}
            />
            <circle
              cx={chart.x(hovered.epoch)}
              cy={chart.y(pick(hovered, 'test'))}
              r={3.6}
              fill={palette.css.classes[2]}
            />
          </g>
        )}
      </svg>
      <div className="mt-1 flex gap-5 font-mono text-[0.68rem] text-muted">
        <span className="flex items-center gap-2">
          <span className="h-0.5 w-5 rounded" style={{ background: palette.css.brand }} /> train
        </span>
        <span className="flex items-center gap-2">
          <span
            className="w-5 border-t-2 border-dashed"
            style={{ borderColor: palette.css.classes[2] }}
          />{' '}
          test
        </span>
      </div>
    </div>
  );
}
