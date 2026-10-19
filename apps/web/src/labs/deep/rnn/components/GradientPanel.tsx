import { createRng } from '@sim/core';
import { CELL_NAMES, cellInfo, createNet, gradientProfile, type CellName } from '@sim/deep';
import { useMemo, useRef, useState, useEffect } from 'react';
import { usePalette } from '@/shared/canvas/colors';
import { useRnnStore } from '../store';
import { logPosition, logRange } from '../view';

const HEIGHT = 230;
const PAD = { left: 52, right: 14, top: 14, bottom: 30 };

/** Learning signal reaching each step, untrained RNN vs GRU vs LSTM, and the network in the lab. */
export function GradientPanel() {
  const palette = usePalette();
  const task = useRnnStore((s) => s.config.task);
  const seqLen = useRnnStore((s) => s.config.seqLen);
  const hidden = useRnnStore((s) => s.config.hidden);
  const seed = useRnnStore((s) => s.config.seed);
  const cell = useRnnStore((s) => s.config.cell);
  const live = useRnnStore((s) => s.profile);
  const epoch = useRnnStore((s) => s.epoch);

  const wrapRef = useRef<HTMLDivElement>(null);
  const [width, setWidth] = useState(560);
  const [hover, setHover] = useState<number | null>(null);

  useEffect(() => {
    const el = wrapRef.current;
    if (!el) return;
    const observer = new ResizeObserver(([entry]) => {
      if (entry) setWidth(Math.floor(entry.contentRect.width));
    });
    observer.observe(el);
    return () => observer.disconnect();
  }, []);

  // Fresh, untrained networks of each kind (same size as the lab network): the starting point.
  const initial = useMemo(() => {
    const out = {} as Record<CellName, Float64Array>;
    for (const name of CELL_NAMES) {
      out[name] = gradientProfile(
        createNet(name, hidden, task, createRng(seed + 2)),
        task,
        seqLen,
        24,
        createRng(seed + 11),
      );
    }
    return out;
  }, [task, seqLen, hidden, seed]);

  const colors: Record<CellName, string> = {
    rnn: palette.css.classes[0] as string,
    gru: palette.css.classes[2] as string,
    lstm: palette.css.classes[1] as string,
  };

  const chart = useMemo(() => {
    const all = [
      ...CELL_NAMES.flatMap((c) => Array.from(initial[c])),
      ...(live ? Array.from(live) : []),
    ];
    const { lo, hi } = logRange(all, 3);
    const innerW = Math.max(10, width - PAD.left - PAD.right);
    const innerH = HEIGHT - PAD.top - PAD.bottom;
    const x = (t: number) => PAD.left + (seqLen === 1 ? 0 : (t / (seqLen - 1)) * innerW);
    const y = (v: number) => PAD.top + innerH - logPosition(v, lo, hi) * innerH;
    const path = (values: ArrayLike<number>) =>
      Array.from(
        values,
        (v, t) => `${t === 0 ? 'M' : 'L'}${x(t).toFixed(1)},${y(v).toFixed(1)}`,
      ).join(' ');
    const ticks: number[] = [];
    for (let e = lo; e <= hi; e += Math.max(1, Math.ceil((hi - lo) / 5))) ticks.push(e);
    return { lo, hi, x, y, path, ticks, innerW };
  }, [initial, live, width, seqLen]);

  const at = hover ?? seqLen - 1;

  return (
    <div className="space-y-4" ref={wrapRef}>
      <p className="max-w-3xl text-[0.9rem] text-muted">
        To learn from step 1, the error at the end has to travel back through every later step,
        multiplied again and again by the same weights. If those factors are below 1 the signal
        shrinks exponentially: this is the <em>vanishing gradient</em>. The chart shows how much
        signal reaches each step before any training: it is largest at the last step and fades
        towards the first.
      </p>
      <svg
        width="100%"
        height={HEIGHT}
        viewBox={`0 0 ${width} ${HEIGHT}`}
        role="img"
        aria-label="Learning signal at every step for a vanilla RNN, a GRU and an LSTM, on a log scale"
        onPointerMove={(e) => {
          const rect = e.currentTarget.getBoundingClientRect();
          const rel = (e.clientX - rect.left - PAD.left) / chart.innerW;
          setHover(Math.round(Math.min(1, Math.max(0, rel)) * (seqLen - 1)));
        }}
        onPointerLeave={() => setHover(null)}
      >
        {chart.ticks.map((e) => (
          <g key={e}>
            <line
              x1={PAD.left}
              x2={width - PAD.right}
              y1={chart.y(10 ** e)}
              y2={chart.y(10 ** e)}
              stroke="var(--line)"
              strokeDasharray="3 5"
            />
            <text
              x={PAD.left - 8}
              y={chart.y(10 ** e) + 3.5}
              textAnchor="end"
              className="fill-muted font-mono text-[10px]"
            >
              1e{e}
            </text>
          </g>
        ))}
        <text x={PAD.left} y={HEIGHT - 8} className="fill-faint font-mono text-[10px]">
          step 1
        </text>
        <text
          x={width - PAD.right}
          y={HEIGHT - 8}
          textAnchor="end"
          className="fill-faint font-mono text-[10px]"
        >
          step {seqLen}
        </text>
        {CELL_NAMES.map((name) => (
          <path
            key={name}
            d={chart.path(initial[name])}
            fill="none"
            stroke={colors[name]}
            strokeWidth={2.2}
            strokeLinejoin="round"
          />
        ))}
        {live && epoch > 0 && (
          <path
            d={chart.path(live)}
            fill="none"
            stroke={colors[cell]}
            strokeWidth={2}
            strokeDasharray="2 5"
            strokeLinecap="round"
          />
        )}
        {hover !== null && (
          <line
            x1={chart.x(hover)}
            x2={chart.x(hover)}
            y1={PAD.top}
            y2={HEIGHT - PAD.bottom}
            stroke="var(--line-strong)"
          />
        )}
      </svg>
      <div className="flex flex-wrap items-center gap-x-6 gap-y-2 font-mono text-[0.72rem]">
        {CELL_NAMES.map((name) => (
          <span key={name} className="flex items-center gap-2 text-muted">
            <span className="h-0.5 w-5 rounded" style={{ background: colors[name] }} />
            {cellInfo[name].label}
            <span className="text-fg tabular-nums">{fmt(initial[name][at] as number)}</span>
          </span>
        ))}
        {live && epoch > 0 && (
          <span className="flex items-center gap-2 text-muted">
            <span className="w-5 border-t-2 border-dotted" style={{ borderColor: colors[cell] }} />
            your {cellInfo[cell].label} after {epoch} epochs
            <span className="text-fg tabular-nums">{fmt(live[at] as number)}</span>
          </span>
        )}
        <span className="text-faint">· at step {at + 1}</span>
      </div>
      <p className="text-[0.78rem] text-faint">
        Gates give the LSTM and GRU a path where the memory is copied almost unchanged (forget gate
        near 1), so the signal decays far more slowly. Training also reshapes the curve: the dotted
        line is the network in the lab as it learns.
      </p>
    </div>
  );
}

const fmt = (v: number) => (v === 0 ? '0' : v.toExponential(1));
