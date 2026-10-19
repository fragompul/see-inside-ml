import { ACTION_COUNT, parseGrid } from '@sim/deep';
import { useMemo, useRef, type PointerEvent } from 'react';
import { cn } from '@/design-system';
import type { Snapshot, Tool } from '../store';
import {
  ACTION_VECTOR,
  actionTriangle,
  bestActions,
  cellAt,
  cellXY,
  normalise,
  valueRange,
} from '../view';

export type ViewMode = 'values' | 'policy' | 'q' | 'visits';
export type PaintTool = Tool | 'look';

const CELL = 48;

/** `t` from 0 (frost) to 1 (ember) as a theme-aware colour. */
const heat = (t: number, alpha = 0.7) =>
  `color-mix(in srgb, color-mix(in srgb, var(--pos) ${Math.round(t * 100)}%, var(--neg)) ${Math.round(alpha * 100)}%, transparent)`;

/** The gridworld with the agent, what it has learned and a paintable layout. */
export function GridView({
  rows,
  snapshot,
  mode,
  showPath,
  tool,
  inspected,
  onPaint,
  onInspect,
}: {
  rows: string[];
  snapshot: Snapshot;
  mode: ViewMode;
  showPath: boolean;
  tool: PaintTool;
  inspected: number | null;
  onPaint: (cell: number, kind: Tool) => void;
  onInspect: (cell: number | null) => void;
}) {
  const layout = useMemo(() => parseGrid(rows), [rows]);
  const { width, height, kinds } = layout;
  const { q, visits, agent, path } = snapshot;
  const pressed = useRef(false);
  const last = useRef<number | null>(null);

  const solid = (s: number) => kinds[s] === 'wall' || kinds[s] === 'goal' || kinds[s] === 'pit';
  const values = useMemo(
    () =>
      Float64Array.from({ length: width * height }, (_, s) =>
        Math.max(
          ...Array.from({ length: ACTION_COUNT }, (_, a) => q[s * ACTION_COUNT + a] as number),
        ),
      ),
    [q, width, height],
  );
  const vRange = useMemo(() => valueRange(values, solid), [values]); // eslint-disable-line react-hooks/exhaustive-deps
  const qRange = useMemo(() => valueRange(q, (i) => solid(Math.floor(i / ACTION_COUNT))), [q]); // eslint-disable-line react-hooks/exhaustive-deps
  const maxVisits = useMemo(() => Math.max(1, ...visits), [visits]);
  const learned = q.some((v) => v !== 0);

  const act = (event: PointerEvent<SVGSVGElement>, first: boolean) => {
    const rect = event.currentTarget.getBoundingClientRect();
    const cell = cellAt(
      ((event.clientX - rect.left) / rect.width) * width * CELL,
      ((event.clientY - rect.top) / rect.height) * height * CELL,
      width,
      height,
      CELL,
    );
    if (cell === null || cell === last.current) return;
    last.current = cell;
    if (tool === 'look') {
      if (first) onInspect(inspected === cell ? null : cell);
    } else onPaint(cell, tool);
  };

  return (
    <svg
      viewBox={`0 0 ${width * CELL} ${height * CELL}`}
      className={cn(
        'block w-full touch-none rounded-xl border border-line-strong bg-surface select-none',
        tool === 'look' ? 'cursor-pointer' : 'cursor-crosshair',
      )}
      style={{ maxWidth: width * CELL }}
      role="img"
      aria-label="The gridworld: the agent, what it has learned in each cell and the layout, which can be painted"
      onPointerDown={(e) => {
        pressed.current = true;
        last.current = null;
        e.currentTarget.setPointerCapture(e.pointerId);
        act(e, true);
      }}
      onPointerMove={(e) => pressed.current && tool !== 'look' && act(e, false)}
      onPointerUp={() => {
        pressed.current = false;
        last.current = null;
      }}
      onPointerCancel={() => {
        pressed.current = false;
        last.current = null;
      }}
    >
      {kinds.map((kind, s) => {
        const [cx, cy] = cellXY(s, width);
        const x = cx * CELL;
        const y = cy * CELL;
        const isInspected = inspected === s;
        return (
          <g key={s} transform={`translate(${x} ${y})`}>
            <rect
              width={CELL}
              height={CELL}
              fill={kind === 'wall' ? 'var(--raised)' : 'var(--surface)'}
              stroke="var(--line)"
              strokeWidth={1}
            />
            {kind === 'wall' && (
              <path
                d={`M4 ${CELL - 4} L${CELL - 4} 4 M4 ${CELL / 2} L${CELL / 2} 4 M${CELL / 2} ${CELL - 4} L${CELL - 4} ${CELL / 2}`}
                stroke="var(--line-strong)"
                strokeWidth={2}
              />
            )}
            {kind === 'goal' && <rect width={CELL} height={CELL} fill="var(--pos)" opacity={0.3} />}
            {kind === 'pit' && <rect width={CELL} height={CELL} fill="var(--neg)" opacity={0.35} />}

            {!solid(s) && mode === 'values' && (
              <>
                <rect
                  width={CELL}
                  height={CELL}
                  fill={learned ? heat(normalise(values[s] as number, vRange)) : 'transparent'}
                />
                {learned && (
                  <text
                    x={CELL / 2}
                    y={CELL / 2 + 4}
                    textAnchor="middle"
                    className="fill-fg font-mono text-[10px]"
                  >
                    {(values[s] as number).toFixed(0)}
                  </text>
                )}
              </>
            )}
            {!solid(s) && mode === 'visits' && (
              <rect
                width={CELL}
                height={CELL}
                fill={heat(0.95, 0.9 * Math.sqrt((visits[s] as number) / maxVisits))}
              />
            )}
            {!solid(s) &&
              mode === 'q' &&
              Array.from({ length: ACTION_COUNT }, (_, a) => (
                <polygon
                  key={a}
                  points={actionTriangle(a, CELL, 1)}
                  fill={
                    learned
                      ? heat(normalise(q[s * ACTION_COUNT + a] as number, qRange), 0.75)
                      : 'transparent'
                  }
                  stroke="var(--line)"
                  strokeWidth={0.6}
                />
              ))}
            {!solid(s) &&
              mode === 'policy' &&
              bestActions(q, s).map((a) => {
                const [dx, dy] = ACTION_VECTOR[a] as readonly [number, number];
                const c = CELL / 2;
                const tip: [number, number] = [c + dx * 15, c + dy * 15];
                const tail: [number, number] = [c - dx * 13, c - dy * 13];
                const px = -dy;
                const py = dx;
                return (
                  <g
                    key={a}
                    stroke="var(--brand)"
                    fill="var(--brand)"
                    strokeWidth={2.4}
                    strokeLinecap="round"
                  >
                    <line x1={tail[0]} y1={tail[1]} x2={tip[0]} y2={tip[1]} />
                    <polygon
                      points={`${tip[0] + dx * 4},${tip[1] + dy * 4} ${tip[0] - dx * 6 + px * 6},${tip[1] - dy * 6 + py * 6} ${tip[0] - dx * 6 - px * 6},${tip[1] - dy * 6 - py * 6}`}
                      strokeWidth={1}
                    />
                  </g>
                );
              })}

            {kind === 'goal' && (
              <text
                x={CELL / 2}
                y={CELL / 2 + 7}
                textAnchor="middle"
                className="fill-fg text-[20px]"
              >
                ★
              </text>
            )}
            {kind === 'pit' && (
              <text
                x={CELL / 2}
                y={CELL / 2 + 6}
                textAnchor="middle"
                className="fill-fg text-[16px] font-bold"
              >
                ✕
              </text>
            )}
            {kind === 'start' && (
              <text x={5} y={13} className="fill-muted font-mono text-[10px] font-semibold">
                S
              </text>
            )}
            {isInspected && (
              <rect
                x={1.5}
                y={1.5}
                width={CELL - 3}
                height={CELL - 3}
                fill="none"
                stroke="var(--fg)"
                strokeWidth={2.5}
              />
            )}
          </g>
        );
      })}

      {showPath && path.cells.length > 1 && (
        <polyline
          points={path.cells
            .map((s) => {
              const [cx, cy] = cellXY(s, width);
              return `${cx * CELL + CELL / 2},${cy * CELL + CELL / 2}`;
            })
            .join(' ')}
          fill="none"
          stroke="var(--fg)"
          strokeWidth={3}
          strokeDasharray="2 6"
          strokeLinecap="round"
          opacity={0.85}
        />
      )}

      {(() => {
        const [ax, ay] = cellXY(agent, width);
        return (
          <circle
            cx={ax * CELL + CELL / 2}
            cy={ay * CELL + CELL / 2}
            r={9}
            fill="var(--brand)"
            stroke="var(--bg)"
            strokeWidth={2.5}
            className="transition-all duration-100"
          />
        );
      })()}
    </svg>
  );
}
