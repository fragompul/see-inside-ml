import { fix } from './MetricsStrip';

export interface CoefficientRow {
  name: string;
  value: number;
  /** A confidence interval (or any uncertainty band). */
  lo?: number;
  hi?: number;
  /** The true value, when the data are synthetic. */
  truth?: number;
}

const W = 520;
const ROW = 34;
const LEFT = 92;
const RIGHT = 16;

/** Horizontal forest plot: a dot per coefficient, its interval, and the true value as a tick. */
export function CoefficientPlot({
  rows,
  ariaLabel,
}: {
  rows: CoefficientRow[];
  ariaLabel: string;
}) {
  if (rows.length === 0) return null;
  const vals = rows.flatMap((r) => [
    r.value,
    r.lo ?? r.value,
    r.hi ?? r.value,
    r.truth ?? r.value,
    0,
  ]);
  let lo = Math.min(...vals);
  let hi = Math.max(...vals);
  const pad = (hi - lo || 1) * 0.08;
  lo -= pad;
  hi += pad;
  const sx = (v: number) => LEFT + ((v - lo) / (hi - lo)) * (W - LEFT - RIGHT);
  const H = rows.length * ROW + 34;
  const ticks = Array.from({ length: 5 }, (_, i) => lo + ((hi - lo) * i) / 4);
  return (
    <figure className="space-y-2">
      <svg
        viewBox={`0 0 ${W} ${H}`}
        role="img"
        aria-label={ariaLabel}
        className="block w-full max-w-[34rem] overflow-visible"
      >
        {ticks.map((t) => (
          <g key={t}>
            <line x1={sx(t)} x2={sx(t)} y1={4} y2={H - 26} className="stroke-line" />
            <text
              x={sx(t)}
              y={H - 10}
              textAnchor="middle"
              className="fill-faint font-mono text-[10px]"
            >
              {t.toFixed(1)}
            </text>
          </g>
        ))}
        <line x1={sx(0)} x2={sx(0)} y1={4} y2={H - 26} className="stroke-line-strong" />
        {rows.map((r, i) => {
          const y = 22 + i * ROW;
          return (
            <g key={r.name}>
              <text
                x={LEFT - 10}
                y={y + 4}
                textAnchor="end"
                className="fill-fg font-mono text-[11px]"
              >
                {r.name}
              </text>
              {r.lo !== undefined && r.hi !== undefined && (
                <line
                  x1={sx(r.lo)}
                  x2={sx(r.hi)}
                  y1={y}
                  y2={y}
                  strokeWidth={3}
                  strokeLinecap="round"
                  className="stroke-brand/40"
                />
              )}
              {r.truth !== undefined && (
                <path
                  d={`M${sx(r.truth)} ${y - 9} L${sx(r.truth)} ${y + 9}`}
                  strokeWidth={2}
                  className="stroke-fg"
                />
              )}
              <circle cx={sx(r.value)} cy={y} r={5} className="fill-brand">
                <title>{`${r.name} = ${fix(r.value)}`}</title>
              </circle>
            </g>
          );
        })}
      </svg>
      <figcaption className="text-[0.74rem] text-faint">
        Dots: fitted coefficients
        {rows.some((r) => r.lo !== undefined) ? ', bars: 95% intervals' : ''}
        {rows.some((r) => r.truth !== undefined) ? ', black ticks: the true values' : ''}.
      </figcaption>
    </figure>
  );
}

/** Predicted vs actual targets: the closer to the diagonal, the better; hollow dots are held out. */
export function PredictedActual({
  train,
  test,
}: {
  train: { actual: ArrayLike<number>; predicted: ArrayLike<number> };
  test: { actual: ArrayLike<number>; predicted: ArrayLike<number> } | null;
}) {
  const size = 300;
  const pad = 34;
  const span = size - pad - 10;
  const all = [
    ...Array.from(train.actual),
    ...Array.from(train.predicted),
    ...(test ? [...Array.from(test.actual), ...Array.from(test.predicted)] : []),
  ];
  let lo = Math.min(...all);
  let hi = Math.max(...all);
  const m = (hi - lo || 1) * 0.06;
  lo -= m;
  hi += m;
  const px = (v: number) => pad + ((v - lo) / (hi - lo)) * span;
  const py = (v: number) => size - pad - ((v - lo) / (hi - lo)) * span;
  const dots = (
    set: { actual: ArrayLike<number>; predicted: ArrayLike<number> },
    hollow: boolean,
  ) =>
    Array.from(set.actual, (a, i) => (
      <circle
        key={`${hollow ? 't' : 'r'}${i}`}
        cx={px(a)}
        cy={py(set.predicted[i] as number)}
        r={3.4}
        className={hollow ? 'fill-surface stroke-pos' : 'fill-brand/70 stroke-surface'}
        strokeWidth={hollow ? 1.6 : 0.8}
      />
    ));
  return (
    <figure className="space-y-2">
      <svg
        viewBox={`0 0 ${size} ${size}`}
        role="img"
        aria-label="Predicted against actual values"
        className="block w-full max-w-[22rem]"
      >
        <rect x={pad} y={10} width={span} height={span} className="fill-none stroke-line" />
        <line
          x1={px(lo)}
          y1={py(lo)}
          x2={px(hi)}
          y2={py(hi)}
          strokeDasharray="4 4"
          className="stroke-faint"
        />
        {dots(train, false)}
        {test && dots(test, true)}
        <text
          x={size / 2}
          y={size - 4}
          textAnchor="middle"
          className="fill-faint font-mono text-[10px]"
        >
          actual
        </text>
        <text
          x={10}
          y={size / 2}
          textAnchor="middle"
          transform={`rotate(-90 10 ${size / 2})`}
          className="fill-faint font-mono text-[10px]"
        >
          predicted
        </text>
      </svg>
      <figcaption className="text-[0.74rem] text-faint">
        Dots on the dashed diagonal are perfect predictions. Filled: training rows, hollow: held
        out.
      </figcaption>
    </figure>
  );
}
