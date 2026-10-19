import { useMemo, useState } from 'react';
import { Badge, Panel } from '@/design-system';
import { usePalette } from '@/shared/canvas/colors';
import { ExperimentBar } from '@/shared/components/ExperimentBar';
import {
  LabHeader,
  MetricsStrip,
  ParamPanel,
  StageNav,
  fix,
  pct,
  type Metric,
  type ParamValue,
  type ParamValues,
} from '@/shared/lab-kit';
import {
  DEFAULT_FORECAST,
  METHOD_LABELS,
  runForecast,
  type Accuracy,
  type ForecastRun,
  type ForecastScenario,
} from './forecastCore';
import { forecastExperiments, forecastParams, type ForecastExperiment } from './forecastSpec';

const toScenario = (p: ParamValues): ForecastScenario => ({
  series: p.series as ForecastScenario['series'],
  method: p.method as ForecastScenario['method'],
  log: Boolean(p.log),
  holdout: Number(p.holdout),
  p: Number(p.p),
  d: Number(p.d),
  q: Number(p.q),
  seasonalDiff: Boolean(p.seasonalDiff),
  seasonalMa: Boolean(p.seasonalMa),
});

const fromScenario = (s: ForecastScenario): ParamValues => ({ ...s });

function Chart({ run }: { run: ForecastRun }) {
  const palette = usePalette();
  const w = 880;
  const h = 340;
  const pad = { l: 44, r: 12, t: 12, b: 26 };
  const n = run.train.length + run.test.length;
  const all = [...run.train, ...run.test, ...run.lower, ...run.upper, ...run.forecast];
  const lo = Math.min(...all);
  const hi = Math.max(...all);
  const sx = (i: number) => pad.l + (i / (n - 1)) * (w - pad.l - pad.r);
  const sy = (v: number) => h - pad.b - ((v - lo) / (hi - lo || 1)) * (h - pad.t - pad.b);
  const nTrain = run.train.length;
  const line = (vals: (number | null)[], offset = 0) => {
    let d = '';
    vals.forEach((v, i) => {
      if (v === null) return;
      d += `${d && vals[i - 1] !== null ? 'L' : 'M'}${sx(i + offset).toFixed(1)},${sy(v).toFixed(1)}`;
    });
    return d;
  };
  const band =
    run.forecast.length > 0
      ? `${run.upper.map((v, k) => `${k ? 'L' : 'M'}${sx(nTrain + k).toFixed(1)},${sy(v).toFixed(1)}`).join('')}${[
          ...run.lower,
        ]
          .map((v, k) => [v, k] as const)
          .reverse()
          .map(([v, k]) => `L${sx(nTrain + k).toFixed(1)},${sy(v).toFixed(1)}`)
          .join('')}Z`
      : '';
  const yTicks = Array.from({ length: 5 }, (_, i) => lo + ((hi - lo) * i) / 4);
  const period = run.info.period;
  return (
    <div className="space-y-3">
      <svg
        viewBox={`0 0 ${w} ${h}`}
        className="w-full rounded-xl border border-line"
        role="img"
        aria-label="The series, its fitted values, the held-back months and the forecast with its band"
      >
        {yTicks.map((t) => (
          <g key={t}>
            <line
              x1={pad.l}
              x2={w - pad.r}
              y1={sy(t)}
              y2={sy(t)}
              stroke="currentColor"
              opacity={0.08}
            />
            <text
              x={pad.l - 6}
              y={sy(t) + 3}
              textAnchor="end"
              className="fill-faint font-mono text-[10px]"
            >
              {t.toFixed(0)}
            </text>
          </g>
        ))}
        {Array.from({ length: Math.floor(n / period) + 1 }, (_, k) => (
          <line
            key={k}
            x1={sx(k * period)}
            x2={sx(k * period)}
            y1={h - pad.b}
            y2={h - pad.b + 4}
            stroke="currentColor"
            opacity={0.3}
          />
        ))}
        <rect
          x={sx(nTrain - 0.5)}
          y={pad.t}
          width={sx(n - 1) - sx(nTrain - 0.5)}
          height={h - pad.t - pad.b}
          fill={palette.css.fg}
          opacity={0.04}
        />
        <path d={band} fill={palette.css.brand} opacity={0.2} />
        <path
          d={line(run.fitted)}
          fill="none"
          stroke={palette.css.muted}
          strokeWidth={1.2}
          opacity={0.8}
        />
        <path d={line(run.train)} fill="none" stroke={palette.css.fg} strokeWidth={1.8} />
        <path
          d={line(run.test, nTrain)}
          fill="none"
          stroke={palette.css.pos}
          strokeWidth={2}
          strokeDasharray="5 3"
        />
        <path
          d={line(run.forecast, nTrain)}
          fill="none"
          stroke={palette.css.brand}
          strokeWidth={2.6}
        />
        <text x={sx(nTrain) + 6} y={pad.t + 12} className="fill-muted font-mono text-[10px]">
          held back
        </text>
      </svg>
      <p className="flex flex-wrap gap-x-5 gap-y-1 text-[0.78rem] text-muted">
        <span>
          <span
            className="mr-1 inline-block h-0.5 w-5 align-middle"
            style={{ background: palette.css.fg }}
          />
          history
        </span>
        <span>
          <span
            className="mr-1 inline-block h-0.5 w-5 align-middle"
            style={{ background: palette.css.muted }}
          />
          one-step fit
        </span>
        <span>
          <span
            className="mr-1 inline-block h-0.5 w-5 align-middle"
            style={{ background: palette.css.pos }}
          />
          what actually happened
        </span>
        <span>
          <span
            className="mr-1 inline-block h-1 w-5 align-middle"
            style={{ background: palette.css.brand }}
          />
          forecast
        </span>
        <span>
          shaded: 95% band ({run.exactBand ? 'from the model' : 'rough, from past errors'})
        </span>
      </p>
    </div>
  );
}

function AccuracyTable({ run }: { run: ForecastRun }) {
  const rows: [string, Accuracy][] = [
    [METHOD_LABELS[run.scenario.method], run.accuracy],
    ['Naive (benchmark)', run.naive],
    ['Seasonal naive (benchmark)', run.seasonalNaive],
  ];
  return (
    <div className="overflow-x-auto">
      <table className="w-full min-w-[22rem] text-left font-mono text-[0.8rem] tabular-nums">
        <thead className="text-muted">
          <tr>
            <th className="pr-4 pb-2 font-sans font-medium">on the held-back months</th>
            <th className="pr-4 pb-2 font-medium">MAE</th>
            <th className="pr-4 pb-2 font-medium">RMSE</th>
            <th className="pr-4 pb-2 font-medium">MAPE</th>
          </tr>
        </thead>
        <tbody>
          {rows.map(([name, a], i) => (
            <tr key={name} className={`border-t border-line ${i === 0 ? 'text-fg' : 'text-muted'}`}>
              <td className="py-1.5 pr-4 font-sans">{name}</td>
              <td className="pr-4">{fix(a.mae, 1)}</td>
              <td className="pr-4">{fix(a.rmse, 1)}</td>
              <td className="pr-4" data-testid={i === 0 ? 'mape' : undefined}>
                {pct(a.mape)}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function Decomposition({ run }: { run: ForecastRun }) {
  const palette = usePalette();
  const d = run.decomposition;
  const panels: { title: string; values: (number | null)[]; color: string }[] = [
    { title: 'Trend', values: d.trend, color: palette.css.brand },
    { title: 'Seasonal pattern (ratio)', values: d.seasonal, color: palette.css.pos },
    { title: 'What is left (ratio)', values: d.residual, color: palette.css.neg },
  ];
  const w = 420;
  const h = 70;
  return (
    <div className="space-y-4">
      <div className="grid gap-x-8 gap-y-3 lg:grid-cols-3">
        {panels.map((p) => {
          const vals = p.values.filter((v): v is number => v !== null);
          const lo = Math.min(...vals);
          const hi = Math.max(...vals);
          const path = p.values
            .map((v, i) =>
              v === null
                ? ''
                : `${p.values[i - 1] === null || i === 0 ? 'M' : 'L'}${((i / (p.values.length - 1)) * (w - 4) + 2).toFixed(1)},${(h - 4 - ((v - lo) / (hi - lo || 1)) * (h - 8)).toFixed(1)}`,
            )
            .join('');
          return (
            <figure key={p.title}>
              <figcaption className="eyebrow mb-1">{p.title}</figcaption>
              <svg viewBox={`0 0 ${w} ${h}`} className="w-full" role="img" aria-label={p.title}>
                <path d={path} fill="none" stroke={p.color} strokeWidth={1.6} />
              </svg>
            </figure>
          );
        })}
      </div>
      <p className="text-sm text-muted">
        Classical decomposition splits a series into a slowly moving <em>trend</em> (a centred
        moving average over one year), a repeating <em>seasonal pattern</em> (the average ratio to
        the trend for each month) and a <em>remainder</em>. Because the swings grow with the level
        here, the multiplicative form is used. What remains should look like noise; structure left
        there is what the forecasting methods still have to catch.
      </p>
    </div>
  );
}

function Residuals({ run }: { run: ForecastRun }) {
  const palette = usePalette();
  const r = run.residualAcf.slice(1);
  const band = 1.96 / Math.sqrt(run.residuals.length);
  const w = 560;
  const h = 130;
  const mid = h / 2;
  const bw = (w - 20) / r.length;
  const white = run.ljungBox < 21;
  return (
    <div className="grid gap-8 lg:grid-cols-[minmax(0,1fr)_20rem]">
      <svg
        viewBox={`0 0 ${w} ${h}`}
        className="w-full"
        role="img"
        aria-label="Autocorrelation of the one-step errors at lags 1 to 24 with the white-noise band"
      >
        <line x1={10} x2={w - 10} y1={mid} y2={mid} stroke="currentColor" opacity={0.3} />
        <rect
          x={10}
          y={mid - band * (mid - 6)}
          width={w - 20}
          height={2 * band * (mid - 6)}
          fill={palette.css.brand}
          opacity={0.12}
        />
        {r.map((v, i) => (
          <rect
            key={i}
            x={10 + i * bw + 2}
            y={v >= 0 ? mid - v * (mid - 6) : mid}
            width={Math.max(2, bw - 4)}
            height={Math.abs(v) * (mid - 6)}
            fill={Math.abs(v) > band ? palette.css.neg : palette.css.muted}
          />
        ))}
      </svg>
      <div className="space-y-2 text-sm text-muted" data-testid="residual-note">
        <p>
          Bars: correlation between the one-step errors and the errors <em>k</em> months earlier.
          Red bars poke out of the shaded band, which is where white noise would stay 95% of the
          time.
        </p>
        <p>
          Ljung-Box statistic over 12 lags:{' '}
          <strong className="text-fg">{fix(run.ljungBox, 1)}</strong> (about 21 or less is
          consistent with noise).{' '}
          {white
            ? 'The errors carry no pattern the model could still use.'
            : 'There is still structure in the errors: the model is leaving information on the table.'}
        </p>
      </div>
    </div>
  );
}

export default function ForecastingPage() {
  const [params, setParams] = useState<ParamValues>(() => fromScenario(DEFAULT_FORECAST));
  const [activeId, setActiveId] = useState<string | null>(null);
  const scenario = useMemo(() => toScenario(params), [params]);
  const run = useMemo(() => runForecast(scenario), [scenario]);
  const setParam = (key: string, value: ParamValue) => setParams((p) => ({ ...p, [key]: value }));

  const apply = (exp: ForecastExperiment) => {
    setParams(fromScenario({ ...DEFAULT_FORECAST, ...exp.scenario }));
    setActiveId(exp.id);
  };
  const stillActive = (() => {
    const exp = forecastExperiments.find((e) => e.id === activeId);
    return (
      !!exp &&
      JSON.stringify(fromScenario({ ...DEFAULT_FORECAST, ...exp.scenario })) ===
        JSON.stringify(params)
    );
  })();

  const metrics: Metric[] = [
    {
      label: 'Error (MAPE)',
      value: pct(run.accuracy.mape),
      testId: 'mape-top',
      hint: 'mean absolute percentage error on the held-back months',
    },
    {
      label: 'Best benchmark',
      value: pct(Math.min(run.naive.mape, run.seasonalNaive.mape)),
      hint: 'the better of naive and seasonal naive',
    },
    {
      label: 'Inside the band',
      value: pct(run.coverage),
      hint: 'share of the held-back values that fell inside the 95% band',
    },
    {
      label: 'Residual check',
      value: run.ljungBox < 21 ? 'noise' : 'pattern left',
      hint: 'Ljung-Box on the one-step errors',
      tone: run.ljungBox < 21 ? 'good' : 'bad',
    },
    ...(run.aic !== null
      ? [
          {
            label: 'AIC',
            value: fix(run.aic, 0),
            hint: 'lower is better (only comparable on the same series and transform)',
          } as Metric,
        ]
      : []),
  ];

  const stages: [string, string][] = [
    ['lab-chart', 'The forecast'],
    ['lab-accuracy', 'Against benchmarks'],
    ['lab-decomp', 'Trend and season'],
    ['lab-resid', 'Residuals'],
    ['lab-how', 'How it works'],
    ['lab-controls', 'Settings'],
  ];

  return (
    <div className="mx-auto max-w-6xl px-4 pt-8 pb-20 sm:px-6 lg:px-8">
      <LabHeader
        eyebrow="Classic · probabilistic"
        title="Time-series forecasting"
        accent="trend, season and what is left"
        blurb="Hide the last months, forecast them, and see who was right. From naive benchmarks through exponential smoothing to ARIMA and the classic airline model, with decomposition and residual checks to say why."
        badges={
          <>
            <Badge>{run.info.label}</Badge>
            <Badge>{METHOD_LABELS[scenario.method]}</Badge>
            {scenario.log && <Badge>log scale</Badge>}
          </>
        }
      />
      <StageNav stages={stages} />
      <ExperimentBar
        lessons={forecastExperiments}
        activeId={stillActive ? activeId : null}
        onApply={apply}
      />

      <div className="flex flex-col gap-5">
        <Panel id="lab-chart" className="scroll-mt-24" title="The forecast" eyebrow="01">
          <div className="space-y-6">
            <p className="text-[0.8rem] text-muted">{run.info.note}</p>
            <MetricsStrip metrics={metrics} className="lg:grid-cols-5" />
            <Chart run={run} />
          </div>
        </Panel>
        <Panel id="lab-accuracy" className="scroll-mt-24" title="Against benchmarks" eyebrow="02">
          <div className="space-y-3">
            <AccuracyTable run={run} />
            <p className="text-sm text-muted">
              A forecast is only good relative to something simple. A method that cannot beat
              &ldquo;same as last month&rdquo; or &ldquo;same as a year ago&rdquo; is not adding
              anything, however sophisticated it is. All errors are measured on months the model
              never saw.
            </p>
          </div>
        </Panel>
        <Panel id="lab-decomp" className="scroll-mt-24" title="Trend and season" eyebrow="03">
          <Decomposition run={run} />
        </Panel>
        <Panel id="lab-resid" className="scroll-mt-24" title="Residuals" eyebrow="04">
          <Residuals run={run} />
        </Panel>
        <Panel id="lab-how" className="scroll-mt-24" title="How it works" eyebrow="05">
          <div className="space-y-3 text-sm text-muted">
            <p>
              <strong className="text-fg">Exponential smoothing</strong> keeps a running estimate of
              the level (and optionally the trend and the seasonal pattern) and updates it as a
              weighted blend of the new observation and the old estimate. The weights are chosen by
              minimising the one-step squared error.
            </p>
            <p>
              <strong className="text-fg">ARIMA(p, d, q)</strong> models the series after d
              differences as a combination of its p previous values (autoregression) and q previous
              forecast errors (moving average). Differencing at lag 12 removes a stable seasonal
              pattern; a seasonal moving-average term completes the airline model. The fit here
              minimises the conditional sum of squares, and the 95% band comes from the model&apos;s
              own variance formula, which grows with the horizon.
            </p>
            <p>
              Tests compare autocorrelation, decomposition, exponential smoothing and the airline
              model with statsmodels.
            </p>
          </div>
        </Panel>
        <Panel id="lab-controls" className="scroll-mt-24" title="Settings" eyebrow="06">
          <ParamPanel specs={forecastParams} values={params} onChange={setParam} columns={3} />
        </Panel>
      </div>
    </div>
  );
}
