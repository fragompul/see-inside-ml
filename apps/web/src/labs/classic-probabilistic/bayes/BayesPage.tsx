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
  DEFAULT_BAYES,
  MODEL_INFO,
  runBayes,
  type BayesRun,
  type BayesScenario,
} from './bayesCore';
import { bayesExperiments, bayesParams, type BayesExperiment } from './bayesSpec';

const toScenario = (p: ParamValues): BayesScenario => ({
  model: p.model as BayesScenario['model'],
  priorCenter: { coin: Number(p.priorCoin), mean: Number(p.priorMean), rate: Number(p.priorRate) },
  priorStrength: Number(p.priorStrength),
  n: Number(p.n),
  observed: { coin: Number(p.obsCoin), mean: Number(p.obsMean), rate: Number(p.obsRate) },
});

const fromScenario = (s: BayesScenario): ParamValues => ({
  model: s.model,
  priorCoin: s.priorCenter.coin,
  priorMean: s.priorCenter.mean,
  priorRate: s.priorCenter.rate,
  priorStrength: s.priorStrength,
  n: s.n,
  obsCoin: s.observed.coin,
  obsMean: s.observed.mean,
  obsRate: s.observed.rate,
});

function Curves({ run }: { run: BayesRun }) {
  const palette = usePalette();
  const w = 760;
  const h = 300;
  const pad = { l: 12, r: 12, t: 14, b: 30 };
  const { grid } = run;
  const x0 = grid[0] as number;
  const x1 = grid[grid.length - 1] as number;
  const top =
    Math.max(...run.posteriorCurve, ...run.priorCurve, ...run.likelihoodCurve) * 1.08 || 1;
  const sx = (v: number) => pad.l + ((v - x0) / (x1 - x0)) * (w - pad.l - pad.r);
  const sy = (v: number) => h - pad.b - (Math.min(v, top) / top) * (h - pad.t - pad.b);
  const line = (ys: number[]) =>
    ys
      .map((y, i) => `${i ? 'L' : 'M'}${sx(grid[i] as number).toFixed(1)},${sy(y).toFixed(1)}`)
      .join(' ');
  const [lo, hi] = run.interval;
  const inside = grid.map((g, i) => (g >= lo && g <= hi ? i : -1)).filter((i) => i >= 0);
  const bandPath =
    inside.length > 1
      ? inside
          .map(
            (i, k) =>
              `${k ? 'L' : 'M'}${sx(grid[i] as number).toFixed(1)},${sy(run.posteriorCurve[i] as number).toFixed(1)}`,
          )
          .join(' ') +
        ` L${sx(grid[inside[inside.length - 1]!] as number).toFixed(1)},${sy(0).toFixed(1)} L${sx(grid[inside[0]!] as number).toFixed(1)},${sy(0).toFixed(1)} Z`
      : '';
  const symbol = MODEL_INFO[run.scenario.model].symbol;
  const ticks = Array.from({ length: 6 }, (_, i) => x0 + ((x1 - x0) * i) / 5);
  return (
    <div className="space-y-3">
      <svg
        viewBox={`0 0 ${w} ${h}`}
        className="w-full rounded-xl border border-line"
        role="img"
        aria-label="The prior belief, what the data alone say, and the posterior belief, as curves over the unknown quantity"
      >
        <line x1={pad.l} x2={w - pad.r} y1={sy(0)} y2={sy(0)} stroke="currentColor" opacity={0.2} />
        {ticks.map((t) => (
          <g key={t}>
            <line
              x1={sx(t)}
              x2={sx(t)}
              y1={sy(0)}
              y2={sy(0) + 4}
              stroke="currentColor"
              opacity={0.3}
            />
            <text
              x={sx(t)}
              y={h - 10}
              textAnchor="middle"
              className="fill-faint font-mono text-[10px]"
            >
              {Math.abs(t) >= 10 ? t.toFixed(0) : t.toFixed(2).replace(/\.?0+$/, '')}
            </text>
          </g>
        ))}
        <path d={bandPath} fill={palette.css.brand} opacity={0.18} />
        <path
          d={line(run.priorCurve)}
          fill="none"
          stroke={palette.css.muted}
          strokeWidth={2}
          strokeDasharray="6 4"
        />
        <path d={line(run.likelihoodCurve)} fill="none" stroke={palette.css.pos} strokeWidth={2} />
        <path d={line(run.posteriorCurve)} fill="none" stroke={palette.css.brand} strokeWidth={3} />
        <line
          x1={sx(run.mle)}
          x2={sx(run.mle)}
          y1={sy(0)}
          y2={sy(top * 0.5)}
          stroke={palette.css.pos}
          strokeWidth={1.5}
        />
        <circle cx={sx(run.mle)} cy={sy(top * 0.5)} r={4} fill={palette.css.pos} />
      </svg>
      <div className="flex flex-wrap gap-x-6 gap-y-1 text-[0.78rem] text-muted">
        <span>
          <span
            className="mr-1 inline-block h-0.5 w-5 align-middle"
            style={{ background: palette.css.muted }}
          />
          prior: what you believed
        </span>
        <span>
          <span
            className="mr-1 inline-block h-0.5 w-5 align-middle"
            style={{ background: palette.css.pos }}
          />
          data alone (likelihood, rescaled)
        </span>
        <span>
          <span
            className="mr-1 inline-block h-1 w-5 align-middle"
            style={{ background: palette.css.brand }}
          />
          posterior: what you should believe now
        </span>
        <span>shaded: 95% credible interval for {symbol}</span>
      </div>
    </div>
  );
}

function Predictive({ run }: { run: BayesRun }) {
  const palette = usePalette();
  if (!run.predictive) return null;
  const max = Math.max(...run.predictive);
  return (
    <div className="space-y-3">
      <div
        className="flex h-28 items-end gap-1.5"
        role="img"
        aria-label="Probability of each number of heads in the next ten flips"
      >
        {run.predictive.map((p, k) => (
          <div key={k} className="flex flex-1 flex-col items-center gap-1">
            <div
              className="w-full rounded-t"
              style={{
                height: `${Math.max(2, (p / max) * 100)}%`,
                background: palette.css.brand,
                opacity: 0.8,
              }}
              title={`${k} heads: ${pct(p)}`}
            />
            <span className="font-mono text-[0.68rem] text-faint">{k}</span>
          </div>
        ))}
      </div>
      <p className="text-sm text-muted">
        What to expect in the next ten flips, averaging over everything you believe about the coin
        (the <em>posterior predictive</em>). It is wider than a plain binomial with the best guess,
        because you are also unsure about the coin itself.
      </p>
    </div>
  );
}

export default function BayesPage() {
  const [params, setParams] = useState<ParamValues>(() => fromScenario(DEFAULT_BAYES));
  const [activeId, setActiveId] = useState<string | null>(null);
  const scenario = useMemo(() => toScenario(params), [params]);
  const run = useMemo(() => runBayes(scenario), [scenario]);
  const setParam = (key: string, value: ParamValue) => setParams((p) => ({ ...p, [key]: value }));
  const info = MODEL_INFO[scenario.model];

  const apply = (exp: BayesExperiment) => {
    setParams(fromScenario(exp.scenario));
    setActiveId(exp.id);
  };
  const stillActive = (() => {
    const exp = bayesExperiments.find((e) => e.id === activeId);
    return !!exp && JSON.stringify(fromScenario(exp.scenario)) === JSON.stringify(params);
  })();

  const metrics: Metric[] = [
    {
      label: 'Data alone say',
      value: scenario.n === 0 ? '–' : fix(run.mle, 2),
      testId: 'mle',
      hint: 'maximum-likelihood estimate',
    },
    { label: 'Prior said', value: fix(run.prior.mean, 2), hint: 'mean of the prior' },
    {
      label: 'Now believed',
      value: fix(run.posterior.mean, 2),
      testId: 'posterior-mean',
      hint: 'mean of the posterior',
    },
    {
      label: '95% credible interval',
      value: `${fix(run.interval[0], 2)} to ${fix(run.interval[1], 2)}`,
      testId: 'interval',
    },
    {
      label: 'Prior counts for',
      value: pct(run.priorWeight),
      hint: 'share of the belief that still comes from the prior',
    },
  ];

  const stages: [string, string][] = [
    ['lab-curves', 'Updating a belief'],
    ...(run.predictive ? ([['lab-predictive', 'What next?']] as [string, string][]) : []),
    ['lab-how', 'How it works'],
    ['lab-controls', 'Settings'],
  ];

  return (
    <div className="mx-auto max-w-6xl px-4 pt-8 pb-20 sm:px-6 lg:px-8">
      <LabHeader
        eyebrow="Classic · probabilistic"
        title="Bayesian inference"
        accent="beliefs that learn"
        blurb="Start from a prior belief, look at data, end with a posterior: the prior reweighted by how well each possibility explains what you saw. With the right pairs of distributions the whole update is one line of arithmetic."
        badges={
          <>
            <Badge>{info.label}</Badge>
            <Badge>{scenario.n} observations</Badge>
          </>
        }
      />
      <StageNav stages={stages} />
      <ExperimentBar
        lessons={bayesExperiments}
        activeId={stillActive ? activeId : null}
        onApply={apply}
      />

      <div className="flex flex-col gap-5">
        <Panel id="lab-curves" className="scroll-mt-24" title="Updating a belief" eyebrow="01">
          <div className="space-y-6">
            <p className="text-[0.8rem] text-muted">{info.blurb}</p>
            <MetricsStrip metrics={metrics} className="lg:grid-cols-5" />
            <Curves run={run} />
          </div>
        </Panel>
        {run.predictive && (
          <Panel
            id="lab-predictive"
            className="scroll-mt-24"
            title="What happens next?"
            eyebrow="02"
          >
            <Predictive run={run} />
          </Panel>
        )}
        <Panel
          id="lab-how"
          className="scroll-mt-24"
          title="How it works"
          eyebrow={run.predictive ? '03' : '02'}
        >
          <div className="space-y-3 text-sm text-muted">
            <p>
              Bayes&apos; rule multiplies the <strong className="text-fg">prior</strong> by the{' '}
              <strong className="text-fg">likelihood</strong> (how probable the data are for each
              possible value) and renormalises. For the pairs used here, the answer is in the same
              family as the prior, only with new parameters:
            </p>
            <ul className="list-disc space-y-1 pl-5">
              <li>
                Beta(a, b) prior + h heads and t tails gives Beta(a + h, b + t): the prior is simply
                a + b imaginary flips.
              </li>
              <li>
                Normal prior + n measurements of known spread: precisions add, and the new mean is
                the precision-weighted average of the prior mean and the data mean.
              </li>
              <li>
                Gamma(shape, rate) prior + n counts totalling T gives Gamma(shape + T, rate + n).
              </li>
            </ul>
            <p>
              Since only the sums matter, evidence can be added in any order or all at once, and as
              the data grow the prior&apos;s share, strength / (strength + n), fades. A credible
              interval is the range that holds 95% of the belief; unlike a frequentist confidence
              interval, it says what it sounds like. Tests check the posterior moments against
              numerical integration and quantiles against the CDFs.
            </p>
          </div>
        </Panel>
        <Panel
          id="lab-controls"
          className="scroll-mt-24"
          title="Settings"
          eyebrow={run.predictive ? '04' : '03'}
        >
          <ParamPanel specs={bayesParams} values={params} onChange={setParam} columns={3} />
        </Panel>
      </div>
    </div>
  );
}
