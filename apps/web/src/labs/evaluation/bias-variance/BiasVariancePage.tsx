import { useDeferredValue, useMemo, useState } from 'react';
import { Badge, Panel } from '@/design-system';
import { usePalette } from '@/shared/canvas/colors';
import { ExperimentBar } from '@/shared/components/ExperimentBar';
import { MiniChart } from '@/shared/components/MiniChart';
import {
  LabHeader,
  MetricsStrip,
  ParamPanel,
  StageNav,
  fix,
  type Metric,
  type ParamValue,
  type ParamValues,
} from '@/shared/lab-kit';
import {
  DEFAULT_BV,
  GRID,
  WORLDS,
  complexityCurve,
  learningCurve,
  simulate,
  type BiasVarianceRun,
  type BiasVarianceScenario,
} from './biasVarianceCore';
import { bvExperiments, bvParams, type BvExperiment } from './biasVarianceSpec';

const toScenario = (p: ParamValues): BiasVarianceScenario => ({
  ...DEFAULT_BV,
  world: p.world as BiasVarianceScenario['world'],
  features: Number(p.features),
  n: Number(p.n),
  noise: Number(p.noise),
  ridge: Number(p.ridge),
});

const fromScenario = (s: BiasVarianceScenario): ParamValues => ({
  world: s.world,
  features: s.features,
  n: s.n,
  noise: s.noise,
  ridge: s.ridge,
});

function Fits({ run }: { run: BiasVarianceRun }) {
  const palette = usePalette();
  const w = 640;
  const h = 360;
  const pad = 14;
  const all = [...run.truth, ...run.sample.y, ...run.curves.flat()];
  // robust vertical range: wild fits are clipped instead of flattening the picture
  const lo = Math.max(Math.min(...run.truth, ...run.sample.y) - 0.6, -3.5);
  const hi = Math.min(Math.max(...run.truth, ...run.sample.y) + 0.6, 3.5);
  void all;
  const sx = (v: number) => pad + v * (w - 2 * pad);
  const sy = (v: number) =>
    h - pad - ((Math.min(Math.max(v, lo - 1), hi + 1) - lo) / (hi - lo)) * (h - 2 * pad);
  const path = (ys: number[]) =>
    ys
      .map((y, i) => `${i ? 'L' : 'M'}${sx(GRID[i] as number).toFixed(1)},${sy(y).toFixed(1)}`)
      .join(' ');
  return (
    <div className="space-y-3">
      <svg
        viewBox={`0 0 ${w} ${h}`}
        className="w-full rounded-xl border border-line"
        role="img"
        aria-label="Twelve fitted curves from twelve different training sets, their average, the truth and one training set"
      >
        <defs>
          <clipPath id="bv-clip">
            <rect x={0} y={0} width={w} height={h} />
          </clipPath>
        </defs>
        <g clipPath="url(#bv-clip)">
          {run.curves.map((c, i) => (
            <path
              key={i}
              d={path(c)}
              fill="none"
              stroke={palette.css.brand}
              strokeWidth={1.2}
              opacity={0.35}
            />
          ))}
          <path
            d={path(run.truth)}
            fill="none"
            stroke={palette.css.fg}
            strokeWidth={2.2}
            strokeDasharray="6 4"
          />
          <path d={path(run.meanCurve)} fill="none" stroke={palette.css.neg} strokeWidth={2.4} />
          {run.sample.x.map((x, i) => (
            <circle
              key={i}
              cx={sx(x)}
              cy={sy(run.sample.y[i] as number)}
              r={3}
              fill={palette.css.muted}
              opacity={0.8}
            />
          ))}
        </g>
      </svg>
      <p className="flex flex-wrap gap-x-5 gap-y-1 text-[0.78rem] text-muted">
        <span>
          <span
            className="mr-1 inline-block h-0.5 w-5 align-middle"
            style={{ background: palette.css.brand }}
          />
          fits from 12 different training sets
        </span>
        <span>
          <span
            className="mr-1 inline-block h-0.5 w-5 align-middle"
            style={{ background: palette.css.neg }}
          />
          their average
        </span>
        <span>
          <span
            className="mr-1 inline-block h-0.5 w-5 align-middle"
            style={{ background: palette.css.fg }}
          />
          the truth
        </span>
        <span>
          <span
            className="mr-1 inline-block size-2 rounded-full align-middle"
            style={{ background: palette.css.muted }}
          />
          one training set
        </span>
      </p>
    </div>
  );
}

function Decomposition({ run }: { run: BiasVarianceRun }) {
  const palette = usePalette();
  const total = run.bias2 + run.variance + run.noiseVariance;
  const parts: [string, number, string][] = [
    ['Bias²', run.bias2, palette.css.neg],
    ['Variance', run.variance, palette.css.brand],
    ['Noise', run.noiseVariance, palette.css.muted],
  ];
  return (
    <div className="space-y-4">
      <div
        className="flex h-8 overflow-hidden rounded-lg border border-line"
        role="img"
        aria-label="Test error split into bias squared, variance and noise"
      >
        {parts.map(([name, v, color]) => (
          <div
            key={name}
            style={{ width: `${(v / total) * 100}%`, background: color, opacity: 0.8 }}
            title={`${name}: ${v.toFixed(3)}`}
          />
        ))}
      </div>
      <dl className="grid grid-cols-3 gap-3 text-sm">
        {parts.map(([name, v, color]) => (
          <div key={name}>
            <dt className="flex items-center gap-2 text-muted">
              <span className="inline-block size-2.5 rounded-sm" style={{ background: color }} />
              {name}
            </dt>
            <dd className="font-mono tabular-nums">{v < 100 ? fix(v, 3) : v.toExponential(1)}</dd>
          </div>
        ))}
      </dl>
      <p className="text-sm text-muted">
        Expected test error = <strong className="text-fg">bias²</strong> (how far the average fit is
        from the truth) + <strong className="text-fg">variance</strong> (how much fits scatter
        around their own average) + <strong className="text-fg">noise</strong> (what no model can
        predict). This is an identity, checked in the tests, and it can be measured here because the
        truth is known.
      </p>
    </div>
  );
}

export default function BiasVariancePage() {
  const [params, setParams] = useState<ParamValues>(() => fromScenario(DEFAULT_BV));
  const [activeId, setActiveId] = useState<string | null>(null);
  const scenario = useMemo(() => toScenario(params), [params]);
  const run = useMemo(() => simulate(scenario), [scenario]);
  const deferred = useDeferredValue(scenario);
  const setParam = (key: string, value: ParamValue) => setParams((p) => ({ ...p, [key]: value }));

  const curve = useMemo(() => {
    const n = deferred.n;
    const maxP = n <= 20 ? Math.min(3 * n, 60) : n <= 40 ? 40 : 25;
    return complexityCurve({ ...deferred, datasets: n >= 60 ? 8 : 20 }, maxP);
  }, [deferred.world, deferred.n, deferred.noise, deferred.ridge, deferred.seed]); // eslint-disable-line react-hooks/exhaustive-deps
  const learning = useMemo(
    () =>
      learningCurve(
        { ...deferred, datasets: 15 },
        [10, 15, 20, 30, 50, 80, 120].filter((n) => deferred.features <= n + 3 || true),
      ),
    [deferred.world, deferred.features, deferred.noise, deferred.ridge, deferred.seed], // eslint-disable-line react-hooks/exhaustive-deps
  );

  const apply = (exp: BvExperiment) => {
    setParams(fromScenario({ ...DEFAULT_BV, ...exp.scenario }));
    setActiveId(exp.id);
  };
  const stillActive = (() => {
    const exp = bvExperiments.find((e) => e.id === activeId);
    return (
      !!exp &&
      JSON.stringify(fromScenario({ ...DEFAULT_BV, ...exp.scenario })) === JSON.stringify(params)
    );
  })();

  const metrics: Metric[] = [
    {
      label: 'Test error',
      value: run.testError < 100 ? fix(run.testError, 3) : run.testError.toExponential(1),
      testId: 'test-error',
      hint: 'mean squared error on fresh, noisy data',
    },
    {
      label: 'Training error',
      value: fix(run.trainError, 3),
      hint: 'mean squared error on the data the model was fitted to',
    },
    {
      label: 'Noise floor',
      value: fix(run.noiseVariance, 3),
      hint: 'the smallest test error any model could reach',
    },
    {
      label: 'Gap',
      value: run.testError < 100 ? fix(run.testError - run.trainError, 3) : '≫',
      hint: 'test error minus training error: the signature of overfitting',
      tone: run.testError - run.trainError > 0.1 ? 'bad' : 'neutral',
    },
  ];

  const stages: [string, string][] = [
    ['lab-fits', 'Many fits'],
    ['lab-split', 'Where the error comes from'],
    ['lab-complexity', 'Flexibility'],
    ['lab-learning', 'More data'],
    ['lab-how', 'How it works'],
    ['lab-controls', 'Settings'],
  ];

  const xs = curve.map((c) => c.features);
  const nTrain = scenario.n;
  const peak = curve.reduce((b, c) => (c.test > b.test ? c : b), curve[0]!);
  const best = curve.reduce((b, c) => (c.test < b.test ? c : b), curve[0]!);

  return (
    <div className="mx-auto max-w-6xl px-4 pt-8 pb-20 sm:px-6 lg:px-8">
      <LabHeader
        eyebrow="Evaluation & optimization"
        title="Bias, variance"
        accent="and generalisation"
        blurb="Refit the same model on dozens of fresh datasets from a world whose truth you know, and watch the error split into what the model gets systematically wrong, how much it wobbles, and what nobody could predict."
        badges={
          <>
            <Badge>{WORLDS[scenario.world].label}</Badge>
            <Badge>{scenario.features} terms</Badge>
            <Badge>{scenario.n} points</Badge>
          </>
        }
      />
      <StageNav stages={stages} />
      <ExperimentBar
        lessons={bvExperiments}
        activeId={stillActive ? activeId : null}
        onApply={apply}
      />

      <div className="flex flex-col gap-5">
        <Panel id="lab-fits" className="scroll-mt-24" title="Many fits" eyebrow="01">
          <div className="space-y-6">
            <p className="text-[0.8rem] text-muted">{WORLDS[scenario.world].note}</p>
            <MetricsStrip metrics={metrics} />
            <Fits run={run} />
          </div>
        </Panel>
        <Panel
          id="lab-split"
          className="scroll-mt-24"
          title="Where the error comes from"
          eyebrow="02"
        >
          <Decomposition run={run} />
        </Panel>
        <Panel id="lab-complexity" className="scroll-mt-24" title="Flexibility" eyebrow="03">
          <div className="grid gap-8 lg:grid-cols-[minmax(0,1fr)_20rem]">
            <MiniChart
              title="Error against number of terms (log scale)"
              x={xs}
              xLabel="terms in the model"
              series={[
                {
                  name: 'training error',
                  color: 'var(--brand)',
                  values: curve.map((c) => Math.max(c.train, 1e-4)),
                },
                {
                  name: 'test error',
                  color: 'var(--neg)',
                  values: curve.map((c) => Math.max(c.test, 1e-4)),
                },
                {
                  name: 'noise floor',
                  color: 'var(--muted)',
                  values: curve.map(() => Math.max(scenario.noise ** 2, 1e-4)),
                  dashed: true,
                },
              ]}
              log
              height={230}
            />
            <div className="space-y-2 text-sm text-muted" data-testid="complexity-note">
              <p>
                Training error keeps falling as terms are added; test error falls, reaches a minimum
                at about <strong className="text-fg">{best.features}</strong> terms, then rises.
              </p>
              {peak.features >= nTrain - 2 && peak.features <= nTrain + 2 && (
                <p>
                  The spike at <strong className="text-fg">{peak.features}</strong> terms is the{' '}
                  <em>interpolation threshold</em>: as many terms as points, so the curve passes
                  through every point, noise included. Beyond it the model has many ways to
                  interpolate; picking the smallest-coefficient one, the error comes down again:{' '}
                  <em>double descent</em>.
                </p>
              )}
            </div>
          </div>
        </Panel>
        <Panel id="lab-learning" className="scroll-mt-24" title="More data" eyebrow="04">
          <div className="grid gap-8 lg:grid-cols-[minmax(0,1fr)_20rem]">
            <MiniChart
              title={`Error against training-set size (${scenario.features} terms, log scale)`}
              x={learning.map((l) => l.n)}
              xLabel="training points"
              series={[
                {
                  name: 'training error',
                  color: 'var(--brand)',
                  values: learning.map((l) => Math.max(l.train, 1e-4)),
                },
                {
                  name: 'test error',
                  color: 'var(--neg)',
                  values: learning.map((l) => Math.max(l.test, 1e-4)),
                },
              ]}
              log
              height={210}
            />
            <p className="text-sm text-muted">
              A <em>learning curve</em>. If the two curves meet at a high level, the model is too
              simple (more data will not help; make the model richer). If a wide gap remains, the
              model has too much freedom for this much data (get more data, simplify or regularise).
            </p>
          </div>
        </Panel>
        <Panel id="lab-how" className="scroll-mt-24" title="How it works" eyebrow="05">
          <div className="space-y-3 text-sm text-muted">
            <p>
              The lab draws {scenario.datasets} datasets from the chosen world, fits the same
              polynomial (Legendre basis, optional ridge penalty) to each, and compares the fits
              with the true curve on a grid. The mean fit gives the bias, the scatter around it the
              variance. When there are more terms than points the fit is the minimum-norm
              interpolating solution (the one gradient descent from zero would find).
            </p>
          </div>
        </Panel>
        <Panel id="lab-controls" className="scroll-mt-24" title="Settings" eyebrow="06">
          <ParamPanel specs={bvParams} values={params} onChange={setParam} columns={3} />
        </Panel>
      </div>
    </div>
  );
}
