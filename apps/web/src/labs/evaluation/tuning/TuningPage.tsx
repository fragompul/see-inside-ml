import { useEffect, useMemo, useState } from 'react';
import { Badge, Panel, Slider } from '@/design-system';
import { usePalette } from '@/shared/canvas/colors';
import { ExperimentBar } from '@/shared/components/ExperimentBar';
import { MiniChart } from '@/shared/components/MiniChart';
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
  DEFAULT_TUNING,
  METHOD_LABELS,
  runTuning,
  type TuningRun,
  type TuningScenario,
} from './tuningCore';
import { tuningExperiments, tuningParams, type TuningExperiment } from './tuningSpec';

const toScenario = (p: ParamValues): TuningScenario => ({
  ...DEFAULT_TUNING,
  landscape: p.landscape as TuningScenario['landscape'],
  method: p.method as TuningScenario['method'],
  budget: Number(p.budget),
  noise: Number(p.noise),
});

const fromScenario = (s: TuningScenario): ParamValues => ({
  landscape: s.landscape,
  method: s.method,
  budget: s.budget,
  noise: s.noise,
});

const G = 36;

function Surface({ run }: { run: TuningRun }) {
  const palette = usePalette();
  const size = 460;
  const cell = size / G;
  const [shown, setShown] = useState(run.single.evaluations.length);
  const k = Math.min(shown, run.single.evaluations.length);
  const values = useMemo(() => {
    const out: number[] = [];
    for (let r = 0; r < G; r++)
      for (let c = 0; c < G; c++) out.push(run.landscape.value([(c + 0.5) / G, 1 - (r + 0.5) / G]));
    return out;
  }, [run.landscape]);
  const lo = Math.min(...values);
  const hi = Math.max(...values);
  const sx = (v: number) => v * size;
  const sy = (v: number) => (1 - v) * size;
  const rec = run.single.recommended;
  return (
    <div className="grid gap-8 lg:grid-cols-[auto_minmax(0,1fr)]">
      <div className="space-y-3">
        <svg
          viewBox={`0 0 ${size} ${size}`}
          className="w-full max-w-[28.75rem] rounded-xl border border-line"
          role="img"
          aria-label="The score surface of two hyperparameters, with the configurations tried in order"
        >
          {values.map((v, i) => (
            <rect
              key={i}
              x={(i % G) * cell}
              y={Math.floor(i / G) * cell}
              width={cell + 0.5}
              height={cell + 0.5}
              fill={palette.css.brand}
              opacity={0.04 + 0.9 * ((v - lo) / (hi - lo || 1))}
            />
          ))}
          {run.single.evaluations.slice(0, k).map((e, i) => (
            <g key={i}>
              <circle
                cx={sx(e.x[0])}
                cy={sy(e.x[1])}
                r={3 + 4 * e.fidelity}
                fill={palette.css.fg}
                opacity={0.2 + 0.6 * e.fidelity}
              />
              {run.single.evaluations.length <= 30 && (
                <text
                  x={sx(e.x[0])}
                  y={sy(e.x[1]) - 7}
                  textAnchor="middle"
                  className="fill-fg font-mono text-[9px]"
                >
                  {i + 1}
                </text>
              )}
            </g>
          ))}
          <circle
            cx={sx(run.landscape.argBest[0])}
            cy={sy(run.landscape.argBest[1])}
            r={7}
            fill="none"
            stroke={palette.css.pos}
            strokeWidth={2.4}
          />
          {k === run.single.evaluations.length && (
            <rect
              x={sx(rec[0]) - 6}
              y={sy(rec[1]) - 6}
              width={12}
              height={12}
              fill="none"
              stroke={palette.css.neg}
              strokeWidth={2.4}
              transform={`rotate(45 ${sx(rec[0])} ${sy(rec[1])})`}
            />
          )}
        </svg>
        <Slider
          label="Experiments shown"
          min={1}
          max={run.single.evaluations.length}
          value={k}
          onValueChange={setShown}
        />
      </div>
      <div className="space-y-3 text-sm text-muted" data-testid="surface-note">
        <p>
          The shading is the score of every pair of hyperparameters (darker is better) - something
          you would never see in practice, where each pixel costs a full training run. The{' '}
          <span className="text-pos">green ring</span> is the true optimum. Dots are the experiments{' '}
          {METHOD_LABELS[run.scenario.method].toLowerCase()} chose, numbered in order; small, faint
          dots are cheap partial runs. The <span className="text-neg">red diamond</span> is what it
          finally recommends.
        </p>
        <p>
          Recommended score <strong className="text-fg">{fix(run.single.finalValue, 3)}</strong>{' '}
          against a best possible <strong className="text-fg">{fix(run.landscape.best, 3)}</strong>:
          regret <strong className="text-fg">{fix(run.single.regret, 3)}</strong>.
          {run.scenario.method === 'halving' &&
            ` Cost ${fix(run.single.cost, 1)} full experiments over ${run.single.evaluations.length} measurements.`}
        </p>
      </div>
    </div>
  );
}

function Race({ run }: { run: TuningRun }) {
  const palette = usePalette();
  const max = Math.max(...run.race.map((r) => r.meanRegret), 0.05);
  return (
    <div className="space-y-4">
      <div className="space-y-3">
        {run.race.map((r) => (
          <div
            key={r.method}
            className="grid grid-cols-[9.5rem_minmax(0,1fr)_5.5rem] items-center gap-3 text-[0.82rem]"
          >
            <span className={r.method === run.scenario.method ? 'text-fg' : 'text-muted'}>
              {METHOD_LABELS[r.method]}
            </span>
            <div className="h-3 rounded-full bg-raised">
              <div
                className="h-3 rounded-full"
                style={{
                  width: `${(r.meanRegret / max) * 100}%`,
                  background: palette.css.brand,
                  opacity: 0.85,
                }}
              />
            </div>
            <span className="text-right font-mono text-faint" data-testid={`regret-${r.method}`}>
              {fix(r.meanRegret, 3)} · {pct(r.success)}
            </span>
          </div>
        ))}
      </div>
      <p className="text-sm text-muted">
        Average regret (shorter is better) and the share of runs that got within 0.05 of the best
        score, over 16 independent runs of each method with the same budget. A single run can be
        lucky or unlucky; only repetitions say which method is better.
      </p>
    </div>
  );
}

export default function TuningPage() {
  const [params, setParams] = useState<ParamValues>(() => fromScenario(DEFAULT_TUNING));
  const [activeId, setActiveId] = useState<string | null>(null);
  const [run, setRun] = useState<TuningRun | null>(null);
  const scenario = useMemo(() => toScenario(params), [params]);

  // the race runs 64 searches: paint first, then compute
  useEffect(() => {
    const timer = window.setTimeout(() => setRun(runTuning(scenario)), 30);
    return () => window.clearTimeout(timer);
  }, [scenario]);

  const setParam = (key: string, value: ParamValue) => setParams((p) => ({ ...p, [key]: value }));
  const apply = (exp: TuningExperiment) => {
    setParams(fromScenario({ ...DEFAULT_TUNING, ...exp.scenario }));
    setActiveId(exp.id);
  };
  const stillActive = (() => {
    const exp = tuningExperiments.find((e) => e.id === activeId);
    return (
      !!exp &&
      JSON.stringify(fromScenario({ ...DEFAULT_TUNING, ...exp.scenario })) ===
        JSON.stringify(params)
    );
  })();

  const metrics: Metric[] = run
    ? [
        {
          label: 'Regret of this run',
          value: fix(run.single.regret, 3),
          testId: 'regret',
          hint: 'best achievable score minus the score of the recommended configuration',
        },
        { label: 'Experiments', value: String(run.single.evaluations.length) },
        { label: 'Cost (full experiments)', value: fix(run.single.cost, 1) },
      ]
    : [];

  const curve = run?.single.incumbent.map((v) => Math.max(run.landscape.best - v, 1e-4));
  const stages: [string, string][] = [
    ['lab-surface', 'The search'],
    ['lab-race', 'Which method wins?'],
    ['lab-curve', 'Progress'],
    ['lab-how', 'How it works'],
    ['lab-controls', 'Settings'],
  ];

  return (
    <div className="mx-auto max-w-6xl px-4 pt-8 pb-20 sm:px-6 lg:px-8">
      <LabHeader
        eyebrow="Evaluation & optimization"
        title="Hyperparameter tuning"
        accent="searching with a budget"
        blurb="Every trial is an expensive, noisy training run. Grid search, random search, successive halving and Bayesian optimisation spend the same budget in very different ways, and which is best depends on the shape of the surface."
        badges={
          <>
            <Badge>{run?.landscape.label ?? 'surface'}</Badge>
            <Badge>{scenario.budget} experiments</Badge>
          </>
        }
      />
      <StageNav stages={stages} />
      <ExperimentBar
        lessons={tuningExperiments}
        activeId={stillActive ? activeId : null}
        onApply={apply}
      />

      <div className="flex flex-col gap-5">
        <Panel id="lab-surface" className="scroll-mt-24" title="The search" eyebrow="01">
          {run ? (
            <div className="space-y-6">
              <p className="text-[0.8rem] text-muted">{run.landscape.note}</p>
              <MetricsStrip metrics={metrics} className="lg:grid-cols-3" />
              <Surface
                key={`${scenario.landscape}-${scenario.method}-${scenario.budget}-${scenario.noise}`}
                run={run}
              />
            </div>
          ) : (
            <p className="text-sm text-muted">Searching…</p>
          )}
        </Panel>
        <Panel id="lab-race" className="scroll-mt-24" title="Which method wins?" eyebrow="02">
          {run ? <Race run={run} /> : <p className="text-sm text-muted">Racing the methods…</p>}
        </Panel>
        <Panel id="lab-curve" className="scroll-mt-24" title="Progress" eyebrow="03">
          {run && curve && (
            <div className="grid gap-8 lg:grid-cols-[minmax(0,1fr)_20rem]">
              <MiniChart
                title="Regret of the best-looking configuration so far (log scale)"
                x={curve.map((_, i) => i + 1)}
                xLabel="measurement"
                series={[
                  { name: METHOD_LABELS[scenario.method], color: 'var(--brand)', values: curve },
                ]}
                log
                height={200}
              />
              <p className="text-sm text-muted">
                Only full-quality measurements compete for &ldquo;best so far&rdquo;; the curve
                shows the <em>true</em> score of that incumbent, which no experimenter can see. It
                steps down when a better configuration is found, and can even step up when noise
                makes a worse one look better.
              </p>
            </div>
          )}
        </Panel>
        <Panel id="lab-how" className="scroll-mt-24" title="How it works" eyebrow="04">
          <div className="space-y-3 text-sm text-muted">
            <p>
              <strong className="text-fg">Grid search</strong> tries an even lattice; with many
              hyperparameters most of its trials repeat the same few values of the ones that matter.{' '}
              <strong className="text-fg">Random search</strong> tries a new value of every
              hyperparameter each time. <strong className="text-fg">Successive halving</strong>{' '}
              starts many configurations on a cheap, noisy run, keeps the best third, runs those
              longer, and so on.
            </p>
            <p>
              <strong className="text-fg">Bayesian optimisation</strong> fits a Gaussian process to
              the scores seen so far and measures next where the <em>expected improvement</em> over
              the best score is largest: a balance between exploiting promising regions and
              exploring uncertain ones. It shines when each trial is costly and the surface is
              smooth; with a tiny budget it has little to learn from. Tests check the
              expected-improvement formula against numerical integration.
            </p>
          </div>
        </Panel>
        <Panel id="lab-controls" className="scroll-mt-24" title="Settings" eyebrow="05">
          <ParamPanel specs={tuningParams} values={params} onChange={setParam} columns={3} />
        </Panel>
      </div>
    </div>
  );
}
