import { useEffect, useMemo, useState } from 'react';
import { Badge, Panel } from '@/design-system';
import { usePalette } from '@/shared/canvas/colors';
import { ExperimentBar } from '@/shared/components/ExperimentBar';
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
  DEFAULT_EXPLAIN,
  FEATURES,
  MODEL_LABELS,
  runExplain,
  type ExplainRun,
  type ExplainScenario,
} from './explainCore';
import { explainExperiments, explainParams, type ExplainExperiment } from './explainSpec';

const toScenario = (p: ParamValues): ExplainScenario => ({
  ...DEFAULT_EXPLAIN,
  model: p.model as ExplainScenario['model'],
  instance: Number(p.instance),
  feature: Number(p.feature),
  limeSamples: Number(p.limeSamples),
});

const fromScenario = (s: ExplainScenario): ParamValues => ({
  model: s.model,
  instance: s.instance,
  feature: String(s.feature),
  limeSamples: s.limeSamples,
});

function Diverging({
  rows,
  scale,
}: {
  rows: { label: string; a: number; b: number; c: number; value: number }[];
  scale: number;
}) {
  const palette = usePalette();
  return (
    <div className="space-y-1.5">
      <div className="grid grid-cols-[3.5rem_minmax(0,1fr)_4.5rem] items-center gap-3 text-[0.68rem] text-faint">
        <span />
        <span className="flex gap-4">
          <span>
            <span
              className="mr-1 inline-block size-2 rounded-sm"
              style={{ background: palette.css.brand }}
            />
            Shapley (exact)
          </span>
          <span>
            <span
              className="mr-1 inline-block size-2 rounded-sm"
              style={{ background: palette.css.pos }}
            />
            KernelSHAP
          </span>
          <span>
            <span
              className="mr-1 inline-block size-2 rounded-sm"
              style={{ background: palette.css.neg }}
            />
            LIME
          </span>
        </span>
        <span className="text-right">value</span>
      </div>
      {rows.map((r) => (
        <div
          key={r.label}
          className="grid grid-cols-[3.5rem_minmax(0,1fr)_4.5rem] items-center gap-3 text-[0.8rem]"
        >
          <span className="text-muted">{r.label}</span>
          <div className="relative space-y-0.5">
            <div className="absolute inset-y-0 left-1/2 w-px bg-line" />
            {[
              [r.a, palette.css.brand],
              [r.b, palette.css.pos],
              [r.c, palette.css.neg],
            ].map(([v, color], i) => (
              <div key={i} className="relative h-1.5">
                <div
                  className="absolute h-1.5 rounded-full"
                  style={{
                    background: color as string,
                    opacity: 0.85,
                    left: (v as number) >= 0 ? '50%' : `${50 + ((v as number) / scale) * 50}%`,
                    width: `${(Math.min(Math.abs(v as number), scale) / scale) * 50}%`,
                  }}
                />
              </div>
            ))}
          </div>
          <span className="text-right font-mono text-[0.74rem] text-faint">{fix(r.value, 3)}</span>
        </div>
      ))}
    </div>
  );
}

function Bars({ title, values, testId }: { title: string; values: number[]; testId?: string }) {
  const palette = usePalette();
  const order = Array.from(values.keys()).sort(
    (a, b) => (values[b] as number) - (values[a] as number),
  );
  const max = Math.max(...values.map(Math.abs), 1e-9);
  return (
    <div className="space-y-1.5" data-testid={testId}>
      <p className="eyebrow">{title}</p>
      {order.map((j) => (
        <div
          key={j}
          className="grid grid-cols-[2.5rem_minmax(0,1fr)_3.5rem] items-center gap-2 text-[0.78rem]"
        >
          <span className="text-muted">{FEATURES[j]}</span>
          <div className="h-2 rounded-full bg-raised">
            <div
              className="h-2 rounded-full"
              style={{
                width: `${(Math.max(values[j] as number, 0) / max) * 100}%`,
                background: palette.css.brand,
                opacity: 0.85,
              }}
            />
          </div>
          <span className="text-right font-mono text-[0.72rem] text-faint">
            {fix(values[j], 3)}
          </span>
        </div>
      ))}
    </div>
  );
}

function Dependence({ run }: { run: ExplainRun }) {
  const palette = usePalette();
  const { grid, curves, average } = run.ice;
  const w = 560;
  const h = 260;
  const pad = { l: 40, r: 10, t: 10, b: 26 };
  const flat = curves.flat();
  const lo = Math.min(...flat);
  const hi = Math.max(...flat);
  const sx = (v: number) =>
    pad.l +
    ((v - (grid[0] as number)) / ((grid[grid.length - 1] as number) - (grid[0] as number) || 1)) *
      (w - pad.l - pad.r);
  const sy = (v: number) => h - pad.b - ((v - lo) / (hi - lo || 1)) * (h - pad.t - pad.b);
  const path = (ys: number[]) =>
    ys
      .map((y, i) => `${i ? 'L' : 'M'}${sx(grid[i] as number).toFixed(1)},${sy(y).toFixed(1)}`)
      .join(' ');
  const mine = run.instanceValues[run.scenario.feature] as number;
  return (
    <div className="space-y-3">
      <svg
        viewBox={`0 0 ${w} ${h}`}
        className="w-full max-w-2xl rounded-xl border border-line"
        role="img"
        aria-label="Individual conditional expectation curves and their average for the chosen feature"
      >
        {curves.map((c, i) => (
          <path
            key={i}
            d={path(c)}
            fill="none"
            stroke={palette.css.muted}
            strokeWidth={1}
            opacity={0.35}
          />
        ))}
        <path d={path(average)} fill="none" stroke={palette.css.brand} strokeWidth={3} />
        <line
          x1={sx(mine)}
          x2={sx(mine)}
          y1={pad.t}
          y2={h - pad.b}
          stroke={palette.css.neg}
          strokeDasharray="4 3"
        />
        <text x={w / 2} y={h - 6} textAnchor="middle" className="fill-muted text-[10px]">
          {FEATURES[run.scenario.feature]} (standardised)
        </text>
        <text x={8} y={pad.t + 10} className="fill-faint font-mono text-[10px]">
          {hi.toFixed(0)}
        </text>
        <text x={8} y={h - pad.b} className="fill-faint font-mono text-[10px]">
          {lo.toFixed(0)}
        </text>
      </svg>
      <p className="text-sm text-muted" data-testid="dependence-note">
        Each grey line is one patient: what the model would predict for them if only{' '}
        {FEATURES[run.scenario.feature]} were changed. The thick line is their average (the partial
        dependence); the dashed red line marks the patient being explained. Lines that cross or fan
        out show interactions: the effect of {FEATURES[run.scenario.feature]} depends on the other
        measurements.
      </p>
    </div>
  );
}

export default function ExplainPage() {
  const [params, setParams] = useState<ParamValues>(() => fromScenario(DEFAULT_EXPLAIN));
  const [activeId, setActiveId] = useState<string | null>(null);
  const [run, setRun] = useState<ExplainRun | null>(null);
  const scenario = useMemo(() => toScenario(params), [params]);

  useEffect(() => {
    const timer = window.setTimeout(() => setRun(runExplain(scenario)), 30);
    return () => window.clearTimeout(timer);
  }, [scenario]);

  const setParam = (key: string, value: ParamValue) => setParams((p) => ({ ...p, [key]: value }));
  const apply = (exp: ExplainExperiment) => {
    setParams(fromScenario({ ...DEFAULT_EXPLAIN, ...exp.scenario }));
    setActiveId(exp.id);
  };
  const stillActive = (() => {
    const exp = explainExperiments.find((e) => e.id === activeId);
    return (
      !!exp &&
      JSON.stringify(fromScenario({ ...DEFAULT_EXPLAIN, ...exp.scenario })) ===
        JSON.stringify(params)
    );
  })();

  const metrics: Metric[] = run
    ? [
        { label: 'Model prediction', value: fix(run.shap.prediction, 1), testId: 'prediction' },
        {
          label: 'Average prediction',
          value: fix(run.shap.base, 1),
          hint: 'what the model says with no information about the patient',
        },
        { label: 'Actual progression', value: fix(run.actual, 0) },
        { label: 'Model quality (R² on new patients)', value: fix(run.testR2, 2) },
      ]
    : [];

  const stages: [string, string][] = [
    ['lab-local', 'One prediction'],
    ['lab-global', 'Overall'],
    ['lab-dep', 'One feature'],
    ['lab-how', 'How it works'],
    ['lab-controls', 'Settings'],
  ];
  const scale = run
    ? Math.max(...run.shap.values.map(Math.abs), ...run.lime.contributions.map(Math.abs), 1e-9)
    : 1;

  return (
    <div className="mx-auto max-w-6xl px-4 pt-8 pb-20 sm:px-6 lg:px-8">
      <LabHeader
        eyebrow="Interpretability & robustness"
        title="Explaining a prediction"
        accent="who gets the credit?"
        blurb="A model predicts how a diabetes patient will progress. Which of the ten measurements pushed this prediction up or down, and by how much? Shapley values, KernelSHAP, LIME, permutation importance and dependence plots, on four different models."
        badges={
          <>
            <Badge>{MODEL_LABELS[scenario.model]}</Badge>
            <Badge>patient {scenario.instance + 1} of 142</Badge>
          </>
        }
      />
      <StageNav stages={stages} />
      <ExperimentBar
        lessons={explainExperiments}
        activeId={stillActive ? activeId : null}
        onApply={apply}
      />

      <div className="flex flex-col gap-5">
        <Panel id="lab-local" className="scroll-mt-24" title="One prediction" eyebrow="01">
          {run ? (
            <div className="space-y-6">
              <MetricsStrip metrics={metrics} className="lg:grid-cols-4" />
              <Diverging
                scale={scale}
                rows={FEATURES.map((f, j) => ({
                  label: f,
                  a: run.shap.values[j] as number,
                  b: run.kernel.values[j] as number,
                  c: run.lime.contributions[j] as number,
                  value: run.instanceValues[j] as number,
                }))}
              />
              <p className="text-sm text-muted" data-testid="local-note">
                Bars to the right push the predicted progression above the average (
                {fix(run.shap.base, 1)}), bars to the left push it below. The{' '}
                <span style={{ color: 'var(--brand)' }}>exact Shapley values</span> add up to the
                gap between this patient&apos;s prediction ({fix(run.shap.prediction, 1)}) and the
                average. KernelSHAP estimates them from 40 sampled coalitions; LIME fits a local
                straight line ({fix(run.lime.fidelity * 100, 0)}% of the model&apos;s local
                behaviour explained). The right-hand column is the patient&apos;s (standardised)
                measurement.
              </p>
            </div>
          ) : (
            <p className="text-sm text-muted">Computing contributions…</p>
          )}
        </Panel>
        <Panel id="lab-global" className="scroll-mt-24" title="Overall" eyebrow="02">
          {run && (
            <div className="grid gap-8 md:grid-cols-2">
              <Bars
                title="Average size of the Shapley value (10 patients)"
                values={run.meanAbsShap}
                testId="mean-shap"
              />
              <Bars
                title="Permutation importance (drop in R² when shuffled)"
                values={run.permutation}
                testId="permutation"
              />
            </div>
          )}
        </Panel>
        <Panel id="lab-dep" className="scroll-mt-24" title="One feature" eyebrow="03">
          {run && <Dependence run={run} />}
        </Panel>
        <Panel id="lab-how" className="scroll-mt-24" title="How it works" eyebrow="04">
          <div className="space-y-3 text-sm text-muted">
            <p>
              <strong className="text-fg">Shapley values</strong> come from cooperative game theory:
              treat the features as players who together produce the prediction, and pay each its
              average marginal contribution over every order in which the players could join. Here a
              feature that has not joined is replaced by the same feature from a background patient,
              and the model is averaged over 30 background patients. They are the only attributions
              that add up to the prediction, give nothing to a feature the model ignores, treat
              equal features equally and are additive across models (all checked in the tests).
            </p>
            <p>
              <strong className="text-fg">LIME</strong> does not use that definition: it perturbs
              the patient, asks the model, and fits a weighted linear model to the answers. It is
              fast and easy to read but depends on the random perturbations and on how wide a
              neighbourhood it looks at. <strong className="text-fg">Permutation importance</strong>{' '}
              and <strong className="text-fg">partial dependence</strong> describe the model as a
              whole rather than one patient.
            </p>
          </div>
        </Panel>
        <Panel id="lab-controls" className="scroll-mt-24" title="Settings" eyebrow="05">
          <ParamPanel specs={explainParams} values={params} onChange={setParam} columns={2} />
        </Panel>
      </div>
    </div>
  );
}
