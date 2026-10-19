import { costAt } from '@sim/classic';
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
  DEFAULT_METRICS,
  MODEL_LABELS,
  runMetrics,
  type MetricsRun,
  type MetricsScenario,
} from './metricsCore';
import { metricsExperiments, metricsParams, type MetricsExperiment } from './metricsSpec';

const toScenario = (p: ParamValues): MetricsScenario => ({
  ...DEFAULT_METRICS,
  positiveShare: Number(p.positiveShare),
  separation: Number(p.separation),
  model: p.model as MetricsScenario['model'],
  rebalance: p.rebalance as MetricsScenario['rebalance'],
  calibrate: Boolean(p.calibrate),
  threshold: Number(p.threshold),
  costRatio: Number(p.costRatio),
});

const fromScenario = (s: MetricsScenario): ParamValues => ({
  positiveShare: s.positiveShare,
  separation: s.separation,
  model: s.model,
  rebalance: s.rebalance,
  calibrate: s.calibrate,
  threshold: s.threshold,
  costRatio: s.costRatio,
});

/** A square plot of a curve with an optional marker and a diagonal reference. */
function CurvePlot({
  x,
  y,
  xLabel,
  yLabel,
  marker,
  diagonal,
  points,
  title,
}: {
  x: number[];
  y: number[];
  xLabel: string;
  yLabel: string;
  marker?: [number, number];
  diagonal?: boolean;
  points?: { x: number; y: number; r: number }[];
  title: string;
}) {
  const palette = usePalette();
  const size = 320;
  const pad = 34;
  const sx = (v: number) => pad + v * (size - pad - 10);
  const sy = (v: number) => size - pad - v * (size - pad - 10);
  const path = x
    .map((v, i) => `${i ? 'L' : 'M'}${sx(v).toFixed(1)},${sy(y[i] as number).toFixed(1)}`)
    .join(' ');
  return (
    <figure className="space-y-1">
      <figcaption className="eyebrow">{title}</figcaption>
      <svg
        viewBox={`0 0 ${size} ${size}`}
        className="w-full max-w-[22rem]"
        role="img"
        aria-label={title}
      >
        {[0, 0.25, 0.5, 0.75, 1].map((t) => (
          <g key={t}>
            <line
              x1={sx(t)}
              x2={sx(t)}
              y1={sy(0)}
              y2={sy(1)}
              stroke="currentColor"
              opacity={0.07}
            />
            <line
              x1={sx(0)}
              x2={sx(1)}
              y1={sy(t)}
              y2={sy(t)}
              stroke="currentColor"
              opacity={0.07}
            />
            <text
              x={sx(t)}
              y={size - pad + 14}
              textAnchor="middle"
              className="fill-faint font-mono text-[9px]"
            >
              {t}
            </text>
            <text
              x={pad - 6}
              y={sy(t) + 3}
              textAnchor="end"
              className="fill-faint font-mono text-[9px]"
            >
              {t}
            </text>
          </g>
        ))}
        {diagonal && (
          <line
            x1={sx(0)}
            y1={sy(0)}
            x2={sx(1)}
            y2={sy(1)}
            stroke="currentColor"
            strokeDasharray="4 4"
            opacity={0.35}
          />
        )}
        {points?.map((p, i) => (
          <circle
            key={i}
            cx={sx(p.x)}
            cy={sy(p.y)}
            r={p.r}
            fill={palette.css.brand}
            opacity={0.7}
          />
        ))}
        <path d={path} fill="none" stroke={palette.css.brand} strokeWidth={2.4} />
        {marker && (
          <circle
            cx={sx(marker[0])}
            cy={sy(marker[1])}
            r={5.5}
            fill={palette.css.neg}
            stroke={palette.css.surface}
            strokeWidth={1.5}
          />
        )}
        <text x={size / 2} y={size - 4} textAnchor="middle" className="fill-muted text-[10px]">
          {xLabel}
        </text>
        <text
          x={10}
          y={size / 2}
          textAnchor="middle"
          transform={`rotate(-90 10 ${size / 2})`}
          className="fill-muted text-[10px]"
        >
          {yLabel}
        </text>
      </svg>
    </figure>
  );
}

function Confusion({ run }: { run: MetricsRun }) {
  const { tp, fp, tn, fn } = run.at;
  const cells: [string, number, string, string][] = [
    ['Caught (true positive)', tp, 'bg-pos/20', 'positive and flagged'],
    ['Missed (false negative)', fn, 'bg-neg/20', 'positive but not flagged'],
    ['False alarm (false positive)', fp, 'bg-neg/20', 'negative but flagged'],
    ['Correctly ignored (true negative)', tn, 'bg-pos/20', 'negative and not flagged'],
  ];
  return (
    <div className="grid grid-cols-2 gap-2" role="group" aria-label="Confusion matrix">
      {cells.map(([label, n, tone, sub]) => (
        <div key={label} className={`rounded-xl border border-line p-3 ${tone}`}>
          <p
            className="font-mono text-xl tabular-nums"
            data-testid={label.startsWith('Caught') ? 'tp' : undefined}
          >
            {n}
          </p>
          <p className="text-[0.78rem] text-fg">{label}</p>
          <p className="text-[0.68rem] text-fg/80">{sub}</p>
        </div>
      ))}
    </div>
  );
}

function CostCurve({ run }: { run: MetricsRun }) {
  const palette = usePalette();
  const ts = Array.from({ length: 50 }, (_, i) => (i + 0.5) / 50);
  const cost = ts.map((t) => costAt(run.scores, run.labels, t, run.scenario.costRatio, 1));
  const w = 520;
  const h = 200;
  const max = Math.max(...cost);
  const sx = (t: number) => 30 + t * (w - 40);
  const sy = (c: number) => h - 28 - (c / max) * (h - 44);
  const path = ts
    .map((t, i) => `${i ? 'L' : 'M'}${sx(t).toFixed(1)},${sy(cost[i] as number).toFixed(1)}`)
    .join(' ');
  return (
    <div className="space-y-3">
      <svg
        viewBox={`0 0 ${w} ${h}`}
        className="w-full max-w-xl"
        role="img"
        aria-label="Total cost against threshold"
      >
        <line x1={30} x2={w - 10} y1={h - 28} y2={h - 28} stroke="currentColor" opacity={0.3} />
        <path d={path} fill="none" stroke={palette.css.brand} strokeWidth={2.4} />
        <line
          x1={sx(run.scenario.threshold)}
          x2={sx(run.scenario.threshold)}
          y1={8}
          y2={h - 28}
          stroke={palette.css.neg}
          strokeWidth={1.6}
        />
        <line
          x1={sx(Math.min(run.best.threshold, 0.995))}
          x2={sx(Math.min(run.best.threshold, 0.995))}
          y1={8}
          y2={h - 28}
          stroke={palette.css.pos}
          strokeWidth={1.6}
          strokeDasharray="5 4"
        />
        {[0, 0.25, 0.5, 0.75, 1].map((t) => (
          <text
            key={t}
            x={sx(t)}
            y={h - 12}
            textAnchor="middle"
            className="fill-faint font-mono text-[10px]"
          >
            {t}
          </text>
        ))}
        <text x={w / 2} y={h - 1} textAnchor="middle" className="fill-muted text-[10px]">
          threshold
        </text>
      </svg>
      <p className="text-sm text-muted" data-testid="cost-note">
        Total cost = {run.scenario.costRatio} × misses + 1 × false alarms.{' '}
        <span className="text-neg">Red</span>: your threshold (cost {run.cost}).{' '}
        <span className="text-pos">Green dashed</span>: the cheapest threshold (
        {fix(run.best.threshold, 2)}, cost {run.best.cost}). The right threshold is not a property
        of the model but of what each kind of mistake costs.
      </p>
    </div>
  );
}

export default function MetricsPage() {
  const [params, setParams] = useState<ParamValues>(() => fromScenario(DEFAULT_METRICS));
  const [activeId, setActiveId] = useState<string | null>(null);
  const scenario = useMemo(() => toScenario(params), [params]);
  // everything except the threshold and the costs comes from the same trained model
  const modelKey = `${scenario.positiveShare}|${scenario.separation}|${scenario.model}|${scenario.rebalance}|${scenario.calibrate}`;
  const run = useMemo(
    () => runMetrics(scenario),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [modelKey, scenario.threshold, scenario.costRatio],
  );
  const setParam = (key: string, value: ParamValue) => setParams((p) => ({ ...p, [key]: value }));

  const apply = (exp: MetricsExperiment) => {
    setParams(fromScenario({ ...DEFAULT_METRICS, ...exp.scenario }));
    setActiveId(exp.id);
  };
  const stillActive = (() => {
    const exp = metricsExperiments.find((e) => e.id === activeId);
    return (
      !!exp &&
      JSON.stringify(fromScenario({ ...DEFAULT_METRICS, ...exp.scenario })) ===
        JSON.stringify(params)
    );
  })();

  const metrics: Metric[] = [
    {
      label: 'Accuracy',
      value: pct(run.at.accuracy),
      testId: 'accuracy',
      hint: `always saying “no” would score ${pct(run.baselineAccuracy)}`,
    },
    {
      label: 'Precision',
      value: pct(run.at.precision),
      hint: 'of the cases flagged, the share that are positive',
    },
    {
      label: 'Recall',
      value: pct(run.at.recall),
      testId: 'recall',
      hint: 'of the positives, the share that were flagged',
    },
    { label: 'F1', value: fix(run.at.f1, 2) },
    { label: 'MCC', value: fix(run.at.mcc, 2), hint: 'balanced score: 0 = chance, 1 = perfect' },
    { label: 'ROC AUC', value: fix(run.rocAuc, 3) },
    { label: 'Average precision', value: fix(run.averagePrecision, 3) },
    {
      label: 'Calibration error',
      value: fix(run.ece, 3),
      hint: 'expected calibration error: how far predicted probabilities are from observed frequencies',
    },
  ];

  const rocMarker: [number, number] = [1 - run.at.specificity, run.at.recall];
  const prMarker: [number, number] = [run.at.recall, run.at.precision];
  const reliability = run.calibration;
  const stages: [string, string][] = [
    ['lab-confusion', 'Confusion'],
    ['lab-curves', 'ROC and precision-recall'],
    ['lab-calibration', 'Calibration'],
    ['lab-cost', 'Cost'],
    ['lab-how', 'How it works'],
    ['lab-controls', 'Settings'],
  ];

  return (
    <div className="mx-auto max-w-6xl px-4 pt-8 pb-20 sm:px-6 lg:px-8">
      <LabHeader
        eyebrow="Evaluation & optimization"
        title="Metrics & thresholds"
        accent="beyond accuracy"
        blurb="A detector for a rare event, scored every way there is. Move the threshold, rebalance the data, recalibrate the probabilities and put a price on each kind of mistake."
        badges={
          <>
            <Badge>{MODEL_LABELS[scenario.model]}</Badge>
            <Badge>{run.positives} positives in the test set</Badge>
          </>
        }
      />
      <StageNav stages={stages} />
      <ExperimentBar
        lessons={metricsExperiments}
        activeId={stillActive ? activeId : null}
        onApply={apply}
      />

      <div className="flex flex-col gap-5">
        <Panel id="lab-confusion" className="scroll-mt-24" title="Confusion" eyebrow="01">
          <div className="space-y-6">
            <MetricsStrip metrics={metrics} className="lg:grid-cols-4" />
            <div className="grid gap-8 lg:grid-cols-[minmax(0,26rem)_minmax(0,1fr)]">
              <Confusion run={run} />
              <p className="text-sm text-muted">
                At threshold {scenario.threshold.toFixed(2)} the model flags {run.at.tp + run.at.fp}{' '}
                of {run.scores.length} cases. Always saying &ldquo;no&rdquo; would be right{' '}
                {pct(run.baselineAccuracy)} of the time, so accuracy alone says little when
                positives are rare: look at <strong className="text-fg">recall</strong> (how many
                real positives were caught) and <strong className="text-fg">precision</strong> (how
                many alarms were real).
              </p>
            </div>
          </div>
        </Panel>
        <Panel
          id="lab-curves"
          className="scroll-mt-24"
          title="ROC and precision-recall"
          eyebrow="02"
        >
          <div className="space-y-4">
            <div className="flex flex-wrap gap-8">
              <CurvePlot
                title="ROC curve"
                x={run.roc.fpr}
                y={run.roc.tpr}
                xLabel="false alarm rate"
                yLabel="recall"
                marker={rocMarker}
                diagonal
              />
              <CurvePlot
                title="Precision-recall curve"
                x={run.pr.recall}
                y={run.pr.precision}
                xLabel="recall"
                yLabel="precision"
                marker={prMarker}
              />
            </div>
            <p className="text-sm text-muted">
              Each point on a curve is one threshold; the red dot is yours. The ROC curve is blind
              to class balance: with rare positives a tiny false-alarm rate is still a lot of false
              alarms, which only the precision-recall curve shows.
            </p>
          </div>
        </Panel>
        <Panel id="lab-calibration" className="scroll-mt-24" title="Calibration" eyebrow="03">
          <div className="grid gap-8 lg:grid-cols-[auto_minmax(0,1fr)]">
            <CurvePlot
              title="Reliability diagram"
              x={reliability.map((b) => b.predicted)}
              y={reliability.map((b) => b.observed)}
              xLabel="predicted probability"
              yLabel="how often it really happens"
              diagonal
              points={reliability.map((b) => ({
                x: b.predicted,
                y: b.observed,
                r: 3 + Math.min(9, Math.sqrt(b.count) / 3),
              }))}
            />
            <div className="space-y-2 text-sm text-muted" data-testid="calibration-note">
              <p>
                Among cases given a probability near 0.3, about 30% should be positive. Points on
                the diagonal mean the probabilities can be taken at face value. Above the diagonal
                the model is under-confident, below it over-confident (dot size = number of cases).
              </p>
              <p>
                Calibration error <strong className="text-fg">{fix(run.ece, 3)}</strong>, Brier
                score <strong className="text-fg">{fix(run.brier, 3)}</strong>. A model can rank
                well (high AUC) and still give misleading probabilities; that matters as soon as a
                decision uses the probability itself.
              </p>
            </div>
          </div>
        </Panel>
        <Panel id="lab-cost" className="scroll-mt-24" title="Cost" eyebrow="04">
          <CostCurve run={run} />
        </Panel>
        <Panel id="lab-how" className="scroll-mt-24" title="How it works" eyebrow="05">
          <div className="space-y-3 text-sm text-muted">
            <p>
              A classifier outputs a probability; a <strong className="text-fg">threshold</strong>{' '}
              turns it into a decision. The confusion matrix counts the four outcomes; precision,
              recall, F1 and the Matthews correlation (MCC) summarise them in different ways. ROC
              and precision-recall curves sweep the threshold to show every possible compromise at
              once.
            </p>
            <p>
              Rare events are handled by choosing the threshold for the real costs, by{' '}
              <strong className="text-fg">rebalancing</strong> the training data (duplicating
              positives, or SMOTE, which creates new positives between existing ones), or by
              weighting classes; rebalancing shifts the model&apos;s probabilities, which{' '}
              <strong className="text-fg">Platt scaling</strong> (a one-dimensional logistic
              regression on held-out scores) corrects. Metrics and calibration helpers are tested
              against scikit-learn.
            </p>
          </div>
        </Panel>
        <Panel id="lab-controls" className="scroll-mt-24" title="Settings" eyebrow="06">
          <ParamPanel specs={metricsParams} values={params} onChange={setParam} columns={3} />
        </Panel>
      </div>
    </div>
  );
}
