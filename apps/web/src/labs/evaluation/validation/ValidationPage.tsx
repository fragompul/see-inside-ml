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
  pct,
  type Metric,
  type ParamValue,
  type ParamValues,
} from '@/shared/lab-kit';
import {
  DEFAULT_VALIDATION,
  mean,
  runValidation,
  type ValidationRun,
  type ValidationScenario,
} from './validationCore';
import {
  validationExperiments,
  validationParams,
  type ValidationExperiment,
} from './validationSpec';

const toScenario = (p: ParamValues): ValidationScenario => ({
  ...DEFAULT_VALIDATION,
  kind: p.kind as ValidationScenario['kind'],
  n: Number(p.n),
  neighbors: Number(p.neighbors),
  candidates: Number(p.candidates),
  signal: Number(p.signal),
  samples: Number(p.samples),
  features: Number(p.features),
  keep: Number(p.keep),
});

const fromScenario = (s: ValidationScenario): ParamValues => ({
  kind: s.kind,
  n: s.n,
  neighbors: s.neighbors,
  candidates: s.candidates,
  signal: s.signal,
  samples: s.samples,
  features: s.features,
  keep: s.keep,
});

interface StripRow {
  label: string;
  values: number[];
  color: string;
}

/** One dot per repetition, stacked in rows, with the real value marked by a vertical line. */
function Strips({
  rows,
  reference,
  referenceLabel,
  lo,
  hi,
}: {
  rows: StripRow[];
  reference: number;
  referenceLabel: string;
  lo: number;
  hi: number;
}) {
  const palette = usePalette();
  const w = 760;
  const rowH = 54;
  const left = 150;
  const h = rows.length * rowH + 34;
  const sx = (v: number) => left + ((v - lo) / (hi - lo)) * (w - left - 16);
  const ticks = Array.from({ length: 6 }, (_, i) => lo + ((hi - lo) * i) / 5);
  return (
    <svg
      viewBox={`0 0 ${w} ${h}`}
      className="w-full"
      role="img"
      aria-label="Every repetition's result as a dot, one row per method, with the real value marked"
    >
      {ticks.map((t) => (
        <g key={t}>
          <line x1={sx(t)} x2={sx(t)} y1={4} y2={h - 24} stroke="currentColor" opacity={0.07} />
          <text
            x={sx(t)}
            y={h - 8}
            textAnchor="middle"
            className="fill-faint font-mono text-[10px]"
          >
            {Math.round(t * 100)}%
          </text>
        </g>
      ))}
      {rows.map((row, ri) => {
        const cy = 10 + ri * rowH + rowH / 2;
        const m = mean(row.values);
        return (
          <g key={row.label}>
            <text x={left - 10} y={cy + 4} textAnchor="end" className="fill-muted text-[11px]">
              {row.label}
            </text>
            {row.values.map((v, i) => (
              <circle
                key={i}
                cx={sx(v)}
                cy={cy + ((i * 37) % 19) - 9}
                r={2.6}
                fill={row.color}
                opacity={0.5}
              />
            ))}
            <line
              x1={sx(m)}
              x2={sx(m)}
              y1={cy - 17}
              y2={cy + 17}
              stroke={row.color}
              strokeWidth={3}
            />
          </g>
        );
      })}
      <line
        x1={sx(reference)}
        x2={sx(reference)}
        y1={2}
        y2={h - 24}
        stroke={palette.css.fg}
        strokeWidth={1.6}
        strokeDasharray="5 4"
      />
      <text x={sx(reference) + 5} y={14} className="fill-fg font-mono text-[10px]">
        {referenceLabel}
      </text>
    </svg>
  );
}

function View({ run }: { run: ValidationRun }) {
  const palette = usePalette();
  if (run.kind === 'estimators') {
    const rows: StripRow[] = run.stats.map((s, i) => ({
      label: s.name,
      values: s.estimates,
      color: palette.css.classes[i % palette.css.classes.length] as string,
    }));
    const all = rows.flatMap((r) => r.values);
    const lo = Math.max(0.5, Math.floor((Math.min(...all) - 0.02) * 20) / 20);
    return (
      <div className="space-y-4">
        <Strips
          rows={rows}
          reference={run.truth}
          referenceLabel={`real accuracy ${pct(run.truth)}`}
          lo={lo}
          hi={1}
        />
        <div className="overflow-x-auto">
          <table className="w-full min-w-[26rem] text-left font-mono text-[0.78rem] tabular-nums">
            <thead className="text-muted">
              <tr>
                <th className="pr-4 pb-2 font-sans font-medium">method</th>
                <th className="pr-4 pb-2 font-medium">average</th>
                <th className="pr-4 pb-2 font-medium">scatter (sd)</th>
                <th className="pr-4 pb-2 font-medium">bias</th>
              </tr>
            </thead>
            <tbody>
              {run.stats.map((s) => (
                <tr key={s.name} className="border-t border-line">
                  <td className="py-1 pr-4 font-sans">{s.name}</td>
                  <td className="pr-4">{pct(s.mean)}</td>
                  <td className="pr-4" data-testid={`sd-${s.name.replace(/\W+/g, '-')}`}>
                    {fix(s.std * 100, 1)} pts
                  </td>
                  <td className="pr-4">{fix(s.bias * 100, 1)} pts</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="text-sm text-muted">
          Each dot is the estimate one repetition would have given, for a fresh sample of{' '}
          {run.scenario.n} points; the thick bar is the average over repetitions. The dashed line is
          the real accuracy of the model, measured on 1,500 new points. A good validation method
          puts its dots tightly around the line (low scatter) and its bar on the line (low bias).
        </p>
      </div>
    );
  }
  if (run.kind === 'selection') {
    return (
      <div className="space-y-4">
        <Strips
          rows={[
            {
              label: 'Winner, on the rows it was picked on',
              values: run.winnerScores,
              color: palette.css.neg,
            },
            { label: 'Same winner, on fresh data', values: run.testScores, color: palette.css.pos },
          ]}
          reference={0.5}
          referenceLabel="coin flip 50%"
          lo={0.4}
          hi={1}
        />
        <p className="text-sm text-muted">
          {run.scenario.candidates} candidate rule{run.scenario.candidates === 1 ? '' : 's'}, each
          judged on the same 60 rows; the best is kept. On average the winner scores{' '}
          <strong className="text-fg">{pct(run.winnerMean)}</strong> where it was chosen and{' '}
          <strong className="text-fg">{pct(run.testMean)}</strong> on fresh data: an optimism of{' '}
          <strong className="text-fg">{fix(run.optimism * 100, 1)} points</strong>. The cure is a
          final test set that played no part in the choice, or nested cross-validation.
        </p>
      </div>
    );
  }
  return (
    <div className="space-y-4">
      <Strips
        rows={[
          {
            label: 'Selection done on all rows (leaky)',
            values: run.leaky,
            color: palette.css.neg,
          },
          {
            label: 'Selection inside each fold (honest)',
            values: run.honest,
            color: palette.css.pos,
          },
        ]}
        reference={run.independent}
        referenceLabel={`independent test ${pct(run.independent)}`}
        lo={0.3}
        hi={1}
      />
      <p className="text-sm text-muted">
        Cross-validation is only honest if <em>everything</em> learned from the data happens inside
        the training folds. Picking the best {run.scenario.keep} of {run.scenario.features} features
        by looking at all {run.scenario.samples} rows lets the validation rows influence the model:
        with leakage the score is <strong className="text-fg">{pct(run.leakyMean)}</strong>, with
        honest folds <strong className="text-fg">{pct(run.honestMean)}</strong>, and a truly
        independent test set says <strong className="text-fg">{pct(run.independent)}</strong>.
      </p>
    </div>
  );
}

export default function ValidationPage() {
  const [params, setParams] = useState<ParamValues>(() => fromScenario(DEFAULT_VALIDATION));
  const [activeId, setActiveId] = useState<string | null>(null);
  const [run, setRun] = useState<ValidationRun | null>(null);
  const scenario = useMemo(() => toScenario(params), [params]);

  // the repetitions take a moment: paint first, then compute
  useEffect(() => {
    const timer = window.setTimeout(() => setRun(runValidation(scenario)), 30);
    return () => window.clearTimeout(timer);
  }, [scenario]);

  const setParam = (key: string, value: ParamValue) => setParams((p) => ({ ...p, [key]: value }));
  const apply = (exp: ValidationExperiment) => {
    setParams(fromScenario({ ...DEFAULT_VALIDATION, ...exp.scenario }));
    setActiveId(exp.id);
  };
  const stillActive = (() => {
    const exp = validationExperiments.find((e) => e.id === activeId);
    return (
      !!exp &&
      JSON.stringify(fromScenario({ ...DEFAULT_VALIDATION, ...exp.scenario })) ===
        JSON.stringify(params)
    );
  })();

  const fresh = run && run.scenario === scenario;
  let metrics: Metric[] = [];
  if (run?.kind === 'estimators') {
    const best = [...run.stats].sort((a, b) => a.std - b.std)[0]!;
    metrics = [
      { label: 'Real accuracy', value: pct(run.truth), testId: 'truth' },
      { label: 'Least scatter', value: `${best.name} · ${fix(best.std * 100, 1)} pts` },
      { label: 'Scatter of one 20% hold-out', value: `${fix(run.stats[0]!.std * 100, 1)} pts` },
    ];
  } else if (run?.kind === 'selection') {
    metrics = [
      { label: 'Winner where picked', value: pct(run.winnerMean), testId: 'truth' },
      { label: 'Winner on fresh data', value: pct(run.testMean) },
      {
        label: 'Optimism',
        value: `${fix(run.optimism * 100, 1)} pts`,
        tone: run.optimism > 0.05 ? 'bad' : 'neutral',
      },
    ];
  } else if (run?.kind === 'leakage') {
    metrics = [
      { label: 'Leaky score', value: pct(run.leakyMean), testId: 'truth', tone: 'bad' },
      { label: 'Honest score', value: pct(run.honestMean), tone: 'good' },
      { label: 'Independent test', value: pct(run.independent) },
    ];
  }

  const stages: [string, string][] = [
    ['lab-view', 'The experiment'],
    ['lab-how', 'How it works'],
    ['lab-controls', 'Settings'],
  ];

  return (
    <div className="mx-auto max-w-6xl px-4 pt-8 pb-20 sm:px-6 lg:px-8">
      <LabHeader
        eyebrow="Evaluation & optimization"
        title="Validation"
        accent="how not to fool yourself"
        blurb="Hold-out or cross-validation? What happens when you pick the best of many models on the same data, or choose features before splitting? Each experiment is repeated many times so you can see the whole spread of what you might have measured."
        badges={
          <>
            <Badge>
              {scenario.kind === 'estimators'
                ? 'validation methods'
                : scenario.kind === 'selection'
                  ? 'model selection'
                  : 'data leakage'}
            </Badge>
          </>
        }
      />
      <StageNav stages={stages} />
      <ExperimentBar
        lessons={validationExperiments}
        activeId={stillActive ? activeId : null}
        onApply={apply}
      />

      <div className="flex flex-col gap-5">
        <Panel id="lab-view" className="scroll-mt-24" title="The experiment" eyebrow="01">
          {run ? (
            <div className="space-y-6" aria-busy={!fresh}>
              <MetricsStrip metrics={metrics} className="lg:grid-cols-3" />
              <View run={run} />
            </div>
          ) : (
            <p className="text-sm text-muted">Running the repetitions…</p>
          )}
        </Panel>
        <Panel id="lab-how" className="scroll-mt-24" title="How it works" eyebrow="02">
          <div className="space-y-3 text-sm text-muted">
            <p>
              <strong className="text-fg">Hold-out</strong> trains on part of the rows and scores on
              the rest: simple, but the score depends on which rows landed where, and a small test
              part is noisy. <strong className="text-fg">k-fold cross-validation</strong> rotates
              the held-out part through the data and averages, using every row for testing exactly
              once and training on a larger share.
            </p>
            <p>
              Any step that uses the labels (choosing the best model, tuning a hyperparameter,
              selecting features, scaling) must be fitted on training rows only; otherwise the
              validation rows have leaked into the model and the score is optimistic. A final test
              set, touched once, or nested cross-validation (an outer loop for the estimate, an
              inner one for the choice), keeps the estimate honest.
            </p>
          </div>
        </Panel>
        <Panel id="lab-controls" className="scroll-mt-24" title="Settings" eyebrow="03">
          <ParamPanel specs={validationParams} values={params} onChange={setParam} columns={3} />
        </Panel>
      </div>
    </div>
  );
}
