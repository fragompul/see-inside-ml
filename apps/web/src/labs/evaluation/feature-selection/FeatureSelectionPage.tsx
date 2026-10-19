import { useMemo, useState } from 'react';
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
  DEFAULT_FEATURES,
  GROUP,
  METHOD_LABELS,
  METHOD_NOTES,
  NAMES,
  runFeatures,
  type FeaturesRun,
  type FeaturesScenario,
} from './featuresCore';
import { featuresExperiments, featuresParams, type FeaturesExperiment } from './featuresSpec';

const toScenario = (p: ParamValues): FeaturesScenario => ({
  ...DEFAULT_FEATURES,
  method: p.method as FeaturesScenario['method'],
  keep: Number(p.keep),
  n: Number(p.n),
  copyNoise: Number(p.copyNoise),
});

const fromScenario = (s: FeaturesScenario): ParamValues => ({
  method: s.method,
  keep: s.keep,
  n: s.n,
  copyNoise: s.copyNoise,
});

function Ranking({ run }: { run: FeaturesRun }) {
  const palette = usePalette();
  const max = Math.max(...run.scores.map((v) => Math.abs(v)), 1e-9);
  const chosen = new Set(run.chosen);
  return (
    <div className="space-y-4">
      <ol className="space-y-1" data-testid="ranking">
        {run.ranking.map((j, rank) => {
          const g = GROUP[j] as number;
          const color = g < 0 ? palette.css.muted : j < 4 ? palette.css.pos : palette.css.brand;
          return (
            <li
              key={j}
              className={`grid grid-cols-[1.4rem_6.5rem_minmax(0,1fr)] items-center gap-2 text-[0.78rem] `}
            >
              <span className="font-mono text-faint">{rank + 1}</span>
              <span className={chosen.has(j) ? 'font-medium text-fg' : 'text-muted'}>
                {NAMES[j]}
              </span>
              <div className="h-2.5 rounded-full bg-raised">
                <div
                  className={`h-2.5 rounded-full ${chosen.has(j) ? '' : 'opacity-55'}`}
                  style={{
                    width: `${(Math.abs(run.scores[j] as number) / max) * 100}%`,
                    background: color,
                  }}
                />
              </div>
            </li>
          );
        })}
      </ol>
      <p className="flex flex-wrap gap-x-5 gap-y-1 text-[0.78rem] text-muted">
        <span>
          <span
            className="mr-1 inline-block size-2 rounded-sm"
            style={{ background: palette.css.pos }}
          />
          true signal
        </span>
        <span>
          <span
            className="mr-1 inline-block size-2 rounded-sm"
            style={{ background: palette.css.brand }}
          />
          noisy copy of a signal
        </span>
        <span>
          <span
            className="mr-1 inline-block size-2 rounded-sm"
            style={{ background: palette.css.muted }}
          />
          pure noise
        </span>
        <span>Bright rows are the {run.scenario.keep} kept.</span>
      </p>
    </div>
  );
}

export default function FeatureSelectionPage() {
  const [params, setParams] = useState<ParamValues>(() => fromScenario(DEFAULT_FEATURES));
  const [activeId, setActiveId] = useState<string | null>(null);
  const scenario = useMemo(() => toScenario(params), [params]);
  const run = useMemo(() => runFeatures(scenario), [scenario]);
  const setParam = (key: string, value: ParamValue) => setParams((p) => ({ ...p, [key]: value }));

  const apply = (exp: FeaturesExperiment) => {
    setParams(fromScenario({ ...DEFAULT_FEATURES, ...exp.scenario }));
    setActiveId(exp.id);
  };
  const stillActive = (() => {
    const exp = featuresExperiments.find((e) => e.id === activeId);
    return (
      !!exp &&
      JSON.stringify(fromScenario({ ...DEFAULT_FEATURES, ...exp.scenario })) ===
        JSON.stringify(params)
    );
  })();

  const metrics: Metric[] = [
    { label: 'Test R² with the kept features', value: fix(run.testR2, 3), testId: 'test-r2' },
    {
      label: 'A model that knows the truth',
      value: fix(run.oracleR2, 3),
      hint: 'uses exactly the four true signals',
    },
    {
      label: 'Signals covered',
      value: `${run.signalsCovered} of 4`,
      hint: 'a signal and its copy count once',
    },
    {
      label: 'Duplicates kept',
      value: String(run.redundant),
      tone: run.redundant > 0 ? 'bad' : 'good',
    },
    {
      label: 'Noise columns kept',
      value: String(run.noisePicked),
      tone: run.noisePicked > 0 ? 'bad' : 'good',
    },
  ];

  const stages: [string, string][] = [
    ['lab-ranking', 'The ranking'],
    ['lab-curve', 'How many to keep'],
    ['lab-how', 'How it works'],
    ['lab-controls', 'Settings'],
  ];

  return (
    <div className="mx-auto max-w-6xl px-4 pt-8 pb-20 sm:px-6 lg:px-8">
      <LabHeader
        eyebrow="Evaluation & optimization"
        title="Feature selection"
        accent="and what importance means"
        blurb="Twenty columns: four carry the signal, four are noisy copies of them and twelve are noise. Filters, wrappers, embedded methods and importance scores each rank them, and each is fooled in its own way."
        badges={
          <>
            <Badge>{METHOD_LABELS[scenario.method]}</Badge>
            <Badge>keep {scenario.keep}</Badge>
          </>
        }
      />
      <StageNav stages={stages} />
      <ExperimentBar
        lessons={featuresExperiments}
        activeId={stillActive ? activeId : null}
        onApply={apply}
      />

      <div className="flex flex-col gap-5">
        <Panel id="lab-ranking" className="scroll-mt-24" title="The ranking" eyebrow="01">
          <div className="space-y-6">
            <p className="text-[0.8rem] text-muted">{METHOD_NOTES[scenario.method]}</p>
            <MetricsStrip metrics={metrics} className="lg:grid-cols-5" />
            <Ranking run={run} />
          </div>
        </Panel>
        <Panel id="lab-curve" className="scroll-mt-24" title="How many to keep" eyebrow="02">
          <div className="grid gap-8 lg:grid-cols-[minmax(0,1fr)_20rem]">
            <MiniChart
              title="Test R² of a linear model using the top k features of this ranking"
              x={run.curve.map((_, i) => i + 1)}
              xLabel="features kept"
              yRange={{ lo: Math.min(0, ...run.curve), hi: 1 }}
              series={[
                { name: 'this ranking', color: 'var(--brand)', values: run.curve },
                {
                  name: 'knows the truth',
                  color: 'var(--pos)',
                  values: run.curve.map(() => run.oracleR2),
                  dashed: true,
                },
              ]}
              height={210}
            />
            <p className="text-sm text-muted">
              A good ranking climbs quickly to the dashed line and then stays flat: the first few
              features are the useful ones and the rest add nothing. A ranking that picks twins
              early climbs more slowly, because the second twin adds no information.
            </p>
          </div>
        </Panel>
        <Panel id="lab-how" className="scroll-mt-24" title="How it works" eyebrow="03">
          <div className="space-y-3 text-sm text-muted">
            <p>
              <strong className="text-fg">Filters</strong> score each feature independently of any
              model. <strong className="text-fg">Wrappers</strong> search subsets by actually
              fitting models (forward selection adds the feature with the best cross-validated
              gain). <strong className="text-fg">Embedded</strong> methods let the model choose
              while it trains (the lasso&apos;s L1 penalty).
            </p>
            <p>
              <strong className="text-fg">Importance scores</strong> explain a fitted model rather
              than select features: impurity importance counts split gains inside trees and is
              biased towards features with many possible splits; permutation importance measures the
              score lost when a column is shuffled on held-out data. Both split the credit between
              correlated features.
            </p>
          </div>
        </Panel>
        <Panel id="lab-controls" className="scroll-mt-24" title="Settings" eyebrow="04">
          <ParamPanel specs={featuresParams} values={params} onChange={setParam} columns={3} />
        </Panel>
      </div>
    </div>
  );
}
