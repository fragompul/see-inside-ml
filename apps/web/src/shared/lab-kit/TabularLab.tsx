import { Dices } from 'lucide-react';
import { useCallback, useMemo, useState, type ReactNode } from 'react';
import { Badge, Button, Panel, Segmented, Slider } from '@/design-system';
import { ExperimentBar } from '@/shared/components/ExperimentBar';
import { tabularSetup, type TabularLabSpec } from './experiments';
import type { LabText } from './ClassifierLab';
import { MetricsStrip, fix, type Metric } from './MetricsStrip';
import { ParamPanel } from './ParamPanel';
import { withDefaults, type ParamValue, type ParamValues } from './params';
import { runRegressor, type RegressorRun } from './pipeline';
import { LabHeader, StageNav } from './StageNav';
import {
  TABULAR_SOURCES,
  buildTabular,
  type TabularSetup,
  type TabularSource,
} from './tabularPipeline';
import { CoefficientPlot, PredictedActual, type CoefficientRow } from './tabularCharts';

export interface TabularContext {
  run: RegressorRun;
  setup: TabularSetup;
  params: ParamValues;
  setParam: (key: string, value: ParamValue) => void;
  names: string[];
}

export interface TabularLabProps {
  spec: TabularLabSpec;
  text: LabText;
  /** The fitted coefficients to draw, with optional intervals and the known truth. */
  coefficients?: (ctx: TabularContext) => CoefficientRow[];
  howItWorks?: ReactNode;
  extras?: { id: string; title: string; render: (ctx: TabularContext) => ReactNode }[];
  extraMetrics?: (ctx: TabularContext) => Metric[];
  sources?: readonly TabularSource[];
}

/** The multi-feature twin of `RegressorLab`: named features, coefficient plot, predicted vs actual. */
export function TabularLab({
  spec,
  text,
  coefficients,
  howItWorks,
  extras = [],
  extraMetrics,
  sources,
}: TabularLabProps) {
  const [setup, setSetup] = useState<TabularSetup>(() => tabularSetup(spec));
  const [params, setParams] = useState<ParamValues>(() => withDefaults(spec.params, spec.defaults));
  const [activeId, setActiveId] = useState<string | null>(null);

  const data = useMemo(() => buildTabular(setup), [setup]);
  const outcome = useMemo(() => {
    try {
      return { run: runRegressor(spec.create(params), data), error: null as string | null };
    } catch (error) {
      return { run: null, error: error instanceof Error ? error.message : String(error) };
    }
  }, [spec, params, data]);
  const run = outcome.run;

  const setParam = useCallback(
    (key: string, value: ParamValue) => setParams((p) => ({ ...p, [key]: value })),
    [],
  );
  const ctx = useMemo<TabularContext | null>(
    () => (run ? { run, setup, params, setParam, names: data.all.names } : null),
    [run, setup, params, setParam, data],
  );

  const apply = (exp: (typeof spec.experiments)[number]) => {
    setSetup(tabularSetup(spec, exp.setup));
    setParams({ ...withDefaults(spec.params, spec.defaults), ...exp.params });
    setActiveId(exp.id);
    window.setTimeout(
      () =>
        document
          .getElementById('lab-plane')
          ?.scrollIntoView({ behavior: 'smooth', block: 'start' }),
      60,
    );
  };
  const stillActive = (() => {
    const exp = spec.experiments.find((e) => e.id === activeId);
    if (!exp) return false;
    return (
      JSON.stringify(tabularSetup(spec, exp.setup)) === JSON.stringify(setup) &&
      JSON.stringify({ ...withDefaults(spec.params, spec.defaults), ...exp.params }) ===
        JSON.stringify(params)
    );
  })();

  const metrics: Metric[] = run
    ? [
        { label: 'Training RMSE', value: fix(run.scores.trainRmse), testId: 'train-rmse' },
        { label: 'Test RMSE', value: fix(run.scores.testRmse), testId: 'test-rmse' },
        { label: 'Training R²', value: fix(run.scores.trainR2), testId: 'train-r2' },
        { label: 'Test R²', value: fix(run.scores.testR2), testId: 'test-r2' },
        ...(ctx && extraMetrics ? extraMetrics(ctx) : []),
      ]
    : [];

  const stages: [string, string][] = [
    ['lab-plane', 'The fit'],
    ...extras.map((e) => [e.id, e.title.split(':')[0] as string] as [string, string]),
    ...(howItWorks ? ([['lab-how', 'How it works']] as [string, string][]) : []),
    ['lab-controls', 'Settings'],
  ];
  const available = TABULAR_SOURCES.filter((s) => !sources || sources.includes(s.value));
  const info = TABULAR_SOURCES.find((s) => s.value === setup.source);
  const synthetic = setup.source === 'synthetic';

  return (
    <div className="mx-auto max-w-6xl px-4 pt-8 pb-20 sm:px-6 lg:px-8">
      <LabHeader
        {...text}
        badges={
          <>
            <Badge>{info?.label}</Badge>
            <Badge>{data.all.x.rows} rows</Badge>
            <Badge>{data.all.x.cols} features</Badge>
          </>
        }
      />
      <StageNav stages={stages} />
      <ExperimentBar
        lessons={spec.experiments}
        activeId={stillActive ? activeId : null}
        onApply={apply}
      />

      <div className="flex flex-col gap-5">
        <Panel id="lab-plane" className="scroll-mt-24" title="The data and the fit" eyebrow="01">
          <div className="space-y-6">
            <Segmented<TabularSource>
              label="Data"
              size="sm"
              fit
              value={setup.source}
              onValueChange={(source) => setSetup((s) => ({ ...s, source }))}
              options={available.map((s) => ({ value: s.value, label: s.label }))}
            />
            <p className="-mt-3 text-[0.8rem] text-muted">{info?.note}</p>
            {outcome.error && (
              <p
                role="alert"
                className="rounded-xl border border-danger/40 bg-danger/10 px-4 py-2 text-sm"
              >
                {outcome.error}
              </p>
            )}
            {run && ctx && (
              <div className="grid gap-8 lg:grid-cols-2">
                <div className="min-w-0 space-y-4">
                  {coefficients && (
                    <CoefficientPlot
                      rows={coefficients(ctx)}
                      ariaLabel="Fitted coefficients, with the true values when known"
                    />
                  )}
                </div>
                <div className="min-w-0 space-y-4">
                  <PredictedActual
                    train={{
                      actual: run.data.train.y,
                      predicted: run.model.predict(run.data.train.x),
                    }}
                    test={
                      run.data.test.x.rows > 0
                        ? { actual: run.data.test.y, predicted: run.model.predict(run.data.test.x) }
                        : null
                    }
                  />
                </div>
              </div>
            )}
            {run && <MetricsStrip metrics={metrics} />}
          </div>
        </Panel>

        {ctx &&
          extras.map((e, i) => (
            <Panel
              key={e.id}
              id={e.id}
              className="scroll-mt-24"
              title={e.title}
              eyebrow={String(i + 2).padStart(2, '0')}
            >
              {e.render(ctx)}
            </Panel>
          ))}

        {howItWorks && (
          <Panel
            id="lab-how"
            className="scroll-mt-24"
            title="How it works"
            eyebrow={String(extras.length + 2).padStart(2, '0')}
          >
            {howItWorks}
          </Panel>
        )}

        <Panel
          id="lab-controls"
          className="scroll-mt-24"
          title="Settings"
          eyebrow={String(extras.length + (howItWorks ? 3 : 2)).padStart(2, '0')}
        >
          <div className="space-y-8">
            <div>
              <p className="eyebrow mb-4">The model</p>
              <ParamPanel specs={spec.params} values={params} onChange={setParam} columns={3} />
            </div>
            <div>
              <p className="eyebrow mb-4">The data</p>
              <div className="grid gap-x-10 gap-y-6 sm:grid-cols-2 lg:grid-cols-3">
                {synthetic && (
                  <>
                    <Slider
                      label="Rows"
                      min={20}
                      max={400}
                      step={5}
                      value={setup.n}
                      onValueChange={(n) => setSetup((s) => ({ ...s, n }))}
                    />
                    <Slider
                      label="Noise"
                      min={0}
                      max={3}
                      step={0.05}
                      value={setup.noise}
                      onValueChange={(noise) =>
                        setSetup((s) => ({ ...s, noise: Number(noise.toFixed(2)) }))
                      }
                      format={(v) => v.toFixed(2)}
                    />
                    <Slider
                      label="Correlation of x₁ and x₂"
                      min={0}
                      max={0.99}
                      step={0.01}
                      value={setup.correlation}
                      onValueChange={(correlation) =>
                        setSetup((s) => ({ ...s, correlation: Number(correlation.toFixed(2)) }))
                      }
                      format={(v) => v.toFixed(2)}
                    />
                  </>
                )}
                <Slider
                  label="Share held out for testing"
                  min={0}
                  max={0.5}
                  step={0.05}
                  value={setup.testFraction}
                  onValueChange={(testFraction) =>
                    setSetup((s) => ({ ...s, testFraction: Number(testFraction.toFixed(2)) }))
                  }
                  format={(v) => `${Math.round(v * 100)}%`}
                />
              </div>
              <Button
                size="sm"
                variant="secondary"
                className="mt-5"
                onClick={() => setSetup((s) => ({ ...s, seed: Math.floor(Math.random() * 1e6) }))}
              >
                <Dices className="size-4" /> New random sample
              </Button>
            </div>
          </div>
        </Panel>
      </div>
    </div>
  );
}
