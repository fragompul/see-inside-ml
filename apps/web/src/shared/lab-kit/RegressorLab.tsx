import { REGRESSION_SETS, regressionSetInfo, type Table } from '@sim/core';
import { Dices, RotateCcw } from 'lucide-react';
import { useCallback, useMemo, useState, type ReactNode } from 'react';
import { Badge, Button, Panel, Segmented, Slider } from '@/design-system';
import { usePalette } from '@/shared/canvas/colors';
import { ExperimentBar } from '@/shared/components/ExperimentBar';
import { CurvePlane, type Band, type Curve } from './CurvePlane';
import type { EditTool, Painter } from './DataPlane';
import { regressionSetup, type RegressorLabSpec } from './experiments';
import { MetricsStrip, fix, type Metric } from './MetricsStrip';
import { ParamPanel } from './ParamPanel';
import { withDefaults, type ParamValue, type ParamValues } from './params';
import {
  buildRegression,
  predictCurve,
  runRegressor,
  type RegressionSetup,
  type RegressorRun,
} from './pipeline';
import { LabHeader, StageNav } from './StageNav';
import { addRow, moveRow, removeRow } from './tableEdit';
import type { LabText } from './ClassifierLab';

export interface RegressorContext {
  run: RegressorRun;
  curve: { xs: Float64Array; ys: Float64Array; x0: number; x1: number };
  params: ParamValues;
  setup: RegressionSetup;
  setParam: (key: string, value: ParamValue) => void;
  /** Replaces the data by a table of the lab's own (null restores the generated data). */
  setTable: (table: Table | null) => void;
}

export interface RegressorLabProps {
  spec: RegressorLabSpec;
  text: LabText;
  howItWorks?: ReactNode;
  extras?: { id: string; title: string; render: (ctx: RegressorContext) => ReactNode }[];
  /** Extra curves (e.g. the true function, other models) drawn with the fitted one. */
  curves?: (ctx: RegressorContext) => Curve[];
  bands?: (ctx: RegressorContext) => Band[];
  overlay?: (ctx: RegressorContext) => ((p: Painter) => void) | undefined;
  highlight?: (ctx: RegressorContext) => ReadonlySet<number> | undefined;
  extraMetrics?: (ctx: RegressorContext) => Metric[];
  datasets?: readonly (typeof REGRESSION_SETS)[number][];
  /** Offer the "show the errors" control (default true). */
  errors?: boolean;
}

const TOOLS: { value: EditTool; label: string }[] = [
  { value: 'none', label: 'Look' },
  { value: 'add', label: 'Add' },
  { value: 'move', label: 'Move' },
  { value: 'erase', label: 'Erase' },
];

type ErrorMode = 'none' | 'lines' | 'squares';

/** The regression twin of `ClassifierLab`: one feature, a fitted curve, errors and measured experiments. */
export function RegressorLab({
  spec,
  text,
  howItWorks,
  extras = [],
  curves: extraCurves,
  bands: bandsFn,
  overlay,
  highlight,
  extraMetrics,
  datasets,
  errors = true,
}: RegressorLabProps) {
  const palette = usePalette();
  const [setup, setSetup] = useState<RegressionSetup>(() => regressionSetup(spec));
  const [params, setParams] = useState<ParamValues>(() => withDefaults(spec.params, spec.defaults));
  const [custom, setCustom] = useState<Table | null>(null);
  const [tool, setTool] = useState<EditTool>('none');
  const [errorMode, setErrorMode] = useState<ErrorMode>('none');
  const [activeId, setActiveId] = useState<string | null>(null);

  const data = useMemo(() => buildRegression(setup, custom), [setup, custom]);
  const outcome = useMemo(() => {
    try {
      return { run: runRegressor(spec.create(params), data), error: null as string | null };
    } catch (error) {
      return { run: null, error: error instanceof Error ? error.message : String(error) };
    }
  }, [spec, params, data]);
  const run = outcome.run;

  const curve = useMemo(() => {
    if (!run) return null;
    try {
      return predictCurve(run.model, data);
    } catch {
      return null;
    }
  }, [run, data]);

  const setParam = useCallback(
    (key: string, value: ParamValue) => setParams((p) => ({ ...p, [key]: value })),
    [],
  );
  const ctx = useMemo<RegressorContext | null>(
    () => (run && curve ? { run, curve, params, setup, setParam, setTable: setCustom } : null),
    [run, curve, params, setup, setParam],
  );

  const fitted = useMemo(
    () => (ctx ? [{ xs: ctx.curve.xs, ys: ctx.curve.ys, color: palette.css.brand } as Curve] : []),
    [ctx, palette],
  );
  const allCurves = useMemo(
    () => [...(ctx ? (extraCurves?.(ctx) ?? []) : []), ...fitted],
    [ctx, extraCurves, fitted],
  );
  const bands = useMemo(() => (ctx ? (bandsFn?.(ctx) ?? []) : []), [ctx, bandsFn]);
  const overlayFn = useMemo(() => (ctx ? overlay?.(ctx) : undefined), [ctx, overlay]);
  const marked = useMemo(() => (ctx ? highlight?.(ctx) : undefined), [ctx, highlight]);
  const residuals = useMemo(
    () =>
      run && errorMode !== 'none' ? { pred: run.model.predict(data.all.x), mode: errorMode } : null,
    [run, data, errorMode],
  );

  const edit = (fn: (t: Table) => Table) => setCustom((c) => fn(c ?? data.all));

  const apply = (exp: (typeof spec.experiments)[number]) => {
    setSetup(regressionSetup(spec, exp.setup));
    setParams({ ...withDefaults(spec.params, spec.defaults), ...exp.params });
    setCustom(null);
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
    if (!exp || custom) return false;
    return (
      JSON.stringify(regressionSetup(spec, exp.setup)) === JSON.stringify(setup) &&
      JSON.stringify({ ...withDefaults(spec.params, spec.defaults), ...exp.params }) ===
        JSON.stringify(params)
    );
  })();

  const metrics: Metric[] = run
    ? [
        {
          label: 'Training RMSE',
          value: fix(run.scores.trainRmse),
          testId: 'train-rmse',
          hint: 'typical size of the error on the data the model saw',
        },
        {
          label: 'Test RMSE',
          value: fix(run.scores.testRmse),
          testId: 'test-rmse',
          hint: 'typical error on held-out points',
        },
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
  const sets = datasets ?? REGRESSION_SETS;

  return (
    <div className="mx-auto max-w-6xl px-4 pt-8 pb-20 sm:px-6 lg:px-8">
      <LabHeader
        {...text}
        badges={
          <>
            <Badge>{regressionSetInfo[setup.dataset].label}</Badge>
            <Badge>{data.all.x.rows} points</Badge>
            {custom && <Badge>edited by hand</Badge>}
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
        <Panel
          id="lab-plane"
          className="scroll-mt-24"
          title="The data and the fitted curve"
          eyebrow="01"
        >
          <div className="space-y-6">
            <div className="flex flex-wrap items-center gap-x-6 gap-y-3">
              <Segmented
                label="Dataset"
                size="sm"
                fit
                value={setup.dataset}
                onValueChange={(dataset) => {
                  setSetup((s) => ({ ...s, dataset }));
                  setCustom(null);
                }}
                options={sets.map((d) => ({ value: d, label: regressionSetInfo[d].label }))}
              />
              <Segmented<EditTool>
                label="Editing tool"
                size="sm"
                fit
                value={tool}
                onValueChange={setTool}
                options={TOOLS}
              />
              {errors && (
                <Segmented<ErrorMode>
                  label="Show the errors"
                  size="sm"
                  fit
                  value={errorMode}
                  onValueChange={setErrorMode}
                  options={[
                    { value: 'none', label: 'No errors' },
                    { value: 'lines', label: 'Error lines' },
                    { value: 'squares', label: 'Squared errors' },
                  ]}
                />
              )}
              {custom && (
                <Button size="sm" variant="ghost" onClick={() => setCustom(null)}>
                  <RotateCcw className="size-4" /> Restore the data
                </Button>
              )}
            </div>
            <p className="-mt-3 text-[0.8rem] text-muted">
              {regressionSetInfo[setup.dataset].note}
            </p>

            <div className="grid gap-8 lg:grid-cols-[auto_minmax(0,1fr)]">
              <div className="space-y-3">
                <CurvePlane
                  all={data.all}
                  testMask={data.testMask}
                  curves={allCurves}
                  bands={bands}
                  residuals={residuals}
                  highlight={marked ?? null}
                  tool={tool}
                  onAdd={(x, y) => edit((t) => addRow(t, [x], y))}
                  onMove={(i, x, y) => edit((t) => moveRow(t, i, [x], y))}
                  onErase={(i) => edit((t) => removeRow(t, i))}
                  {...(overlayFn ? { overlay: overlayFn } : {})}
                  ariaLabel="Data points and the curve the model fits through them. Filled dots train the model, hollow dots test it."
                />
                <p className="text-[0.74rem] text-faint">
                  Filled dots train the model, hollow dots are held out to test it.
                  {tool === 'none' ? ' Pick Add, Move or Erase to edit the data by hand.' : ''}
                </p>
              </div>
              <div className="min-w-0 space-y-5">
                {outcome.error && (
                  <p
                    role="alert"
                    className="rounded-xl border border-danger/40 bg-danger/10 px-4 py-2 text-sm"
                  >
                    {outcome.error}
                  </p>
                )}
                {run && <MetricsStrip metrics={metrics} className="lg:grid-cols-2" />}
              </div>
            </div>
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
                <Slider
                  label="Points"
                  min={10}
                  max={400}
                  step={5}
                  value={setup.n}
                  onValueChange={(n) => {
                    setSetup((s) => ({ ...s, n }));
                    setCustom(null);
                  }}
                />
                <Slider
                  label="Noise"
                  min={0}
                  max={2}
                  step={0.05}
                  value={setup.noise}
                  onValueChange={(noise) => {
                    setSetup((s) => ({ ...s, noise: Number(noise.toFixed(2)) }));
                    setCustom(null);
                  }}
                  format={(v) => v.toFixed(2)}
                />
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
                onClick={() => {
                  setSetup((s) => ({ ...s, seed: Math.floor(Math.random() * 1e6) }));
                  setCustom(null);
                }}
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
