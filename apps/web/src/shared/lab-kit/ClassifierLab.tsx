import {
  CLASSIFICATION_SETS,
  classificationSetInfo,
  makeClassification,
  createRng,
  type Table,
} from '@sim/core';
import { Dices, RotateCcw } from 'lucide-react';
import { useCallback, useMemo, useState, type ReactNode } from 'react';
import { Badge, Button, Panel, Segmented, Slider, cn } from '@/design-system';
import { ConfusionMatrix } from '@/shared/components/ConfusionMatrix';
import { ExperimentBar } from '@/shared/components/ExperimentBar';
import { DataPlane, type EditTool, type Painter } from './DataPlane';
import { classificationSetup, type ClassifierLabSpec } from './experiments';
import { MetricsStrip, fix, pct, type Metric } from './MetricsStrip';
import { ParamPanel } from './ParamPanel';
import { withDefaults, type ParamValue, type ParamValues } from './params';
import {
  buildClassification,
  predictRegions,
  runClassifier,
  type ClassificationSetup,
  type ClassifierRun,
  type Regions,
} from './pipeline';
import { LabHeader, StageNav } from './StageNav';
import { addRow, moveRow, removeRow } from './tableEdit';
import { plotBounds } from '@sim/core';

export interface LabText {
  eyebrow: string;
  title: string;
  accent: string;
  blurb: string;
}

export interface ClassifierContext {
  run: ClassifierRun;
  regions: Regions;
  params: ParamValues;
  setup: ClassificationSetup;
  setParam: (key: string, value: ParamValue) => void;
  /** Replaces the data by a table of the lab's own (null restores the generated data). */
  setTable: (table: Table | null) => void;
}

export interface ClassifierLabProps {
  spec: ClassifierLabSpec;
  text: LabText;
  /** The "how it works" section (explanation, formulas). */
  howItWorks?: ReactNode;
  /** Extra sections specific to the model, rendered after the plane. */
  extras?: { id: string; title: string; render: (ctx: ClassifierContext) => ReactNode }[];
  overlay?: (ctx: ClassifierContext) => ((p: Painter) => void) | undefined;
  underlay?: (ctx: ClassifierContext) => ((p: Painter) => void) | undefined;
  highlight?: (ctx: ClassifierContext) => ReadonlySet<number> | undefined;
  extraMetrics?: (ctx: ClassifierContext) => Metric[];
  /** Datasets offered (default: all). */
  datasets?: readonly (typeof CLASSIFICATION_SETS)[number][];
}

const TOOLS: { value: EditTool; label: string }[] = [
  { value: 'none', label: 'Look' },
  { value: 'add', label: 'Add' },
  { value: 'move', label: 'Move' },
  { value: 'erase', label: 'Erase' },
];

/**
 * A complete lab for a two-feature classifier: editable data, decision regions, scores, the
 * confusion matrix, the model's controls and measured experiments. Models plug in through a
 * `ClassifierLabSpec`; model-specific views come in as `extras`/`overlay`.
 */
export function ClassifierLab({
  spec,
  text,
  howItWorks,
  extras = [],
  overlay,
  underlay,
  highlight,
  extraMetrics,
  datasets,
}: ClassifierLabProps) {
  const [setup, setSetup] = useState<ClassificationSetup>(() => classificationSetup(spec));
  const [params, setParams] = useState<ParamValues>(() => withDefaults(spec.params, spec.defaults));
  const [custom, setCustom] = useState<Table | null>(null);
  const [tool, setTool] = useState<EditTool>('none');
  const [activeClass, setActiveClass] = useState(0);
  const [activeId, setActiveId] = useState<string | null>(null);

  const data = useMemo(() => buildClassification(setup, custom), [setup, custom]);
  const outcome = useMemo(() => {
    try {
      return { run: runClassifier(spec.create(params), data), error: null as string | null };
    } catch (error) {
      return { run: null, error: error instanceof Error ? error.message : String(error) };
    }
  }, [spec, params, data]);
  const run = outcome.run;

  const regions = useMemo(() => {
    if (!run) return null;
    try {
      return predictRegions(run.model, data);
    } catch {
      return null;
    }
  }, [run, data]);

  const bounds = useMemo(() => plotBounds(data.all.x, 0.2), [data]);
  const wrong = useMemo(() => {
    if (!run || data.all.x.rows === 0) return null;
    const pred = run.model.predict(data.all.x);
    return Uint8Array.from(pred, (p, i) => (p === data.all.y[i] ? 0 : 1));
  }, [run, data]);

  const setParam = useCallback((key: string, value: ParamValue) => {
    setParams((p) => ({ ...p, [key]: value }));
  }, []);

  const ctx = useMemo<ClassifierContext | null>(
    () => (run && regions ? { run, regions, params, setup, setParam, setTable: setCustom } : null),
    [run, regions, params, setup, setParam],
  );
  const overlayFn = useMemo(() => (ctx ? overlay?.(ctx) : undefined), [ctx, overlay]);
  const underlayFn = useMemo(() => (ctx ? underlay?.(ctx) : undefined), [ctx, underlay]);
  const marked = useMemo(() => (ctx ? highlight?.(ctx) : undefined), [ctx, highlight]);

  const classes = data.all.classes ?? ['A', 'B'];
  const edit = (fn: (t: Table) => Table) => setCustom((c) => fn(c ?? data.all));

  const apply = (exp: (typeof spec.experiments)[number]) => {
    setSetup(classificationSetup(spec, exp.setup));
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
      JSON.stringify(classificationSetup(spec, exp.setup)) === JSON.stringify(setup) &&
      JSON.stringify({ ...withDefaults(spec.params, spec.defaults), ...exp.params }) ===
        JSON.stringify(params)
    );
  })();

  const metrics: Metric[] = run
    ? [
        {
          label: 'Training accuracy',
          value: pct(run.scores.trainAccuracy),
          testId: 'train-accuracy',
        },
        {
          label: 'Test accuracy',
          value: pct(run.scores.testAccuracy),
          testId: 'test-accuracy',
          hint: 'on points the model never saw',
        },
        ...(run.scores.trainLoss !== null
          ? [{ label: 'Training log-loss', value: fix(run.scores.trainLoss) }]
          : []),
        ...(run.scores.testLoss !== null
          ? [{ label: 'Test log-loss', value: fix(run.scores.testLoss) }]
          : []),
        ...(ctx && extraMetrics ? extraMetrics(ctx) : []),
      ]
    : [];

  const stages: [string, string][] = [
    ['lab-plane', 'The model'],
    ...extras.map((e) => [e.id, e.title.split(':')[0] as string] as [string, string]),
    ...(howItWorks ? ([['lab-how', 'How it works']] as [string, string][]) : []),
    ['lab-controls', 'Settings'],
  ];
  const sets = datasets ?? CLASSIFICATION_SETS;

  return (
    <div className="mx-auto max-w-6xl px-4 pt-8 pb-20 sm:px-6 lg:px-8">
      <LabHeader
        {...text}
        badges={
          <>
            <Badge>{classificationSetInfo[setup.dataset].label}</Badge>
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
          title="The data and the decision regions"
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
                options={sets.map((d) => ({ value: d, label: classificationSetInfo[d].label }))}
              />
              <Segmented<EditTool>
                label="Editing tool"
                size="sm"
                fit
                value={tool}
                onValueChange={setTool}
                options={TOOLS}
              />
              {tool === 'add' && (
                <div className="flex items-center gap-1.5" role="group" aria-label="Class to add">
                  {classes.map((name, c) => (
                    <button
                      key={name}
                      type="button"
                      aria-pressed={activeClass === c}
                      onClick={() => setActiveClass(c)}
                      className={cn(
                        'cursor-pointer rounded-full border px-3 py-1 font-mono text-[0.74rem] transition-colors',
                        activeClass === c
                          ? 'border-brand bg-brand-soft text-fg'
                          : 'border-line-strong text-muted hover:text-fg',
                      )}
                    >
                      <span
                        className="mr-1.5 inline-block size-2 rounded-full"
                        style={{ background: `var(--class-${c % 5})` }}
                      />
                      {name}
                    </button>
                  ))}
                </div>
              )}
              {custom && (
                <Button size="sm" variant="ghost" onClick={() => setCustom(null)}>
                  <RotateCcw className="size-4" /> Restore the data
                </Button>
              )}
            </div>
            <p className="-mt-3 text-[0.8rem] text-muted">
              {classificationSetInfo[setup.dataset].note}
            </p>

            <div className="grid gap-8 lg:grid-cols-[auto_minmax(0,1fr)]">
              <div className="space-y-3">
                <DataPlane
                  all={data.all}
                  testMask={data.testMask}
                  regions={regions}
                  bounds={bounds}
                  wrong={wrong}
                  highlight={marked ?? null}
                  tool={tool}
                  activeClass={activeClass}
                  onAdd={(x, y, c) => edit((t) => addRow(t, [x, y], c))}
                  onMove={(i, x, y) => edit((t) => moveRow(t, i, [x, y]))}
                  onErase={(i) => edit((t) => removeRow(t, i))}
                  {...(underlayFn ? { underlay: underlayFn } : {})}
                  {...(overlayFn ? { overlay: overlayFn } : {})}
                  ariaLabel="Points of each class over the regions the model assigns to them. Filled dots train the model, hollow dots test it, a ring marks a mistake."
                />
                <p className="text-[0.74rem] text-faint">
                  Filled dots are the training data, hollow dots the held-out test data; a ring
                  around a dot marks a mistake.
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
                {run && (
                  <div>
                    <p className="eyebrow mb-3">
                      Confusion matrix ({run.data.test.x.rows > 0 ? 'test' : 'training'} points)
                    </p>
                    <div className="max-w-xs">
                      <ConfusionMatrix
                        matrix={run.scores.confusion}
                        labels={classes}
                        noun="point"
                      />
                    </div>
                  </div>
                )}
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
                  min={30}
                  max={600}
                  step={10}
                  value={setup.n}
                  onValueChange={(n) => {
                    setSetup((s) => ({ ...s, n }));
                    setCustom(null);
                  }}
                />
                <Slider
                  label="Noise"
                  min={0}
                  max={0.8}
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

/** A generated table with the lab's current setup (used by tests and by labs that need the raw data). */
export function tableFor(setup: ClassificationSetup): Table {
  return makeClassification(setup.dataset, setup.n, setup.noise, createRng(setup.seed));
}
