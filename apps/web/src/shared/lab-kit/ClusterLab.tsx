import { CLASSIFICATION_SETS, classificationSetInfo, plotBounds, type Table } from '@sim/core';
import { Dices, RotateCcw } from 'lucide-react';
import { useCallback, useMemo, useState, type ReactNode } from 'react';
import { Badge, Button, Panel, Segmented, Slider } from '@/design-system';
import { ExperimentBar } from '@/shared/components/ExperimentBar';
import type { LabText } from './ClassifierLab';
import {
  buildClusterData,
  clusterRegions,
  runClusterer,
  type ClusterRun,
  type ClusterSetup,
} from './clusterPipeline';
import { DataPlane, type EditTool, type Painter } from './DataPlane';
import { clusterSetup, type ClusterLabSpec } from './experiments';
import { MetricsStrip, fix, type Metric } from './MetricsStrip';
import { ParamPanel } from './ParamPanel';
import { withDefaults, type ParamValue, type ParamValues } from './params';
import type { Regions } from './pipeline';
import { LabHeader, StageNav } from './StageNav';
import { addRow, moveRow, removeRow } from './tableEdit';

export interface ClusterContext {
  run: ClusterRun;
  regions: Regions | null;
  params: ParamValues;
  setup: ClusterSetup;
  setParam: (key: string, value: ParamValue) => void;
  setTable: (table: Table | null) => void;
}

export interface ClusterLabProps {
  spec: ClusterLabSpec;
  text: LabText;
  howItWorks?: ReactNode;
  extras?: { id: string; title: string; render: (ctx: ClusterContext) => ReactNode }[];
  overlay?: (ctx: ClusterContext) => ((p: Painter) => void) | undefined;
  underlay?: (ctx: ClusterContext) => ((p: Painter) => void) | undefined;
  highlight?: (ctx: ClusterContext) => ReadonlySet<number> | undefined;
  extraMetrics?: (ctx: ClusterContext) => Metric[];
  datasets?: readonly (typeof CLASSIFICATION_SETS)[number][];
  /** Shade the plane by the cluster a new point would join, when the model can predict (default true). */
  showRegions?: boolean;
  /** Offer a slider for scattered outliers and name the colourings differently (anomaly detection). */
  anomaly?: boolean;
  /** Replace the cluster metrics (clusters found, silhouette, ARI) with the lab's own. */
  metricsOnly?: (ctx: ClusterContext) => Metric[];
}

const TOOLS: { value: EditTool; label: string }[] = [
  { value: 'none', label: 'Look' },
  { value: 'add', label: 'Add' },
  { value: 'move', label: 'Move' },
  { value: 'erase', label: 'Erase' },
];

type ColorBy = 'found' | 'truth';

/** The unsupervised twin of `ClassifierLab`: points without labels, clusters found, scored against the hidden truth. */
export function ClusterLab({
  spec,
  text,
  howItWorks,
  extras = [],
  overlay,
  underlay,
  highlight,
  extraMetrics,
  datasets,
  showRegions = true,
  anomaly = false,
  metricsOnly,
}: ClusterLabProps) {
  const [setup, setSetup] = useState<ClusterSetup>(() => clusterSetup(spec));
  const [params, setParams] = useState<ParamValues>(() => withDefaults(spec.params, spec.defaults));
  const [custom, setCustom] = useState<Table | null>(null);
  const [tool, setTool] = useState<EditTool>('none');
  const [colorBy, setColorBy] = useState<ColorBy>('found');
  const [activeId, setActiveId] = useState<string | null>(null);

  const data = useMemo(() => buildClusterData(setup, custom), [setup, custom]);
  const outcome = useMemo(() => {
    try {
      return { run: runClusterer(spec.create(params), data), error: null as string | null };
    } catch (error) {
      return { run: null, error: error instanceof Error ? error.message : String(error) };
    }
  }, [spec, params, data]);
  const run = outcome.run;
  const regions = useMemo(() => {
    if (!run || !showRegions) return null;
    try {
      return clusterRegions(run.model, data);
    } catch {
      return null;
    }
  }, [run, data, showRegions]);
  const bounds = useMemo(() => plotBounds(data.x, 0.2), [data]);

  const setParam = useCallback(
    (key: string, value: ParamValue) => setParams((p) => ({ ...p, [key]: value })),
    [],
  );
  const ctx = useMemo<ClusterContext | null>(
    () => (run ? { run, regions, params, setup, setParam, setTable: setCustom } : null),
    [run, regions, params, setup, setParam],
  );
  const overlayFn = useMemo(() => (ctx ? overlay?.(ctx) : undefined), [ctx, overlay]);
  const underlayFn = useMemo(() => (ctx ? underlay?.(ctx) : undefined), [ctx, underlay]);
  const marked = useMemo(() => (ctx ? highlight?.(ctx) : undefined), [ctx, highlight]);

  // what the points are painted with: the clusters found or the groups that were really there
  const painted = useMemo<Table>(
    () => (run && colorBy === 'found' ? { ...data, y: Float64Array.from(run.labels) } : data),
    [run, data, colorBy],
  );
  const noTest = useMemo(() => new Uint8Array(data.x.rows), [data]);
  const edit = (fn: (t: Table) => Table) => setCustom((c) => fn(c ?? data));

  const apply = (exp: (typeof spec.experiments)[number]) => {
    setSetup(clusterSetup(spec, exp.setup));
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
      JSON.stringify(clusterSetup(spec, exp.setup)) === JSON.stringify(setup) &&
      JSON.stringify({ ...withDefaults(spec.params, spec.defaults), ...exp.params }) ===
        JSON.stringify(params)
    );
  })();

  const metrics: Metric[] =
    run && ctx && metricsOnly
      ? metricsOnly(ctx)
      : run
        ? [
            { label: 'Clusters found', value: String(run.scores.clusters), testId: 'clusters' },
            { label: 'Noise points', value: String(run.scores.noise), testId: 'noise' },
            {
              label: 'Silhouette',
              value: run.scores.silhouette === null ? '–' : fix(run.scores.silhouette),
              testId: 'silhouette',
              hint: 'how well separated the clusters are, without using the true groups (−1 … 1)',
            },
            {
              label: 'Match with the true groups',
              value: fix(run.scores.ari),
              testId: 'ari',
              hint: 'adjusted Rand index against the hidden groups: 1 = identical, near 0 = chance',
            },
            ...(ctx && extraMetrics ? extraMetrics(ctx) : []),
          ]
        : [];

  const stages: [string, string][] = [
    ['lab-plane', 'The clusters'],
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
            <Badge>{data.x.rows} points</Badge>
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
          title="The data and the clusters"
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
              <Segmented<ColorBy>
                label="Colour the points by"
                size="sm"
                fit
                value={colorBy}
                onValueChange={setColorBy}
                options={[
                  { value: 'found', label: anomaly ? 'Verdict' : 'Clusters found' },
                  { value: 'truth', label: anomaly ? 'Real points / strays' : 'True groups' },
                ]}
              />
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
                  all={painted}
                  testMask={noTest}
                  regions={colorBy === 'found' ? regions : null}
                  bounds={bounds}
                  highlight={marked ?? null}
                  tool={tool}
                  onAdd={(x, y) => edit((t) => addRow(t, [x, y], 0))}
                  onMove={(i, x, y) => edit((t) => moveRow(t, i, [x, y]))}
                  onErase={(i) => edit((t) => removeRow(t, i))}
                  {...(underlayFn ? { underlay: underlayFn } : {})}
                  {...(overlayFn ? { overlay: overlayFn } : {})}
                  ariaLabel="The points, coloured by the cluster the algorithm assigned (grey = noise) or by the true group."
                />
                <p className="text-[0.74rem] text-faint">
                  The algorithm never sees the true groups; they are only used to score it.
                  {tool === 'none' ? ' Pick Add, Move or Erase to reshape the data by hand.' : ''}
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
              <p className="eyebrow mb-4">The algorithm</p>
              <ParamPanel specs={spec.params} values={params} onChange={setParam} columns={3} />
            </div>
            <div>
              <p className="eyebrow mb-4">The data</p>
              <div className="grid gap-x-10 gap-y-6 sm:grid-cols-2 lg:grid-cols-3">
                <Slider
                  label="Points"
                  min={30}
                  max={400}
                  step={10}
                  value={setup.n}
                  onValueChange={(n) => {
                    setSetup((s) => ({ ...s, n }));
                    setCustom(null);
                  }}
                />
                {anomaly && (
                  <Slider
                    label="Injected outliers"
                    min={0}
                    max={60}
                    step={2}
                    value={setup.outliers}
                    onValueChange={(outliers) => {
                      setSetup((s) => ({ ...s, outliers }));
                      setCustom(null);
                    }}
                  />
                )}
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
