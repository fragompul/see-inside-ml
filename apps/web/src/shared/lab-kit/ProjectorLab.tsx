import { pairwiseDistances } from '@sim/classic';
import { Dices, Pause, Play, RotateCcw } from 'lucide-react';
import { useCallback, useEffect, useMemo, useRef, useState, type ReactNode } from 'react';
import { Badge, Button, Panel, Segmented, Slider } from '@/design-system';
import { ExperimentBar } from '@/shared/components/ExperimentBar';
import type { LabText } from './ClassifierLab';
import { EmbeddingPlane } from './EmbeddingPlane';
import { projectionSetup, type ProjectorLabSpec } from './experiments';
import { MetricsStrip, fix, type Metric } from './MetricsStrip';
import { ParamPanel } from './ParamPanel';
import { withDefaults, type ParamValue, type ParamValues } from './params';
import {
  PROJECTION_DATASETS,
  buildProjectionData,
  projectionDatasetInfo,
  type ProjectionData,
  type ProjectionSetup,
} from './projectionData';
import { scoreEmbedding, type EmbedFrame, type EmbeddingScores } from './projectorPipeline';
import { LabHeader, StageNav } from './StageNav';

export interface ProjectorContext {
  data: ProjectionData;
  frame: EmbedFrame;
  /** Final quality scores (null while an iterative method is still running). */
  scores: EmbeddingScores | null;
  done: boolean;
  params: ParamValues;
  setup: ProjectionSetup;
  setParam: (key: string, value: ParamValue) => void;
}

export interface ProjectorLabProps {
  spec: ProjectorLabSpec;
  text: LabText;
  howItWorks?: ReactNode;
  extras?: { id: string; title: string; render: (ctx: ProjectorContext) => ReactNode }[];
  extraMetrics?: (ctx: ProjectorContext) => Metric[];
  /** Show the speed control (iterative methods only). */
  iterative?: boolean;
}

interface Running {
  key: string;
  frame: EmbedFrame;
  done: boolean;
  scores: EmbeddingScores | null;
  error: string | null;
}

const SPEEDS = [1, 4, 16, 64] as const;

/** The dimensionality-reduction lab: a dataset, a method that animates towards a 2-D layout, and how well it kept the neighbours. */
export function ProjectorLab({
  spec,
  text,
  howItWorks,
  extras = [],
  extraMetrics,
  iterative = false,
}: ProjectorLabProps) {
  const [setup, setSetup] = useState<ProjectionSetup>(() => projectionSetup(spec));
  const [params, setParams] = useState<ParamValues>(() => withDefaults(spec.params, spec.defaults));
  const [activeId, setActiveId] = useState<string | null>(null);
  const [playing, setPlaying] = useState(true);
  const [speed, setSpeed] = useState<number>(iterative ? 4 : 1);
  const [restarts, setRestarts] = useState(0);
  const [focus, setFocus] = useState<number | null>(null);
  const [current, setCurrent] = useState<Running | null>(null);
  const playingRef = useRef(true);
  const speedRef = useRef<number>(speed);
  // the animation loop reads these without being restarted when they change
  useEffect(() => {
    playingRef.current = playing;
    speedRef.current = speed;
  }, [playing, speed]);

  const data = useMemo(() => buildProjectionData(setup), [setup]);
  const key = useMemo(() => JSON.stringify([setup, params, restarts]), [setup, params, restarts]);

  // drive the method's generator a few steps per animation frame
  useEffect(() => {
    let cancelled = false;
    let raf = 0;
    let generator: Generator<EmbedFrame, void, undefined> | null = null;
    let failure: string | null = null;
    let latest: EmbedFrame | null = null;
    try {
      generator = spec.create(params).run(data.x);
    } catch (error) {
      failure = error instanceof Error ? error.message : String(error);
    }
    const tick = () => {
      if (cancelled) return;
      if (!playingRef.current && generator) {
        raf = requestAnimationFrame(tick);
        return;
      }
      let done = false;
      try {
        if (!generator) throw new Error(failure ?? 'The method could not start');
        for (let step = 0; step < speedRef.current; step++) {
          const r = generator.next();
          if (r.done) {
            done = true;
            break;
          }
          latest = r.value;
        }
        if (!latest) throw new Error('The method produced no embedding');
        const frame = latest;
        setCurrent({
          key,
          frame,
          done,
          scores: done ? scoreEmbedding(data.x, frame.embedding) : null,
          error: null,
        });
        if (!done) raf = requestAnimationFrame(tick);
      } catch (error) {
        setCurrent({
          key,
          frame: { embedding: latest?.embedding ?? data.x },
          done: true,
          scores: null,
          error: error instanceof Error ? error.message : String(error),
        });
      }
    };
    raf = requestAnimationFrame(tick);
    return () => {
      cancelled = true;
      cancelAnimationFrame(raf);
    };
  }, [key, spec, data, params]);

  const run = current && current.key === key ? current : null;
  const setParam = useCallback(
    (k: string, value: ParamValue) => setParams((p) => ({ ...p, [k]: value })),
    [],
  );
  const ctx = useMemo<ProjectorContext | null>(
    () =>
      run && !run.error
        ? { data, frame: run.frame, scores: run.scores, done: run.done, params, setup, setParam }
        : null,
    [run, data, params, setup, setParam],
  );

  // distances in the original space, for "who are this point's real neighbours?"
  const original = useMemo(() => pairwiseDistances(data.x), [data]);
  const related = useMemo(() => {
    if (focus === null) return null;
    const n = data.x.rows;
    const near = Array.from({ length: n }, (_, j) => j)
      .filter((j) => j !== focus)
      .sort((a, b) => (original[focus * n + a] as number) - (original[focus * n + b] as number))
      .slice(0, 8);
    return new Set(near);
  }, [focus, original, data]);

  const apply = (exp: (typeof spec.experiments)[number]) => {
    setSetup(projectionSetup(spec, exp.setup));
    setParams({ ...withDefaults(spec.params, spec.defaults), ...exp.params });
    setActiveId(exp.id);
    setPlaying(true);
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
      JSON.stringify(projectionSetup(spec, exp.setup)) === JSON.stringify(setup) &&
      JSON.stringify({ ...withDefaults(spec.params, spec.defaults), ...exp.params }) ===
        JSON.stringify(params)
    );
  })();

  const metrics: Metric[] = [
    { label: 'Original dimensions', value: String(data.dimensions), testId: 'dimensions' },
    {
      label: 'Trustworthiness',
      value: run?.scores ? fix(run.scores.trustworthiness) : '…',
      testId: 'trust',
      hint: 'neighbours in the picture that are truly near in the data (1 = no false neighbours)',
    },
    {
      label: 'Continuity',
      value: run?.scores ? fix(run.scores.continuity) : '…',
      testId: 'continuity',
      hint: 'neighbours in the data that are still near in the picture',
    },
    {
      label: 'Neighbours kept',
      value: run?.scores ? fix(run.scores.preservation) : '…',
      testId: 'preservation',
      hint: 'share of each point’s 5 nearest neighbours that are the same in both spaces',
    },
    ...(ctx && extraMetrics ? extraMetrics(ctx) : []),
  ];

  const sets = spec.datasets ?? PROJECTION_DATASETS;
  const stages: [string, string][] = [
    ['lab-plane', 'The embedding'],
    ...extras.map((e) => [e.id, e.title.split(':')[0] as string] as [string, string]),
    ...(howItWorks ? ([['lab-how', 'How it works']] as [string, string][]) : []),
    ['lab-controls', 'Settings'],
  ];

  return (
    <div className="mx-auto max-w-6xl px-4 pt-8 pb-20 sm:px-6 lg:px-8">
      <LabHeader
        {...text}
        badges={
          <>
            <Badge>{projectionDatasetInfo[setup.dataset].label}</Badge>
            <Badge>{data.x.rows} points</Badge>
            <Badge>{data.dimensions}-D → 2-D</Badge>
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
          title="The data in two dimensions"
          eyebrow="01"
        >
          <div className="space-y-6">
            <div className="flex flex-wrap items-center gap-x-6 gap-y-3">
              <Segmented
                label="Dataset"
                size="sm"
                fit
                value={setup.dataset}
                onValueChange={(dataset) => setSetup((s) => ({ ...s, dataset }))}
                options={sets.map((d) => ({ value: d, label: projectionDatasetInfo[d].label }))}
              />
              {iterative && (
                <>
                  <Button
                    size="sm"
                    variant="secondary"
                    aria-label={playing ? 'Pause the optimisation' : 'Resume the optimisation'}
                    onClick={() => setPlaying((p) => !p)}
                  >
                    {playing ? <Pause className="size-4" /> : <Play className="size-4" />}
                    {playing ? 'Pause' : 'Play'}
                  </Button>
                  <Button size="sm" variant="ghost" onClick={() => setRestarts((r) => r + 1)}>
                    <RotateCcw className="size-4" /> Restart
                  </Button>
                  <Segmented<string>
                    label="Iterations per frame"
                    size="sm"
                    fit
                    value={String(speed)}
                    onValueChange={(v) => setSpeed(Number(v))}
                    options={SPEEDS.map((s) => ({ value: String(s), label: `×${s}` }))}
                  />
                </>
              )}
            </div>
            <p className="-mt-3 text-[0.8rem] text-muted">
              {projectionDatasetInfo[setup.dataset].note}
            </p>

            <div className="grid gap-8 lg:grid-cols-[auto_minmax(0,1fr)]">
              <div className="space-y-3">
                {run && !run.error ? (
                  <EmbeddingPlane
                    points={run.frame.embedding}
                    values={data.y}
                    colour={data.colour}
                    related={related}
                    focus={focus}
                    onFocus={setFocus}
                    ariaLabel="The data squeezed into two dimensions. Hover a point to ring its nearest neighbours in the original space."
                  />
                ) : (
                  <div
                    className="grid h-[400px] w-[520px] max-w-full place-items-center rounded-xl border border-line-strong text-sm text-muted"
                    aria-busy={!run}
                  >
                    {run?.error ?? 'Computing…'}
                  </div>
                )}
                <p className="text-[0.74rem] text-faint">
                  Hover a point: the ringed points are its true nearest neighbours in the original{' '}
                  {data.dimensions}-D space. If they sit close to it here, the picture is faithful
                  at that spot.
                </p>
              </div>
              <div className="min-w-0 space-y-5">
                {run?.error && (
                  <p
                    role="alert"
                    className="rounded-xl border border-danger/40 bg-danger/10 px-4 py-2 text-sm"
                  >
                    {run.error}
                  </p>
                )}
                <MetricsStrip metrics={metrics} className="lg:grid-cols-2" />
                {run && !run.error && (
                  <p
                    className="font-mono text-[0.78rem] text-muted tabular-nums"
                    data-testid="embed-info"
                  >
                    {run.done ? 'done' : 'running'}
                    {run.frame.info ? ` · ${run.frame.info}` : ''}
                  </p>
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
              <p className="eyebrow mb-4">The method</p>
              <ParamPanel specs={spec.params} values={params} onChange={setParam} columns={3} />
            </div>
            <div>
              <p className="eyebrow mb-4">The data</p>
              <div className="grid gap-x-10 gap-y-6 sm:grid-cols-2 lg:grid-cols-3">
                <Slider
                  label="Points"
                  min={60}
                  max={300}
                  step={10}
                  value={setup.n}
                  onValueChange={(n) => setSetup((s) => ({ ...s, n }))}
                />
                <Slider
                  label="Noise"
                  min={0}
                  max={1}
                  step={0.05}
                  value={setup.noise}
                  onValueChange={(noise) =>
                    setSetup((s) => ({ ...s, noise: Number(noise.toFixed(2)) }))
                  }
                  format={(v) => v.toFixed(2)}
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
