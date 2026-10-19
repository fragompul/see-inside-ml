import { useMemo, useState } from 'react';
import { Badge, Panel } from '@/design-system';
import { usePalette } from '@/shared/canvas/colors';
import { useCompute } from '@/shared/compute/useCompute';
import { ExperimentBar } from '@/shared/components/ExperimentBar';
import { Heat } from '@/shared/components/Heat';
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
  DEFAULT_FLOW_LAB,
  FLOW_DATASETS,
  GRID,
  HALF,
  type FlowLabRun,
  type FlowScenario,
} from './flowCore';
import { flowExperiments, flowParams, type FlowExperiment } from './flowSpec';

const toScenario = (p: ParamValues): FlowScenario => ({
  ...DEFAULT_FLOW_LAB,
  dataset: p.dataset as FlowScenario['dataset'],
  layers: Number(p.layers),
  steps: Number(p.steps),
});

const fromScenario = (s: FlowScenario): ParamValues => ({
  dataset: s.dataset,
  layers: s.layers,
  steps: s.steps,
});

function Scatter({
  points,
  size,
  range,
  color,
  label,
}: {
  points: number[];
  size: number;
  range: number;
  color: string;
  label: string;
}) {
  const at = (v: number) => ((v + range) / (2 * range)) * size;
  return (
    <svg
      viewBox={`0 0 ${size} ${size}`}
      className="w-full rounded-lg border border-line bg-surface"
      style={{ maxWidth: size }}
      role="img"
      aria-label={label}
    >
      {Array.from({ length: points.length / 2 }, (_, i) => {
        const x = points[2 * i] as number;
        const y = points[2 * i + 1] as number;
        if (Math.abs(x) > range || Math.abs(y) > range) return null;
        return <circle key={i} cx={at(x)} cy={size - at(y)} r={1.8} fill={color} opacity={0.75} />;
      })}
    </svg>
  );
}

function Results({ run }: { run: FlowLabRun }) {
  const palette = usePalette();
  // the heat map draws row 0 at the top: flip so that y points up
  const flipped = useMemo(() => {
    const out = new Float64Array(run.density.length);
    for (let j = 0; j < GRID; j++)
      for (let i = 0; i < GRID; i++)
        out[(GRID - 1 - j) * GRID + i] = run.density[j * GRID + i] as number;
    return out;
  }, [run.density]);
  const peak = Math.max(...run.density, 1e-9);
  const picks = useMemo(() => {
    const last = run.stages.length - 1;
    const want = [0, 0.25, 0.5, 0.75, 1].map((f) => Math.round(f * last));
    return [...new Set(want)];
  }, [run.stages.length]);

  const metrics: Metric[] = [
    {
      label: 'Cost per point, flow',
      value: fix(run.nll, 2),
      testId: 'nll',
      hint: 'negative log-likelihood, nats',
    },
    { label: 'On new points', value: fix(run.heldOutNll, 2) },
    { label: 'Best single Gaussian', value: fix(run.gaussianNll, 2) },
    { label: 'Largest inversion error', value: run.invertError.toExponential(0) },
  ];
  return (
    <>
      <Panel id="lab-density" className="scroll-mt-24" title="Density and samples" eyebrow="01">
        <div className="space-y-6">
          <MetricsStrip metrics={metrics} className="lg:grid-cols-4" />
          <div className="grid gap-6 sm:grid-cols-2">
            <figure className="space-y-2">
              <Heat
                values={flipped}
                cols={GRID}
                scale={peak}
                cellW={6}
                cellH={6}
                ariaLabel="Density the flow assigns to each place"
              />
              <figcaption className="text-xs text-muted">Density the flow assigns</figcaption>
            </figure>
            <figure className="space-y-2">
              <Scatter
                points={run.samples}
                size={336}
                range={HALF}
                color={palette.css.brand}
                label="Points sampled from the flow"
              />
              <figcaption className="text-xs text-muted">
                400 points sampled by pushing Gaussian noise backwards through the flow
              </figcaption>
            </figure>
          </div>
        </div>
      </Panel>
      <Panel id="lab-stages" className="scroll-mt-24" title="From data to Gaussian" eyebrow="02">
        <div className="space-y-4">
          <div className="flex flex-wrap gap-3">
            {picks.map((k) => (
              <figure key={k} className="space-y-1">
                <Scatter
                  points={run.stages[k] as number[]}
                  size={150}
                  range={k === run.stages.length - 1 ? 3.5 : HALF}
                  color={k === run.stages.length - 1 ? palette.css.pos : palette.css.muted}
                  label={k === 0 ? 'The data' : `After ${k} layers`}
                />
                <figcaption className="text-center font-mono text-[11px] text-muted">
                  {k === 0
                    ? 'data'
                    : k === run.stages.length - 1
                      ? `after ${k}: latent`
                      : `after ${k}`}
                </figcaption>
              </figure>
            ))}
          </div>
          <p className="text-sm text-muted">
            The same 300 data points after 0, some, and all of the layers. The last picture is the
            latent space, where the flow has tried to make them look like noise from a standard
            Gaussian.
          </p>
        </div>
      </Panel>
      <Panel id="lab-curve" className="scroll-mt-24" title="Training" eyebrow="03">
        <MiniChart
          title="Cost per training point (lower is better)"
          x={run.history.map((h) => h.step)}
          xLabel="training step"
          series={[
            { name: 'flow', color: palette.css.brand, values: run.history.map((h) => h.nll) },
            {
              name: 'best Gaussian',
              color: palette.css.muted,
              values: run.history.map(() => run.gaussianNll),
              dashed: true,
            },
          ]}
          height={220}
        />
      </Panel>
    </>
  );
}

export default function FlowsPage() {
  const [params, setParams] = useState<ParamValues>(() => fromScenario(DEFAULT_FLOW_LAB));
  const [activeId, setActiveId] = useState<string | null>(null);
  const scenario = useMemo(() => toScenario(params), [params]);
  const { result: run, error } = useCompute('flows', scenario);
  const setParam = (key: string, value: ParamValue) => setParams((p) => ({ ...p, [key]: value }));

  const apply = (exp: FlowExperiment) => {
    setParams(fromScenario({ ...DEFAULT_FLOW_LAB, ...exp.scenario }));
    setActiveId(exp.id);
  };
  const stillActive = (() => {
    const exp = flowExperiments.find((e) => e.id === activeId);
    return (
      !!exp &&
      JSON.stringify(fromScenario({ ...DEFAULT_FLOW_LAB, ...exp.scenario })) ===
        JSON.stringify(params)
    );
  })();

  const stages: [string, string][] = [
    ['lab-density', 'Density and samples'],
    ['lab-stages', 'From data to Gaussian'],
    ['lab-curve', 'Training'],
    ['lab-how', 'How it works'],
    ['lab-controls', 'Settings'],
  ];

  return (
    <div className="mx-auto max-w-6xl px-4 pt-8 pb-20 sm:px-6 lg:px-8">
      <LabHeader
        eyebrow="Generative models"
        title="Normalising flows"
        accent="exact likelihood by bending space"
        blurb="A flow turns a simple distribution into a complicated one through a chain of invertible steps. Because every step can be undone and its stretching measured, the density of any point is known exactly, and new points come from running the chain backwards. Stack layers and watch two moons straighten into a Gaussian."
        badges={
          <>
            <Badge>{FLOW_DATASETS[scenario.dataset]}</Badge>
            <Badge>{scenario.layers} coupling layers</Badge>
          </>
        }
      />
      <StageNav stages={stages} />
      <ExperimentBar
        lessons={flowExperiments}
        activeId={stillActive ? activeId : null}
        onApply={apply}
      />

      <div className="flex flex-col gap-5">
        {run ? (
          <Results run={run} />
        ) : (
          <Panel id="lab-density" className="scroll-mt-24" title="Density and samples" eyebrow="01">
            <p className="text-sm text-muted" role="status">
              {error ? `Something went wrong: ${error}` : 'Training the flow…'}
            </p>
          </Panel>
        )}
        <Panel id="lab-how" className="scroll-mt-24" title="How it works" eyebrow="04">
          <div className="space-y-3 text-sm text-muted">
            <p>
              Each coupling layer leaves one of the two coordinates alone and rescales and shifts
              the other by amounts computed from the first by a small network. Undoing it needs only
              the first coordinate, which is unchanged, so the layer is exactly invertible; and
              because the Jacobian is triangular its determinant is just the sum of the log-scales.
              Layers alternate which coordinate they leave alone.
            </p>
            <p>
              The density of a point is the Gaussian density of where it lands, times the product of
              the determinants. Training maximises that on the data. The tests check the inverse,
              the log-determinant against a numerical Jacobian, the gradients against finite
              differences, and that the density integrates to one.
            </p>
          </div>
        </Panel>
        <Panel id="lab-controls" className="scroll-mt-24" title="Settings" eyebrow="05">
          <ParamPanel specs={flowParams} values={params} onChange={setParam} columns={2} />
        </Panel>
      </div>
    </div>
  );
}
