import { useMemo, useState } from 'react';
import { Badge, Panel } from '@/design-system';
import { usePalette } from '@/shared/canvas/colors';
import { useCompute } from '@/shared/compute/useCompute';
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
  DATASET_LABEL,
  DEFAULT_ODE,
  GRID,
  RANGE,
  type OdeResult,
  type OdeScenario,
} from './odeCore';
import { odeExperiments, odeParams, type OdeExperiment } from './odeSpec';

const toScenario = (p: ParamValues): OdeScenario => ({
  dataset: p.dataset as OdeScenario['dataset'],
  solver: p.solver as OdeScenario['solver'],
  steps: Number(p.steps),
});

const fromScenario = (s: OdeScenario): ParamValues => ({ ...s });

const SIZE = 340;
const at = (v: number) => ((v + RANGE) / (2 * RANGE)) * SIZE;
const HUE = [20, 215];

function Results({ run }: { run: OdeResult }) {
  const palette = usePalette();
  const s = run.scenario;
  const cell = SIZE / GRID;
  const metrics: Metric[] = [
    {
      label: 'Accuracy with its own solver',
      value: `${fix(run.own * 100, 0)}%`,
      testId: 'own',
      hint: `${s.solver === 'rk4' ? 'Runge-Kutta' : 'Euler'}, ${s.steps} step${s.steps > 1 ? 's' : ''}`,
    },
    {
      label: 'Same weights, accurate solver',
      value: `${fix(run.fine * 100, 0)}%`,
      hint: '128 Runge-Kutta steps',
    },
    {
      label: 'Final states differ by',
      value: fix(run.stateError, 2),
      hint: 'average distance between the two solvers',
    },
  ];
  const maxSpeed = Math.max(
    ...run.field.map((f) => Math.hypot(f[2] as number, f[3] as number)),
    1e-9,
  );
  return (
    <>
      <Panel id="lab-flow" className="scroll-mt-24" title="The flow" eyebrow="01">
        <div className="space-y-6">
          <MetricsStrip metrics={metrics} className="lg:grid-cols-3" />
          <figure className="space-y-2">
            <svg
              viewBox={`0 0 ${SIZE} ${SIZE}`}
              className="w-full max-w-[26rem] rounded-xl border border-line bg-surface"
              role="img"
              aria-label="Paths of points under the learned velocity field, over the predicted class"
            >
              {run.map.map((p, i) => (
                <rect
                  key={i}
                  x={(i % GRID) * cell}
                  y={Math.floor(i / GRID) * cell}
                  width={cell + 0.5}
                  height={cell + 0.5}
                  fill={`hsl(${HUE[p > 0.5 ? 1 : 0]} 60% 55%)`}
                  opacity={0.1 + 0.18 * Math.abs(p - 0.5) * 2}
                />
              ))}
              {run.field.map(([x, y, vx, vy], i) => {
                const scale = (cell * 1.6) / maxSpeed;
                return (
                  <line
                    key={i}
                    x1={at(x as number)}
                    y1={SIZE - at(y as number)}
                    x2={at(x as number) + (vx as number) * scale}
                    y2={SIZE - at(y as number) - (vy as number) * scale}
                    stroke={palette.css.muted}
                    strokeOpacity={0.55}
                    strokeWidth={1.5}
                  />
                );
              })}
              {run.paths.map((path, i) => (
                <g key={i}>
                  <polyline
                    points={path.points
                      .map((p) => `${at(p[0] as number)},${SIZE - at(p[1] as number)}`)
                      .join(' ')}
                    fill="none"
                    stroke={`hsl(${HUE[path.label]} 65% 45%)`}
                    strokeWidth={1.4}
                    strokeOpacity={0.8}
                  />
                  <circle
                    cx={at((path.points[0] as number[])[0] as number)}
                    cy={SIZE - at((path.points[0] as number[])[1] as number)}
                    r={2.4}
                    fill={`hsl(${HUE[path.label]} 65% 45%)`}
                  />
                </g>
              ))}
            </svg>
            <figcaption className="text-sm text-muted">
              The tint is the class the model predicts (two colours); grey strokes are the velocity
              field; each coloured line follows one training point from its start (the dot) to its
              final position after unit time.
            </figcaption>
          </figure>
        </div>
      </Panel>
      <Panel id="lab-training" className="scroll-mt-24" title="Training" eyebrow="02">
        <MiniChart
          title="Training loss"
          x={run.history.map((h) => h.epoch)}
          xLabel="epoch"
          series={[
            { name: 'loss', color: palette.css.brand, values: run.history.map((h) => h.loss) },
          ]}
          height={200}
        />
      </Panel>
    </>
  );
}

export default function OdePage() {
  const [params, setParams] = useState<ParamValues>(() => fromScenario(DEFAULT_ODE));
  const [activeId, setActiveId] = useState<string | null>(null);
  const scenario = useMemo(() => toScenario(params), [params]);
  const { result: run, error } = useCompute('neural-ode', scenario);
  const setParam = (key: string, value: ParamValue) => setParams((p) => ({ ...p, [key]: value }));

  const apply = (exp: OdeExperiment) => {
    setParams(fromScenario({ ...DEFAULT_ODE, ...exp.scenario }));
    setActiveId(exp.id);
  };
  const stillActive = (() => {
    const exp = odeExperiments.find((e) => e.id === activeId);
    return (
      !!exp &&
      JSON.stringify(fromScenario({ ...DEFAULT_ODE, ...exp.scenario })) === JSON.stringify(params)
    );
  })();

  const stages: [string, string][] = [
    ['lab-flow', 'The flow'],
    ['lab-training', 'Training'],
    ['lab-how', 'How it works'],
    ['lab-controls', 'Settings'],
  ];

  return (
    <div className="mx-auto max-w-6xl px-4 pt-8 pb-20 sm:px-6 lg:px-8">
      <LabHeader
        eyebrow="Modern topics"
        title="Neural ODEs"
        accent="a network with continuous depth"
        blurb="A residual network adds a small change to its state at every layer. Let the layers become infinitely thin and the state follows a differential equation whose velocity is a neural network. Train one, look at the flow it learns, and check whether it is really a flow or just a few layers in disguise."
        badges={
          <>
            <Badge>{DATASET_LABEL[scenario.dataset]}</Badge>
            <Badge>
              {scenario.solver === 'rk4' ? 'Runge-Kutta' : 'Euler'} · {scenario.steps} steps
            </Badge>
          </>
        }
      />
      <StageNav stages={stages} />
      <ExperimentBar
        lessons={odeExperiments}
        activeId={stillActive ? activeId : null}
        onApply={apply}
      />

      <div className="flex flex-col gap-5">
        {run ? (
          <Results run={run} />
        ) : (
          <Panel id="lab-flow" className="scroll-mt-24" title="The flow" eyebrow="01">
            <p className="text-sm text-muted" role="status">
              {error ? `Something went wrong: ${error}` : 'Training the neural ODE…'}
            </p>
          </Panel>
        )}
        <Panel id="lab-how" className="scroll-mt-24" title="How it works" eyebrow="03">
          <div className="space-y-3 text-sm text-muted">
            <p>
              The state of a point starts as its two coordinates. For unit time it moves with
              velocity <em>f(x)</em>, a small network, and a linear read-out of where it ends up
              gives the class. To follow the path a computer takes a fixed number of steps: Euler
              adds the velocity at the start of each step; Runge-Kutta 4 averages four velocity
              estimates inside the step and is far more accurate for the same number of steps.
            </p>
            <p>
              Training differentiates through those steps. That is why the solver matters: the
              network is optimised for the discretised path, and if the discretisation is crude,
              what it learns is a stack of residual layers that happens to work with that step size,
              not a flow. The tests check the gradients through both solvers, and that Euler
              converges at first order and Runge-Kutta at fourth order as the steps get finer.
            </p>
          </div>
        </Panel>
        <Panel id="lab-controls" className="scroll-mt-24" title="Settings" eyebrow="04">
          <ParamPanel specs={odeParams} values={params} onChange={setParam} columns={2} />
        </Panel>
      </div>
    </div>
  );
}
