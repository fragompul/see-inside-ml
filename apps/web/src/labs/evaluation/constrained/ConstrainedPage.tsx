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
  DEFAULT_CONSTRAINED,
  GRID,
  runConstrained,
  type ConstrainedScenario,
} from './constrainedCore';
import {
  METHOD_LABEL,
  constrainedExperiments,
  constrainedParams,
  type ConstrainedExperiment,
} from './constrainedSpec';

const toScenario = (p: ParamValues): ConstrainedScenario => ({
  problem: String(p.problem),
  method: p.method as ConstrainedScenario['method'],
  outer: Number(p.outer),
  weight: Number(p.weight),
  factor: Number(p.factor),
});

const fromScenario = (s: ConstrainedScenario): ParamValues => ({ ...s });

const SIZE = 340;
const sci = (v: number) => (Number.isFinite(v) ? v.toExponential(1) : '–');

export default function ConstrainedPage() {
  const palette = usePalette();
  const [params, setParams] = useState<ParamValues>(() => fromScenario(DEFAULT_CONSTRAINED));
  const [activeId, setActiveId] = useState<string | null>(null);
  const scenario = useMemo(() => toScenario(params), [params]);
  const run = useMemo(() => runConstrained(scenario), [scenario]);
  const setParam = (key: string, value: ParamValue) => setParams((q) => ({ ...q, [key]: value }));

  const apply = (exp: ConstrainedExperiment) => {
    setParams(fromScenario({ ...DEFAULT_CONSTRAINED, ...exp.scenario }));
    setActiveId(exp.id);
  };
  const stillActive = (() => {
    const exp = constrainedExperiments.find((e) => e.id === activeId);
    return (
      !!exp &&
      JSON.stringify(fromScenario({ ...DEFAULT_CONSTRAINED, ...exp.scenario })) ===
        JSON.stringify(params)
    );
  })();

  const { x0, x1, y0, y1 } = run.bounds;
  const at = (x: number, y: number): [number, number] => [
    ((x - x0) / (x1 - x0)) * SIZE,
    SIZE - ((y - y0) / (y1 - y0)) * SIZE,
  ];
  const cell = SIZE / GRID;
  const lo = Math.min(...run.surface);
  const hi = Math.max(...run.surface);
  const shade = (v: number) => {
    const t = Math.log(1 + v - lo) / Math.log(1 + hi - lo);
    return `color-mix(in srgb, ${palette.css.brand} ${Math.round((1 - t) * 38)}%, ${palette.css.surface})`;
  };
  const last = run.steps[run.steps.length - 1];
  const metrics: Metric[] = last
    ? [
        {
          label: 'Distance from the true solution',
          value: sci(run.error[run.error.length - 1] as number),
          testId: 'distance',
        },
        {
          label:
            scenario.method === 'barrier' ? 'Distance inside the boundary' : 'Largest violation',
          value: sci(Math.abs(last.violation)),
        },
        {
          label: 'Multipliers found',
          value: last.multipliers.map((m) => fix(m, 3)).join(', '),
          hint: `true: ${run.solution.multipliers.map((m) => fix(m, 3)).join(', ')}`,
        },
        { label: 'Optimality-condition residual', value: sci(run.kkt) },
      ]
    : [{ label: 'Distance from the true solution', value: '–', testId: 'distance' }];

  const stages: [string, string][] = [
    ['lab-picture', 'The problem'],
    ['lab-errors', 'Convergence'],
    ['lab-how', 'How it works'],
    ['lab-controls', 'Settings'],
  ];
  const sol = at(run.solution.x[0], run.solution.x[1]);
  const start = at(run.start[0], run.start[1]);

  return (
    <div className="mx-auto max-w-6xl px-4 pt-8 pb-20 sm:px-6 lg:px-8">
      <LabHeader
        eyebrow="Evaluation & optimisation"
        title="Constrained optimisation"
        accent="the best point you are allowed to pick"
        blurb="Most real optimisation problems come with rules: stay inside a region, meet a budget. Penalties, Lagrange multipliers and barriers each turn the rules into a sequence of ordinary minimisations, and each pays a different price. Follow them to the answer and check it against the optimality conditions."
        badges={
          <>
            <Badge>{run.problem.label}</Badge>
            <Badge>{METHOD_LABEL[scenario.method]}</Badge>
          </>
        }
      />
      <StageNav stages={stages} />
      <ExperimentBar
        lessons={constrainedExperiments}
        activeId={stillActive ? activeId : null}
        onApply={apply}
      />

      <div className="flex flex-col gap-5">
        <Panel id="lab-picture" className="scroll-mt-24" title="The problem" eyebrow="01">
          <div className="space-y-6">
            <MetricsStrip metrics={metrics} className="lg:grid-cols-4" />
            {run.steps.length === 0 && (
              <p className="text-sm text-neg" role="status">
                The logarithmic barrier needs inequality constraints only; this problem has an
                equality. Choose another method.
              </p>
            )}
            <figure className="space-y-2">
              <svg
                viewBox={`0 0 ${SIZE} ${SIZE}`}
                className="w-full max-w-[26rem] rounded-xl border border-line"
                role="img"
                aria-label="The objective as a shaded surface, the feasible region and the stages of the method"
              >
                {run.surface.map((v, i) => (
                  <rect
                    key={i}
                    x={(i % GRID) * cell}
                    y={Math.floor(i / GRID) * cell}
                    width={cell + 0.5}
                    height={cell + 0.5}
                    fill={shade(v)}
                  />
                ))}
                {run.feasible.map((f, i) =>
                  f ? (
                    <rect
                      key={`f${i}`}
                      x={(i % GRID) * cell}
                      y={Math.floor(i / GRID) * cell}
                      width={cell + 0.5}
                      height={cell + 0.5}
                      fill={palette.css.pos}
                      opacity={0.28}
                    />
                  ) : null,
                )}
                <polyline
                  points={[run.start, ...run.steps.map((s) => s.x)]
                    .map((p) => at(p[0], p[1]).join(','))
                    .join(' ')}
                  fill="none"
                  stroke={palette.css.fg}
                  strokeWidth={1.4}
                />
                {run.steps.map((s, i) => {
                  const [px, py] = at(s.x[0], s.x[1]);
                  return <circle key={i} cx={px} cy={py} r={3} fill={palette.css.fg} />;
                })}
                <circle
                  cx={start[0]}
                  cy={start[1]}
                  r={4}
                  fill="none"
                  stroke={palette.css.fg}
                  strokeWidth={1.5}
                />
                <path
                  d={`M${sol[0] - 6},${sol[1]} L${sol[0] + 6},${sol[1]} M${sol[0]},${sol[1] - 6} L${sol[0]},${sol[1] + 6}`}
                  stroke={palette.css.neg}
                  strokeWidth={2}
                />
              </svg>
              <figcaption className="text-sm text-muted">
                Darker means a higher objective. The green region is where every constraint (
                {run.problem.constraints.join('; ')}) holds. The ring is the start, the dots the
                solution after each stage, the cross the true answer. {run.problem.note}
              </figcaption>
            </figure>
          </div>
        </Panel>
        <Panel id="lab-errors" className="scroll-mt-24" title="Convergence" eyebrow="02">
          <MiniChart
            title="Distance from the true solution after each stage (log scale)"
            x={Array.from({ length: 8 }, (_, i) => i + 1)}
            xLabel="stage"
            log
            series={[
              ...(run.compare.penalty
                ? [
                    {
                      name: 'quadratic penalty (weight ×10 each stage)',
                      color: palette.css.neg,
                      values: run.compare.penalty.map((v) => Math.max(v, 1e-12)),
                    },
                  ]
                : []),
              ...(run.compare.augmented
                ? [
                    {
                      name: 'augmented Lagrangian (weight 10)',
                      color: palette.css.pos,
                      values: run.compare.augmented.map((v) => Math.max(v, 1e-12)),
                    },
                  ]
                : []),
              ...(run.compare.barrier
                ? [
                    {
                      name: 'logarithmic barrier (weight ÷10 each stage)',
                      color: palette.css.brand,
                      values: run.compare.barrier.map((v) => Math.max(v, 1e-12)),
                    },
                  ]
                : []),
            ]}
            height={240}
          />
        </Panel>
        <Panel id="lab-how" className="scroll-mt-24" title="How it works" eyebrow="03">
          <div className="space-y-3 text-sm text-muted">
            <p>
              At the best allowed point the gradient of the objective is cancelled by the gradients
              of the constraints that are active, each weighted by a multiplier (the
              Karush-Kuhn-Tucker conditions):{' '}
              <em>
                ∇f + Σ λ<sub>i</sub> ∇g<sub>i</sub> = 0
              </em>
              , with λ ≥ 0 for an inequality and λ = 0 for any constraint that is not touching. The
              multiplier says how much the objective would improve if that constraint were loosened.
            </p>
            <p>
              The <strong className="text-fg">penalty</strong> method minimises f plus μ/2 times the
              squared violation, with μ growing; the{' '}
              <strong className="text-fg">augmented Lagrangian</strong> adds multiplier terms and
              updates them after each stage; the <strong className="text-fg">barrier</strong> adds
              −t·log(−g) and lets t shrink. Each stage is solved with BFGS. The tests check the
              solutions and multipliers against closed forms, that the error of the penalty falls in
              proportion to its weight, and the optimality conditions at the result.
            </p>
          </div>
        </Panel>
        <Panel id="lab-controls" className="scroll-mt-24" title="Settings" eyebrow="04">
          <ParamPanel specs={constrainedParams} values={params} onChange={setParam} columns={2} />
        </Panel>
      </div>
    </div>
  );
}
