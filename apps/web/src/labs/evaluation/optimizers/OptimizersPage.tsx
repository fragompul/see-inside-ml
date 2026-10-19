import { DESCENT_METHODS, distanceToMinimum, type DescentMethod } from '@sim/core';
import { useMemo, useState, type MouseEvent } from 'react';
import { Badge, Panel, Slider, Switch } from '@/design-system';
import { usePalette } from '@/shared/canvas/colors';
import { ExperimentBar } from '@/shared/components/ExperimentBar';
import { MiniChart } from '@/shared/components/MiniChart';
import {
  LabHeader,
  MetricsStrip,
  ParamPanel,
  StageNav,
  contourSegments,
  fix,
  type Metric,
  type ParamValue,
  type ParamValues,
} from '@/shared/lab-kit';
import {
  DEFAULT_OPTIMIZERS,
  METHOD_LABELS,
  METHOD_NOTES,
  endOf,
  runOptimizers,
  type OptimizerRun,
  type OptimizerScenario,
} from './optimizersCore';
import { optimizerExperiments, optimizerParams, type OptimizerExperiment } from './optimizersSpec';

const toScenario = (
  p: ParamValues,
  methods: DescentMethod[],
  start: [number, number] | null,
): OptimizerScenario => ({
  ...DEFAULT_OPTIMIZERS,
  fn: p.fn as OptimizerScenario['fn'],
  learningRate: Number(p.learningRate),
  steps: Number(p.steps),
  momentum: Number(p.momentum),
  methods,
  start,
});

const fromScenario = (s: OptimizerScenario): ParamValues => ({
  fn: s.fn,
  learningRate: s.learningRate,
  steps: s.steps,
  momentum: s.momentum,
});

const N = 90;

function Landscape({
  run,
  shown,
  onPick,
}: {
  run: OptimizerRun;
  shown: number;
  onPick: (x: number, y: number) => void;
}) {
  const palette = usePalette();
  const w = 620;
  const h = 460;
  const { x0, x1, y0, y1 } = run.fn.bounds;
  const sx = (v: number) => ((v - x0) / (x1 - x0)) * w;
  const sy = (v: number) => h - ((v - y0) / (y1 - y0)) * h;
  const contours = useMemo(() => {
    const vals = new Float64Array(N * N);
    for (let r = 0; r < N; r++)
      for (let c = 0; c < N; c++)
        vals[r * N + c] = run.fn.f([
          x0 + ((c + 0.5) / N) * (x1 - x0),
          y1 - ((r + 0.5) / N) * (y1 - y0),
        ]);
    const sorted = Array.from(vals).sort((a, b) => a - b);
    const q = (p: number) => sorted[Math.floor(p * (sorted.length - 1))] as number;
    const levels = [0.02, 0.05, 0.1, 0.2, 0.35, 0.55, 0.8].map(q);
    return levels.map((l) => contourSegments(vals, N, l));
  }, [run.fn, x0, x1, y0, y1]);
  const colors = palette.css.classes;
  const click = (e: MouseEvent<SVGSVGElement>) => {
    const box = e.currentTarget.getBoundingClientRect();
    const px = ((e.clientX - box.left) / box.width) * w;
    const py = ((e.clientY - box.top) / box.height) * h;
    onPick(x0 + (px / w) * (x1 - x0), y1 - (py / h) * (y1 - y0));
  };
  return (
    <svg
      viewBox={`0 0 ${w} ${h}`}
      className="w-full cursor-crosshair rounded-xl border border-line"
      role="img"
      aria-label="Contours of the function with the path of every optimiser; click to move the starting point"
      onClick={click}
    >
      {contours.map((segs, li) =>
        segs.map(([ax, ay, bx, by], i) => (
          <line
            key={`${li}-${i}`}
            x1={(ax / N) * w}
            y1={(ay / N) * h}
            x2={(bx / N) * w}
            y2={(by / N) * h}
            stroke={palette.css.fg}
            strokeWidth={1}
            opacity={0.2 + 0.03 * li}
          />
        )),
      )}
      {run.fn.minima.map((m, i) => (
        <circle
          key={i}
          cx={sx(m[0])}
          cy={sy(m[1])}
          r={7}
          fill="none"
          stroke={palette.css.pos}
          strokeWidth={2.2}
        />
      ))}
      {run.paths.map((p) => {
        const idx = DESCENT_METHODS.indexOf(p.method);
        const pts = p.points.slice(0, shown + 1);
        const d = pts
          .map(
            (q, i) =>
              `${i ? 'L' : 'M'}${sx(Math.max(x0 - 1, Math.min(x1 + 1, q[0]))).toFixed(1)},${sy(Math.max(y0 - 1, Math.min(y1 + 1, q[1]))).toFixed(1)}`,
          )
          .join(' ');
        const last = pts[pts.length - 1] as [number, number];
        return (
          <g key={p.method}>
            <path
              d={d}
              fill="none"
              stroke={colors[idx % colors.length]}
              strokeWidth={2}
              opacity={0.9}
            />
            <circle
              cx={sx(Math.max(x0 - 1, Math.min(x1 + 1, last[0])))}
              cy={sy(Math.max(y0 - 1, Math.min(y1 + 1, last[1])))}
              r={4}
              fill={colors[idx % colors.length]}
            />
          </g>
        );
      })}
      <circle
        cx={sx(run.start[0])}
        cy={sy(run.start[1])}
        r={6}
        fill={palette.css.fg}
        stroke={palette.css.surface}
        strokeWidth={2}
      />
    </svg>
  );
}

export default function OptimizersPage() {
  const palette = usePalette();
  const [params, setParams] = useState<ParamValues>(() => fromScenario(DEFAULT_OPTIMIZERS));
  const [methods, setMethods] = useState<DescentMethod[]>([...DESCENT_METHODS]);
  const [start, setStart] = useState<[number, number] | null>(null);
  const [activeId, setActiveId] = useState<string | null>(null);
  const [shownRaw, setShown] = useState<number | null>(null);
  const scenario = useMemo(() => toScenario(params, methods, start), [params, methods, start]);
  const run = useMemo(() => runOptimizers(scenario), [scenario]);
  const shown = Math.min(shownRaw ?? scenario.steps, scenario.steps);
  const setParam = (key: string, value: ParamValue) => setParams((p) => ({ ...p, [key]: value }));

  const apply = (exp: OptimizerExperiment) => {
    const s = { ...DEFAULT_OPTIMIZERS, ...exp.scenario };
    setParams(fromScenario(s));
    setStart(s.start);
    setMethods([...DESCENT_METHODS]);
    setShown(null);
    setActiveId(exp.id);
  };
  const stillActive = (() => {
    const exp = optimizerExperiments.find((e) => e.id === activeId);
    if (!exp || methods.length !== DESCENT_METHODS.length) return false;
    const s = { ...DEFAULT_OPTIMIZERS, ...exp.scenario };
    return (
      JSON.stringify(fromScenario(s)) === JSON.stringify(params) &&
      JSON.stringify(s.start) === JSON.stringify(start)
    );
  })();

  const toggle = (m: DescentMethod) =>
    setMethods((cur) =>
      cur.includes(m)
        ? cur.filter((x) => x !== m)
        : DESCENT_METHODS.filter((x) => x === m || cur.includes(x)),
    );

  const best = run.paths.reduce<{ method: DescentMethod; v: number } | null>((b, p) => {
    const v = p.values[p.values.length - 1] as number;
    return !p.diverged && Number.isFinite(v) && (!b || v < b.v) ? { method: p.method, v } : b;
  }, null);
  const metrics: Metric[] = [
    {
      label: 'Lowest value reached',
      value: best
        ? `${METHOD_LABELS[best.method]} · ${best.v < 0.001 ? best.v.toExponential(1) : fix(best.v, 3)}`
        : '–',
      testId: 'best',
    },
    {
      label: 'Converged',
      value: `${run.paths.filter((p) => p.converged !== null).length} of ${run.paths.length}`,
      hint: 'gradient norm below 0.0001',
    },
    {
      label: 'Diverged',
      value: String(run.paths.filter((p) => p.diverged).length),
      tone: run.paths.some((p) => p.diverged) ? 'bad' : 'neutral',
    },
  ];

  const stages: [string, string][] = [
    ['lab-race', 'The race'],
    ['lab-table', 'Scoreboard'],
    ['lab-curves', 'Value per step'],
    ['lab-how', 'How it works'],
    ['lab-controls', 'Settings'],
  ];
  const lengthMax = Math.max(...run.paths.map((p) => p.values.length), 2);

  return (
    <div className="mx-auto max-w-6xl px-4 pt-8 pb-20 sm:px-6 lg:px-8">
      <LabHeader
        eyebrow="Evaluation & optimization"
        title="The optimiser race"
        accent="seven ways downhill"
        blurb="Gradient descent, momentum, Nesterov, RMSprop, Adam, Newton and L-BFGS start from the same point on the same terrain. Click the map to move the start, change the learning rate, and watch who gets stuck, who overshoots and who takes the shortcut."
        badges={
          <>
            <Badge>{run.fn.label}</Badge>
            <Badge>{scenario.steps} steps</Badge>
          </>
        }
      />
      <StageNav stages={stages} />
      <ExperimentBar
        lessons={optimizerExperiments}
        activeId={stillActive ? activeId : null}
        onApply={apply}
      />

      <div className="flex flex-col gap-5">
        <Panel id="lab-race" className="scroll-mt-24" title="The race" eyebrow="01">
          <div className="space-y-6">
            <p className="text-[0.8rem] text-muted">{run.fn.note}</p>
            <MetricsStrip metrics={metrics} className="lg:grid-cols-3" />
            <div className="grid gap-8 lg:grid-cols-[minmax(0,1fr)_15rem]">
              <div className="space-y-3">
                <Landscape run={run} shown={shown} onPick={(x, y) => setStart([x, y])} />
                <Slider
                  label="Steps shown"
                  min={0}
                  max={scenario.steps}
                  value={shown}
                  onValueChange={setShown}
                />
                <p className="text-[0.74rem] text-faint">
                  Lines are contours of the function (closer together = steeper). Green rings mark
                  the minima; the white dot is the start. Click anywhere to restart from there.
                </p>
              </div>
              <div className="space-y-3">
                <p className="eyebrow">Optimisers</p>
                {DESCENT_METHODS.map((m, i) => (
                  <div key={m} className="space-y-0.5">
                    <div className="flex items-center gap-2">
                      <span
                        className="inline-block size-2.5 rounded-sm"
                        style={{ background: palette.css.classes[i % palette.css.classes.length] }}
                      />
                      <div className="flex-1">
                        <Switch
                          label={METHOD_LABELS[m]}
                          checked={methods.includes(m)}
                          onCheckedChange={() => toggle(m)}
                        />
                      </div>
                    </div>
                    <p className="pl-[1.15rem] text-[0.68rem] text-faint">{METHOD_NOTES[m]}</p>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </Panel>
        <Panel id="lab-table" className="scroll-mt-24" title="Scoreboard" eyebrow="02">
          <div className="overflow-x-auto">
            <table className="w-full min-w-[34rem] text-left font-mono text-[0.78rem] tabular-nums">
              <thead className="text-muted">
                <tr>
                  <th className="pr-4 pb-2 font-sans font-medium">optimiser</th>
                  <th className="pr-4 pb-2 font-medium">final value</th>
                  <th className="pr-4 pb-2 font-medium">distance to nearest minimum</th>
                  <th className="pr-4 pb-2 font-medium">steps to converge</th>
                </tr>
              </thead>
              <tbody>
                {run.paths.map((p) => {
                  const v = p.values[p.values.length - 1] as number;
                  return (
                    <tr key={p.method} className="border-t border-line">
                      <td className="py-1.5 pr-4 font-sans">{METHOD_LABELS[p.method]}</td>
                      <td className="pr-4">
                        {p.diverged ? (
                          <span className="text-neg">diverged</span>
                        ) : Math.abs(v) < 0.001 && v !== 0 ? (
                          v.toExponential(1)
                        ) : (
                          fix(v, 3)
                        )}
                      </td>
                      <td className="pr-4">
                        {p.diverged ? '–' : fix(distanceToMinimum(run.fn, endOf(p)), 3)}
                      </td>
                      <td className="pr-4">{p.converged === null ? '–' : p.converged}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </Panel>
        <Panel id="lab-curves" className="scroll-mt-24" title="Value per step" eyebrow="03">
          <div className="grid gap-8 lg:grid-cols-[minmax(0,1fr)_20rem]">
            <MiniChart
              title="Function value after each step (shifted so the lowest is 1e-8; log scale)"
              x={Array.from({ length: lengthMax }, (_, i) => i)}
              xLabel="step"
              series={run.paths.map((p) => {
                const floor = Math.min(
                  ...run.paths.flatMap((q) => q.values.filter(Number.isFinite)),
                );
                const vals = Array.from({ length: lengthMax }, (_, i) => {
                  const v = (p.values[Math.min(i, p.values.length - 1)] as number) - floor + 1e-8;
                  return Number.isFinite(v) ? Math.min(Math.max(v, 1e-8), 1e6) : 1e6;
                });
                return {
                  name: METHOD_LABELS[p.method],
                  color: palette.css.classes[
                    DESCENT_METHODS.indexOf(p.method) % palette.css.classes.length
                  ] as string,
                  values: vals,
                };
              })}
              log
              height={240}
            />
            <p className="text-sm text-muted">
              Methods that curve downward steeply are converging fast (Newton and L-BFGS in a
              handful of steps); a long straight line means a crawl. A curve that shoots upward has
              diverged.
            </p>
          </div>
        </Panel>
        <Panel id="lab-how" className="scroll-mt-24" title="How it works" eyebrow="04">
          <div className="space-y-3 text-sm text-muted">
            <p>
              <strong className="text-fg">First-order</strong> methods only use the gradient. Plain
              descent steps against it; <strong className="text-fg">momentum</strong> adds a running
              velocity; <strong className="text-fg">Nesterov</strong> evaluates the gradient at the
              point the velocity is about to reach; <strong className="text-fg">RMSprop</strong> and{' '}
              <strong className="text-fg">Adam</strong> rescale each coordinate by the recent size
              of its gradients, which makes them far less sensitive to the learning rate.
            </p>
            <p>
              <strong className="text-fg">Second-order</strong> methods use curvature. Newton solves
              the local quadratic exactly (here with the negative curvature of a saddle flipped so
              that it still descends); L-BFGS reconstructs a curvature estimate from its last ten
              steps and finds a good step by backtracking. They need far fewer steps on smooth
              problems but each step costs more, and neural networks have too many parameters for a
              Hessian, which is why Adam dominates deep learning.
            </p>
            <p>
              Gradients and Hessians of every test function are checked against finite differences
              in the tests.
            </p>
          </div>
        </Panel>
        <Panel id="lab-controls" className="scroll-mt-24" title="Settings" eyebrow="05">
          <ParamPanel specs={optimizerParams} values={params} onChange={setParam} columns={3} />
        </Panel>
      </div>
    </div>
  );
}
