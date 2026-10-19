import { HEURISTICS } from '@sim/core';
import { useMemo, useState } from 'react';
import { Badge, Panel, Slider } from '@/design-system';
import { usePalette } from '@/shared/canvas/colors';
import { ExperimentBar } from '@/shared/components/ExperimentBar';
import { MiniChart } from '@/shared/components/MiniChart';
import {
  LabHeader,
  MetricsStrip,
  ParamPanel,
  StageNav,
  fix,
  pct,
  type Metric,
  type ParamValue,
  type ParamValues,
} from '@/shared/lab-kit';
import {
  DEFAULT_HEURISTICS,
  METHOD_LABELS,
  METHOD_NOTES,
  RACE_REPEATS,
  runHeuristics,
  type HeuristicsRun,
  type HeuristicsScenario,
} from './heuristicsCore';
import {
  heuristicsExperiments,
  heuristicsParams,
  type HeuristicsExperiment,
} from './heuristicsSpec';

const toScenario = (p: ParamValues): HeuristicsScenario => ({
  ...DEFAULT_HEURISTICS,
  objective: p.objective as HeuristicsScenario['objective'],
  method: p.method as HeuristicsScenario['method'],
  population: Number(p.population),
  generations: Number(p.generations),
});

const fromScenario = (s: HeuristicsScenario): ParamValues => ({
  objective: s.objective,
  method: s.method,
  population: s.population,
  generations: s.generations,
});

const G = 40;

function Terrain({ run }: { run: HeuristicsRun }) {
  const palette = usePalette();
  const last = run.history.length - 1;
  const [shownRaw, setShown] = useState<number | null>(null);
  const shown = Math.min(shownRaw ?? last, last);
  const w = 560;
  const h = 460;
  const { x0, x1, y0, y1 } = run.objective.bounds;
  const sx = (v: number) => ((v - x0) / (x1 - x0)) * w;
  const sy = (v: number) => h - ((v - y0) / (y1 - y0)) * h;
  const values = useMemo(() => {
    const out: number[] = [];
    for (let r = 0; r < G; r++)
      for (let c = 0; c < G; c++)
        out.push(
          run.objective.f([x0 + ((c + 0.5) / G) * (x1 - x0), y1 - ((r + 0.5) / G) * (y1 - y0)]),
        );
    return out;
  }, [run.objective, x0, x1, y0, y1]);
  const lo = Math.min(...values);
  const hi = Math.max(...values);
  const gen = run.history[shown]!;
  const dist = gen.distribution;
  // 2-sigma ellipse of the CMA-ES search distribution
  let ellipse = '';
  if (dist) {
    const a = dist.cov[0][0];
    const b = dist.cov[0][1];
    const d = dist.cov[1][1];
    const tr = a + d;
    const det = a * d - b * b;
    const disc = Math.sqrt(Math.max((tr * tr) / 4 - det, 0));
    const l1 = Math.max(tr / 2 + disc, 1e-30);
    const l2 = Math.max(tr / 2 - disc, 1e-30);
    const th = 0.5 * Math.atan2(2 * b, a - d);
    const pts: string[] = [];
    for (let i = 0; i <= 40; i++) {
      const t = (i / 40) * 2 * Math.PI;
      const ex = 2 * dist.sigma * Math.sqrt(l1) * Math.cos(t);
      const ey = 2 * dist.sigma * Math.sqrt(l2) * Math.sin(t);
      const x = dist.mean[0] + ex * Math.cos(th) - ey * Math.sin(th);
      const y = dist.mean[1] + ex * Math.sin(th) + ey * Math.cos(th);
      pts.push(`${i ? 'L' : 'M'}${sx(x).toFixed(1)},${sy(y).toFixed(1)}`);
    }
    ellipse = pts.join(' ');
  }
  const trail =
    run.scenario.method === 'annealing'
      ? run.history.slice(0, shown + 1).flatMap((g) => g.points)
      : [];
  return (
    <div className="grid gap-8 lg:grid-cols-[minmax(0,1fr)_17rem]">
      <div className="space-y-3">
        <svg
          viewBox={`0 0 ${w} ${h}`}
          className="w-full rounded-xl border border-line"
          role="img"
          aria-label="The terrain as a heat map with the points evaluated in the chosen generation"
        >
          {values.map((v, i) => (
            <rect
              key={i}
              x={(i % G) * (w / G)}
              y={Math.floor(i / G) * (h / G)}
              width={w / G + 0.5}
              height={h / G + 0.5}
              fill={palette.css.brand}
              opacity={0.04 + 0.7 * (1 - (v - lo) / (hi - lo || 1))}
            />
          ))}
          {trail.length > 1 && (
            <path
              d={trail
                .map((p, i) => `${i ? 'L' : 'M'}${sx(p[0]).toFixed(1)},${sy(p[1]).toFixed(1)}`)
                .join(' ')}
              fill="none"
              stroke={palette.css.fg}
              strokeWidth={1}
              opacity={0.35}
            />
          )}
          {ellipse && (
            <path
              d={ellipse}
              fill={palette.css.neg}
              fillOpacity={0.1}
              stroke={palette.css.neg}
              strokeWidth={1.4}
            />
          )}
          {gen.points.map((p, i) => (
            <circle
              key={i}
              cx={sx(p[0])}
              cy={sy(p[1])}
              r={3.4}
              fill={palette.css.fg}
              opacity={0.75}
            />
          ))}
          <circle
            cx={sx(run.objective.optimum.x[0])}
            cy={sy(run.objective.optimum.x[1])}
            r={8}
            fill="none"
            stroke={palette.css.pos}
            strokeWidth={2.4}
          />
          <rect
            x={sx(gen.best.x[0]) - 5}
            y={sy(gen.best.x[1]) - 5}
            width={10}
            height={10}
            fill="none"
            stroke={palette.css.neg}
            strokeWidth={2.2}
            transform={`rotate(45 ${sx(gen.best.x[0])} ${sy(gen.best.x[1])})`}
          />
        </svg>
        <Slider label="Generation" min={0} max={last} value={shown} onValueChange={setShown} />
      </div>
      <div className="space-y-3 text-sm text-muted" data-testid="terrain-note">
        <p>
          Darker is lower. Dots are the points the method evaluated in this generation; the{' '}
          <span className="text-pos">green ring</span> is the true global minimum and the{' '}
          <span className="text-neg">red diamond</span> the best point found so far.
          {dist && " The red ellipse is CMA-ES's search distribution (2 standard deviations)."}
        </p>
        <p>
          After <strong className="text-fg">{gen.evaluations}</strong> evaluations the best value is{' '}
          <strong className="text-fg">
            {gen.best.f < 0.001 ? gen.best.f.toExponential(1) : fix(gen.best.f, 3)}
          </strong>
          {gen.temperature !== undefined && (
            <>
              ; the temperature is{' '}
              <strong className="text-fg">
                {gen.temperature < 0.01
                  ? gen.temperature.toExponential(1)
                  : fix(gen.temperature, 2)}
              </strong>
            </>
          )}
          .
        </p>
      </div>
    </div>
  );
}

export default function MetaheuristicsPage() {
  const palette = usePalette();
  const [params, setParams] = useState<ParamValues>(() => fromScenario(DEFAULT_HEURISTICS));
  const [activeId, setActiveId] = useState<string | null>(null);
  const scenario = useMemo(() => toScenario(params), [params]);
  const run = useMemo(() => runHeuristics(scenario), [scenario]);
  const setParam = (key: string, value: ParamValue) => setParams((p) => ({ ...p, [key]: value }));

  const apply = (exp: HeuristicsExperiment) => {
    setParams(fromScenario({ ...DEFAULT_HEURISTICS, ...exp.scenario }));
    setActiveId(exp.id);
  };
  const stillActive = (() => {
    const exp = heuristicsExperiments.find((e) => e.id === activeId);
    return (
      !!exp &&
      JSON.stringify(fromScenario({ ...DEFAULT_HEURISTICS, ...exp.scenario })) ===
        JSON.stringify(params)
    );
  })();

  const mine = run.race.find((r) => r.method === scenario.method)!;
  const metrics: Metric[] = [
    {
      label: 'Best value in this run',
      value:
        run.final.best.f < 0.001 ? run.final.best.f.toExponential(1) : fix(run.final.best.f, 3),
      testId: 'best',
    },
    { label: 'Evaluations', value: String(run.final.evaluations) },
    {
      label: `Success over ${RACE_REPEATS} runs`,
      value: pct(mine.success),
      hint: `share of runs ending within 0.1 of the global minimum, for ${METHOD_LABELS[scenario.method]}`,
    },
  ];

  const curve = run.history.map((g) => Math.max(g.best.f - run.objective.optimum.f, 1e-12));
  const stages: [string, string][] = [
    ['lab-terrain', 'The search'],
    ['lab-race', 'Which method wins?'],
    ['lab-curve', 'Progress'],
    ['lab-how', 'How it works'],
    ['lab-controls', 'Settings'],
  ];

  return (
    <div className="mx-auto max-w-6xl px-4 pt-8 pb-20 sm:px-6 lg:px-8">
      <LabHeader
        eyebrow="Evaluation & optimization"
        title="Metaheuristics"
        accent="search without gradients"
        blurb="Genetic algorithms, simulated annealing, particle swarms and CMA-ES look for a global minimum using only function values. Watch a population explore, then race all four across sixteen random starts to see which one is reliable on which terrain."
        badges={
          <>
            <Badge>{run.objective.label}</Badge>
            <Badge>{METHOD_LABELS[scenario.method]}</Badge>
          </>
        }
      />
      <StageNav stages={stages} />
      <ExperimentBar
        lessons={heuristicsExperiments}
        activeId={stillActive ? activeId : null}
        onApply={apply}
      />

      <div className="flex flex-col gap-5">
        <Panel id="lab-terrain" className="scroll-mt-24" title="The search" eyebrow="01">
          <div className="space-y-6">
            <p className="text-[0.8rem] text-muted">
              {run.objective.note}{' '}
              <span className="text-faint">{METHOD_NOTES[scenario.method]}</span>
            </p>
            <MetricsStrip metrics={metrics} className="lg:grid-cols-3" />
            <Terrain
              key={`${scenario.objective}-${scenario.method}-${scenario.population}-${scenario.generations}`}
              run={run}
            />
          </div>
        </Panel>
        <Panel id="lab-race" className="scroll-mt-24" title="Which method wins?" eyebrow="02">
          <div className="space-y-4">
            {run.race.map((r) => (
              <div
                key={r.method}
                className="grid grid-cols-[9.5rem_minmax(0,1fr)_7.5rem] items-center gap-3 text-[0.82rem]"
              >
                <span className={r.method === scenario.method ? 'text-fg' : 'text-muted'}>
                  {METHOD_LABELS[r.method]}
                </span>
                <div className="h-3 rounded-full bg-raised">
                  <div
                    className="h-3 rounded-full"
                    style={{
                      width: `${r.success * 100}%`,
                      background:
                        palette.css.classes[
                          HEURISTICS.indexOf(r.method) % palette.css.classes.length
                        ],
                      opacity: 0.85,
                    }}
                  />
                </div>
                <span
                  className="text-right font-mono text-faint"
                  data-testid={`success-${r.method}`}
                >
                  {pct(r.success)} ·{' '}
                  {r.medianBest < 0.001 ? r.medianBest.toExponential(0) : fix(r.medianBest, 2)}
                </span>
              </div>
            ))}
            <p className="text-sm text-muted">
              Bars: share of the {RACE_REPEATS} random restarts that ended at the global minimum;
              the number after is the median best value. One run can be lucky, so a heuristic is
              judged by its success rate for a given budget of evaluations.
            </p>
          </div>
        </Panel>
        <Panel id="lab-curve" className="scroll-mt-24" title="Progress" eyebrow="03">
          <div className="grid gap-8 lg:grid-cols-[minmax(0,1fr)_20rem]">
            <MiniChart
              title="Gap to the global minimum of the best point so far (log scale)"
              x={run.history.map((g) => g.evaluations)}
              xLabel="evaluations"
              series={[
                { name: METHOD_LABELS[scenario.method], color: 'var(--brand)', values: curve },
              ]}
              log
              height={200}
            />
            <p className="text-sm text-muted">
              Flat stretches are plateaus where the method is stuck in a pit or still exploring;
              steps down are escapes or improvements. Since only the best point so far is kept, the
              curve can never rise.
            </p>
          </div>
        </Panel>
        <Panel id="lab-how" className="scroll-mt-24" title="How it works" eyebrow="04">
          <div className="space-y-3 text-sm text-muted">
            <p>
              <strong className="text-fg">Genetic algorithm</strong>: pick parents by tournament,
              blend their coordinates (with some overshoot) and mutate, keeping the best individual.{' '}
              <strong className="text-fg">Simulated annealing</strong>: propose a random step;
              always accept improvements and accept a worse point with probability exp(-loss/T),
              cooling T geometrically so that early on it explores and later it descends.{' '}
              <strong className="text-fg">Particle swarm</strong>: velocity = inertia + a pull to
              the particle&apos;s own best + a pull to the swarm&apos;s best.
            </p>
            <p>
              <strong className="text-fg">CMA-ES</strong> keeps a Gaussian over the search space: it
              moves the mean to the weighted average of the best samples, adapts the covariance so
              that successful steps become more likely (so the ellipse aligns with a valley) and
              controls the overall step size by comparing the path it has travelled with what random
              steps would give. It is the strongest general derivative-free method on smooth, badly
              scaled problems and weaker on terrains full of small pits unless its population is
              large. Tests check the formulas&apos; consequences: the cooling ratio, monotone best
              values, and that CMA-ES learns the orientation of a tilted valley.
            </p>
          </div>
        </Panel>
        <Panel id="lab-controls" className="scroll-mt-24" title="Settings" eyebrow="05">
          <ParamPanel specs={heuristicsParams} values={params} onChange={setParam} columns={3} />
        </Panel>
      </div>
    </div>
  );
}
