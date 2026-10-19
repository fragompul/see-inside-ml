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
  pct,
  type Metric,
  type ParamValue,
  type ParamValues,
} from '@/shared/lab-kit';
import {
  DEFAULT_PAGERANK,
  PAGE_NAMES,
  runPageRank,
  type PageRankRun,
  type PageRankScenario,
} from './pagerankCore';
import { pagerankExperiments, pagerankParams, type PageRankExperiment } from './pagerankSpec';

const toScenario = (p: ParamValues): PageRankScenario => ({
  ...DEFAULT_PAGERANK,
  web: p.web as PageRankScenario['web'],
  damping: Number(p.damping),
  iterations: Number(p.iterations),
  surfers: Number(p.surfers),
});

const fromScenario = (s: PageRankScenario): ParamValues => ({
  web: s.web,
  damping: s.damping,
  iterations: s.iterations,
  surfers: s.surfers,
});

function Graph({ run }: { run: PageRankRun }) {
  const palette = usePalette();
  const w = 560;
  const h = 420;
  const cx = w / 2;
  const cy = h / 2;
  const R = 160;
  const pos = PAGE_NAMES.map((_, i) => {
    const a = -Math.PI / 2 + (i / PAGE_NAMES.length) * 2 * Math.PI;
    return [cx + R * Math.cos(a), cy + R * Math.sin(a)] as const;
  });
  const radius = (rank: number) => 10 + 70 * Math.sqrt(rank) * 0.6;
  return (
    <svg
      viewBox={`0 0 ${w} ${h}`}
      className="w-full max-w-xl rounded-xl border border-line"
      role="img"
      aria-label="Eight web pages as circles sized by their rank, with arrows for links"
    >
      <defs>
        <marker
          id="pr-arrow"
          viewBox="0 0 8 8"
          refX="7"
          refY="4"
          markerWidth="7"
          markerHeight="7"
          orient="auto"
        >
          <path d="M0,0 L8,4 L0,8 z" fill={palette.css.muted} />
        </marker>
      </defs>
      {run.web.links.map((out, i) =>
        out.map((j) => {
          const [x1, y1] = pos[i]!;
          const [x2, y2] = pos[j]!;
          const dx = x2 - x1;
          const dy = y2 - y1;
          const len = Math.hypot(dx, dy) || 1;
          const r1 = radius(run.shown[i] as number);
          const r2 = radius(run.shown[j] as number) + 6;
          // curve slightly so that two-way links do not overlap
          const mx = (x1 + x2) / 2 - (dy / len) * 14;
          const my = (y1 + y2) / 2 + (dx / len) * 14;
          const sx = x1 + (dx / len) * r1;
          const sy = y1 + (dy / len) * r1;
          const ex = x2 - (dx / len) * r2;
          const ey = y2 - (dy / len) * r2;
          return (
            <path
              key={`${i}-${j}`}
              d={`M${sx.toFixed(1)},${sy.toFixed(1)} Q${mx.toFixed(1)},${my.toFixed(1)} ${ex.toFixed(1)},${ey.toFixed(1)}`}
              fill="none"
              stroke={palette.css.muted}
              strokeWidth={1.2}
              opacity={0.65}
              markerEnd="url(#pr-arrow)"
            />
          );
        }),
      )}
      {pos.map(([x, y], i) => (
        <g key={i}>
          <circle
            cx={x}
            cy={y}
            r={radius(run.shown[i] as number)}
            fill={palette.css.brand}
            opacity={0.75}
          />
          <text
            x={x}
            y={y + 4}
            textAnchor="middle"
            className="fill-brand-fg font-display text-[13px] font-semibold"
          >
            {PAGE_NAMES[i]}
          </text>
          <text
            x={x}
            y={y + radius(run.shown[i] as number) + 13}
            textAnchor="middle"
            className="fill-muted font-mono text-[10px]"
          >
            {pct(run.shown[i])}
          </text>
        </g>
      ))}
    </svg>
  );
}

function Bars({ run }: { run: PageRankRun }) {
  const palette = usePalette();
  const max = Math.max(...run.exact, ...run.surfed, ...run.shown);
  return (
    <div className="space-y-3">
      <div
        className="flex h-44 items-end gap-3"
        role="img"
        aria-label="Rank of each page: after the chosen iterations, exact, and as measured by a random surfer"
      >
        {PAGE_NAMES.map((name, i) => (
          <div key={name} className="flex flex-1 flex-col items-center gap-1">
            <div className="flex h-36 w-full items-end justify-center gap-0.5">
              {[
                [run.shown[i], palette.css.brand, 'after the shown steps'],
                [run.exact[i], palette.css.pos, 'exact'],
                [run.surfed[i], palette.css.neg, 'random surfer'],
              ].map(([v, color, label]) => (
                <div
                  key={label as string}
                  className="w-1/3 rounded-t"
                  style={{
                    height: `${Math.max(1, ((v as number) / max) * 100)}%`,
                    background: color as string,
                    opacity: 0.85,
                  }}
                  title={`${label as string}: ${pct(v as number)}`}
                />
              ))}
            </div>
            <span className="font-mono text-xs text-muted">{name}</span>
          </div>
        ))}
      </div>
      <p className="flex flex-wrap gap-x-5 text-[0.78rem] text-muted">
        <span>
          <span
            className="mr-1 inline-block size-2 rounded-sm"
            style={{ background: palette.css.brand }}
          />
          after {run.scenario.iterations} power-iteration steps
        </span>
        <span>
          <span
            className="mr-1 inline-block size-2 rounded-sm"
            style={{ background: palette.css.pos }}
          />
          exact (eigenvector)
        </span>
        <span>
          <span
            className="mr-1 inline-block size-2 rounded-sm"
            style={{ background: palette.css.neg }}
          />
          random surfer, {run.scenario.surfers} clicks
        </span>
      </p>
    </div>
  );
}

export default function PageRankPage() {
  const [params, setParams] = useState<ParamValues>(() => fromScenario(DEFAULT_PAGERANK));
  const [activeId, setActiveId] = useState<string | null>(null);
  const scenario = useMemo(() => toScenario(params), [params]);
  const run = useMemo(() => runPageRank(scenario), [scenario]);
  const setParam = (key: string, value: ParamValue) => setParams((p) => ({ ...p, [key]: value }));

  const apply = (exp: PageRankExperiment) => {
    setParams(fromScenario({ ...DEFAULT_PAGERANK, ...exp.scenario }));
    setActiveId(exp.id);
  };
  const stillActive = (() => {
    const exp = pagerankExperiments.find((e) => e.id === activeId);
    return (
      !!exp &&
      JSON.stringify(fromScenario({ ...DEFAULT_PAGERANK, ...exp.scenario })) ===
        JSON.stringify(params)
    );
  })();

  const top = run.exact.indexOf(Math.max(...run.exact));
  const metrics: Metric[] = [
    { label: 'Top page', value: `${PAGE_NAMES[top]} · ${pct(run.exact[top])}`, testId: 'top-page' },
    {
      label: 'Steps to converge',
      value: Number.isFinite(run.stepsToConverge) ? String(run.stepsToConverge) : '> 200',
      hint: 'power-iteration steps until the ranks are within 0.001 of the exact answer',
    },
    {
      label: 'Error shrink per step',
      value: fix(run.rate, 2),
      hint: 'second eigenvalue of the Google matrix: never above the damping factor',
    },
    {
      label: 'Surfer vs exact',
      value: fix(run.surferGap, 3),
      hint: 'distance between the surfer’s time shares and the exact ranks',
    },
  ];

  const stages: [string, string][] = [
    ['lab-web', 'The web'],
    ['lab-bars', 'Three ways to the answer'],
    ['lab-conv', 'Convergence'],
    ['lab-how', 'How it works'],
    ['lab-controls', 'Settings'],
  ];

  return (
    <div className="mx-auto max-w-6xl px-4 pt-8 pb-20 sm:px-6 lg:px-8">
      <LabHeader
        eyebrow="Classic · probabilistic"
        title="Markov chains & PageRank"
        accent="where the random surfer ends up"
        blurb="A page is important if important pages link to it. That circular definition is the stationary distribution of a random walk on the links: find it by multiplying, by solving, or by simply letting a surfer click for long enough."
        badges={
          <>
            <Badge>{run.web.label}</Badge>
            <Badge>damping {scenario.damping.toFixed(2)}</Badge>
          </>
        }
      />
      <StageNav stages={stages} />
      <ExperimentBar
        lessons={pagerankExperiments}
        activeId={stillActive ? activeId : null}
        onApply={apply}
      />

      <div className="flex flex-col gap-5">
        <Panel id="lab-web" className="scroll-mt-24" title="The web" eyebrow="01">
          <div className="space-y-6">
            <MetricsStrip metrics={metrics} />
            <div className="grid gap-8 lg:grid-cols-[auto_minmax(0,1fr)]">
              <Graph run={run} />
              <p className="text-sm text-muted" data-testid="web-note">
                {run.web.note} Circles are sized by the rank after {scenario.iterations} power
                iteration{scenario.iterations === 1 ? '' : 's'}; arrows are links.
              </p>
            </div>
          </div>
        </Panel>
        <Panel id="lab-bars" className="scroll-mt-24" title="Three ways to the answer" eyebrow="02">
          <Bars run={run} />
        </Panel>
        <Panel id="lab-conv" className="scroll-mt-24" title="Convergence" eyebrow="03">
          <div className="grid gap-8 lg:grid-cols-[minmax(0,1fr)_20rem]">
            <MiniChart
              title="Distance to the exact ranks after each step"
              x={run.error.map((_, i) => i)}
              series={[
                {
                  name: 'distance',
                  color: 'var(--brand)',
                  values: run.error.map((e) => Math.max(e, 1e-12)),
                },
              ]}
              log
              xLabel="step"
              height={200}
            />
            <p className="text-sm text-muted">
              Every multiplication by the matrix shrinks the error by about the second-largest
              eigenvalue, which is at most the damping factor. On a log scale the error is a
              straight line whose slope is that rate. A stickier surfer (damping close to 1) or a
              trap makes the line shallower: convergence needs many more steps.
            </p>
          </div>
        </Panel>
        <Panel id="lab-how" className="scroll-mt-24" title="How it works" eyebrow="04">
          <div className="space-y-3 text-sm text-muted">
            <p>
              A Markov chain forgets its past: where it goes next only depends on where it is.
              Collect the probabilities in a matrix P, and the distribution of the walker after t
              steps is the starting distribution times P^t. Under mild conditions this converges to
              a <strong className="text-fg">stationary distribution</strong> π with πP = π, whatever
              the start.
            </p>
            <p>
              PageRank builds P from the links: follow a random link with probability d (damping),
              otherwise teleport to a uniformly random page, and teleport always from a page with no
              links. The teleport makes the chain connected, so π exists, is unique and is reached
              at rate d. π can be found by <strong className="text-fg">power iteration</strong>,
              exactly by solving a linear system, or by simulation. Tests check all three against
              each other and against closed forms for two-state chains.
            </p>
          </div>
        </Panel>
        <Panel id="lab-controls" className="scroll-mt-24" title="Settings" eyebrow="05">
          <ParamPanel specs={pagerankParams} values={params} onChange={setParam} columns={3} />
        </Panel>
      </div>
    </div>
  );
}
