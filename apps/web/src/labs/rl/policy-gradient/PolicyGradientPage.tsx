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
  BASELINES,
  BASELINE_LABELS,
  DEFAULT_POLICY_LAB,
  SEEDS,
  runPolicy,
  type PolicyRun,
  type PolicyScenario,
} from './policyCore';
import { policyExperiments, policyParams, type PolicyExperiment } from './policySpec';

const toScenario = (p: ParamValues): PolicyScenario => ({
  ...DEFAULT_POLICY_LAB,
  layout: p.layout as PolicyScenario['layout'],
  baseline: p.baseline as PolicyScenario['baseline'],
  learningRate: Number(p.learningRate),
  episodes: Number(p.episodes),
  batch: Number(p.batch),
  entropy: Number(p.entropy),
  slip: Number(p.slip),
});

const fromScenario = (s: PolicyScenario): ParamValues => ({
  layout: s.layout,
  baseline: s.baseline,
  learningRate: s.learningRate,
  episodes: s.episodes,
  batch: s.batch,
  entropy: s.entropy,
  slip: s.slip,
});

const DX = [0, 1, 0, -1];
const DY = [-1, 0, 1, 0];

function Policy({ run }: { run: PolicyRun }) {
  const palette = usePalette();
  const { width, height } = run.world.layout;
  const cell = Math.min(60, Math.floor(620 / width));
  const w = width * cell;
  const h = height * cell;
  const agent = run.training.agent;
  // the path of a greedy rollout from the start (stops at a terminal cell or after a loop)
  const path = useMemo(() => {
    const out = [run.world.start];
    const greedy = agent.greedyPolicy();
    const seen = new Set(out);
    let s = run.world.start;
    for (let i = 0; i < 60; i++) {
      if (run.world.isTerminal(s)) break;
      s = run.world.move(s, greedy[s] as number);
      if (seen.has(s)) break;
      seen.add(s);
      out.push(s);
    }
    return out;
  }, [run, agent]);
  return (
    <div className="space-y-3">
      <svg
        viewBox={`0 0 ${w} ${h}`}
        className="w-full max-w-2xl rounded-xl border border-line"
        role="img"
        aria-label="The learned policy: arrows in each cell are as long as the probability of that move"
      >
        {run.world.layout.kinds.map((kind, s) => {
          const x = (s % width) * cell;
          const y = Math.floor(s / width) * cell;
          const fill =
            kind === 'wall'
              ? palette.css.fg
              : kind === 'goal'
                ? palette.css.pos
                : kind === 'pit'
                  ? palette.css.neg
                  : palette.css.surface;
          return (
            <g key={s}>
              <rect
                x={x}
                y={y}
                width={cell}
                height={cell}
                fill={fill}
                opacity={kind === 'wall' ? 0.35 : kind === 'empty' || kind === 'start' ? 0.9 : 0.5}
                stroke={palette.css.muted}
                strokeOpacity={0.25}
              />
              {(kind === 'empty' || kind === 'start') &&
                agent
                  .probabilities(s)
                  .map((p, a) => (
                    <line
                      key={a}
                      x1={x + cell / 2}
                      y1={y + cell / 2}
                      x2={x + cell / 2 + (DX[a] as number) * p * cell * 0.45}
                      y2={y + cell / 2 + (DY[a] as number) * p * cell * 0.45}
                      stroke={palette.css.brand}
                      strokeWidth={Math.max(1, 3.2 * p)}
                      strokeLinecap="round"
                      opacity={0.85}
                    />
                  ))}
              {kind === 'start' && (
                <text x={x + 4} y={y + 12} className="fill-fg font-mono text-[10px]">
                  S
                </text>
              )}
              {kind === 'goal' && (
                <text
                  x={x + cell / 2}
                  y={y + cell / 2 + 5}
                  textAnchor="middle"
                  className="fill-fg font-mono text-[14px] font-semibold"
                >
                  G
                </text>
              )}
              {kind === 'pit' && (
                <text
                  x={x + cell / 2}
                  y={y + cell / 2 + 5}
                  textAnchor="middle"
                  className="fill-fg font-mono text-[14px] font-semibold"
                >
                  X
                </text>
              )}
            </g>
          );
        })}
        <polyline
          points={path
            .map(
              (s) => `${(s % width) * cell + cell / 2},${Math.floor(s / width) * cell + cell / 2}`,
            )
            .join(' ')}
          fill="none"
          stroke={palette.css.fg}
          strokeWidth={2.4}
          strokeDasharray="4 3"
          opacity={0.8}
        />
      </svg>
      <p className="text-sm text-muted" data-testid="policy-note">
        Each open cell shows the policy as four arrows whose length and thickness are the
        probabilities of moving up, right, down and left. Early on they are equal (a cross); as the
        agent learns, one arrow dominates. The dashed line follows the most likely move from the
        start. Walls are shaded, G is the goal and X a pit.
      </p>
    </div>
  );
}

export default function PolicyGradientPage() {
  const palette = usePalette();
  const [params, setParams] = useState<ParamValues>(() => fromScenario(DEFAULT_POLICY_LAB));
  const [activeId, setActiveId] = useState<string | null>(null);
  const scenario = useMemo(() => toScenario(params), [params]);
  const run = useMemo(() => runPolicy(scenario), [scenario]);
  const setParam = (key: string, value: ParamValue) => setParams((p) => ({ ...p, [key]: value }));

  const apply = (exp: PolicyExperiment) => {
    setParams(fromScenario({ ...DEFAULT_POLICY_LAB, ...exp.scenario }));
    setActiveId(exp.id);
  };
  const stillActive = (() => {
    const exp = policyExperiments.find((e) => e.id === activeId);
    return (
      !!exp &&
      JSON.stringify(fromScenario({ ...DEFAULT_POLICY_LAB, ...exp.scenario })) ===
        JSON.stringify(params)
    );
  })();

  const metrics: Metric[] = [
    {
      label: 'Return (last 100 episodes)',
      value: fix(run.finalReturn, 2),
      testId: 'final-return',
      hint: 'average total reward of the shown run',
    },
    {
      label: 'Best possible',
      value: fix(run.optimal, 2),
      hint: 'the optimal policy, from value iteration',
    },
    {
      label: 'Reached the goal',
      value: pct(run.goalRate),
      hint: 'share of the last 100 episodes that ended at the goal',
    },
  ];

  const stages: [string, string][] = [
    ['lab-policy', 'The learned policy'],
    ['lab-curve', 'Learning curves'],
    ['lab-how', 'How it works'],
    ['lab-controls', 'Settings'],
  ];

  return (
    <div className="mx-auto max-w-6xl px-4 pt-8 pb-20 sm:px-6 lg:px-8">
      <LabHeader
        eyebrow="Reinforcement learning"
        title="Policy gradients"
        accent="learning the policy itself"
        blurb="Instead of learning how good each state is, learn directly the probability of each move. REINFORCE nudges the policy towards actions that were followed by high return; a baseline cuts the noise of that estimate without biasing it."
        badges={
          <>
            <Badge>{BASELINE_LABELS[scenario.baseline]}</Badge>
            <Badge>{scenario.episodes} episodes</Badge>
          </>
        }
      />
      <StageNav stages={stages} />
      <ExperimentBar
        lessons={policyExperiments}
        activeId={stillActive ? activeId : null}
        onApply={apply}
      />

      <div className="flex flex-col gap-5">
        <Panel id="lab-policy" className="scroll-mt-24" title="What the agent learned" eyebrow="01">
          <div className="space-y-6">
            <MetricsStrip metrics={metrics} className="lg:grid-cols-3" />
            <Policy run={run} />
          </div>
        </Panel>
        <Panel id="lab-curve" className="scroll-mt-24" title="Learning curves" eyebrow="02">
          <div className="grid gap-8 lg:grid-cols-[minmax(0,1fr)_21rem]">
            <MiniChart
              title={`Return per episode, averaged over ${SEEDS} runs (moving average of 20)`}
              x={run.curves.none.map((_, i) => i + 1)}
              xLabel="episode"
              series={BASELINES.map((b, i) => ({
                name: BASELINE_LABELS[b],
                color: palette.css.classes[i % palette.css.classes.length] as string,
                values: run.curves[b],
              }))}
              height={240}
            />
            <div className="space-y-2 text-sm text-muted">
              <p>Final return, mean ± spread over {SEEDS} runs:</p>
              <ul className="space-y-1 font-mono text-[0.78rem]" data-testid="summary">
                {BASELINES.map((b) => (
                  <li key={b} className="flex justify-between gap-3">
                    <span className="font-sans">{BASELINE_LABELS[b]}</span>
                    <span>
                      {fix(run.summary[b].mean, 2)} ± {fix(run.summary[b].sd, 2)}
                    </span>
                  </li>
                ))}
              </ul>
              <p>
                A large spread means that some runs learned and others did not: the estimate is too
                noisy for the step size. A baseline shrinks it.
              </p>
            </div>
          </div>
        </Panel>
        <Panel id="lab-how" className="scroll-mt-24" title="How it works" eyebrow="03">
          <div className="space-y-3 text-sm text-muted">
            <p>
              The policy is a softmax over a table of preferences. Play an episode, compute the
              return G that followed each move, and for each move raise the preference of the action
              taken in proportion to (G − b) and lower the others by their probability:{' '}
              <em>θ ← θ + α (G − b) (1[a] − π)</em>. In expectation this is the true gradient of the
              average return (the tests check that by sampling), but any single episode gives a very
              noisy direction.
            </p>
            <p>
              The <strong className="text-fg">baseline</strong> b may depend on the state but not on
              the action, so it adds no bias; a good one (the typical return from that state) makes
              the advantage G − b small, so the update is much less noisy. The{' '}
              <strong className="text-fg">entropy bonus</strong> keeps the policy from becoming
              deterministic too early. This is the root of actor-critic methods, where the baseline
              is a learned value function.
            </p>
          </div>
        </Panel>
        <Panel id="lab-controls" className="scroll-mt-24" title="Settings" eyebrow="04">
          <ParamPanel specs={policyParams} values={params} onChange={setParam} columns={3} />
        </Panel>
      </div>
    </div>
  );
}
