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
import { DEFAULT_DYNA_LAB, RUNS, episodesToLearn, runDyna, type DynaScenario } from './dynaCore';
import { dynaExperiments, dynaParams, type DynaExperiment } from './dynaSpec';

const toScenario = (p: ParamValues): DynaScenario => ({
  ...DEFAULT_DYNA_LAB,
  layout: p.layout as DynaScenario['layout'],
  planning: Number(p.planning),
  slip: Number(p.slip),
  episodes: Number(p.episodes),
});

const fromScenario = (s: DynaScenario): ParamValues => ({
  layout: s.layout,
  planning: s.planning,
  slip: s.slip,
  episodes: s.episodes,
});

const ARROWS = ['↑', '→', '↓', '←'];

export default function DynaPage() {
  const palette = usePalette();
  const [params, setParams] = useState<ParamValues>(() => fromScenario(DEFAULT_DYNA_LAB));
  const [activeId, setActiveId] = useState<string | null>(null);
  const scenario = useMemo(() => toScenario(params), [params]);
  const run = useMemo(() => runDyna(scenario), [scenario]);
  const setParam = (key: string, value: ParamValue) => setParams((p) => ({ ...p, [key]: value }));

  const apply = (exp: DynaExperiment) => {
    setParams(fromScenario({ ...DEFAULT_DYNA_LAB, ...exp.scenario }));
    setActiveId(exp.id);
  };
  const stillActive = (() => {
    const exp = dynaExperiments.find((e) => e.id === activeId);
    return (
      !!exp &&
      JSON.stringify(fromScenario({ ...DEFAULT_DYNA_LAB, ...exp.scenario })) ===
        JSON.stringify(params)
    );
  })();

  const learn = episodesToLearn(run.curve, run.shortest);
  const learnBase = episodesToLearn(run.baseline, run.shortest);
  const metrics: Metric[] = [
    {
      label: 'Real steps used (all episodes)',
      value: fix(run.realSteps, 0),
      testId: 'real-steps',
      hint: `average over ${RUNS} runs`,
    },
    { label: 'Without planning', value: fix(run.baselineSteps, 0) },
    {
      label: 'Episodes to find the shortest path',
      value: Number.isFinite(learn) ? String(learn) : `more than ${scenario.episodes}`,
      hint: `shortest path: ${run.shortest} steps`,
    },
    {
      label: 'Without planning',
      value: Number.isFinite(learnBase) ? String(learnBase) : `more than ${scenario.episodes}`,
    },
  ];

  const { width, height } = run.world.layout;
  const cell = Math.min(46, Math.floor(520 / width));
  const stages: [string, string][] = [
    ['lab-curve', 'Learning speed'],
    ['lab-policy', 'The learned route'],
    ['lab-how', 'How it works'],
    ['lab-controls', 'Settings'],
  ];

  return (
    <div className="mx-auto max-w-6xl px-4 pt-8 pb-20 sm:px-6 lg:px-8">
      <LabHeader
        eyebrow="Reinforcement learning"
        title="Dyna: planning with a model"
        accent="thinking instead of doing"
        blurb="Real experience is slow and costly; remembered experience is free. Dyna-Q learns a model of the world from what it has seen and replays it between real steps. Turn the amount of imagination up and watch the number of real steps fall, until the model is wrong."
        badges={
          <>
            <Badge>{scenario.planning} imagined updates per step</Badge>
            <Badge>{RUNS} runs averaged</Badge>
          </>
        }
      />
      <StageNav stages={stages} />
      <ExperimentBar
        lessons={dynaExperiments}
        activeId={stillActive ? activeId : null}
        onApply={apply}
      />

      <div className="flex flex-col gap-5">
        <Panel id="lab-curve" className="scroll-mt-24" title="Learning speed" eyebrow="01">
          <div className="space-y-6">
            <MetricsStrip metrics={metrics} className="lg:grid-cols-4" />
            <MiniChart
              title="Real steps needed to reach the goal, per episode (log scale)"
              x={run.curve.map((_, i) => i + 1)}
              xLabel="episode"
              log
              series={[
                {
                  name: `planning ${scenario.planning}`,
                  color: palette.css.brand,
                  values: run.curve,
                },
                {
                  name: 'no planning',
                  color: palette.css.muted,
                  values: run.baseline,
                  dashed: true,
                },
                {
                  name: 'shortest path',
                  color: palette.css.pos,
                  values: run.curve.map(() => run.shortest),
                  dashed: true,
                },
              ]}
              height={240}
            />
          </div>
        </Panel>
        <Panel id="lab-policy" className="scroll-mt-24" title="The learned route" eyebrow="02">
          <div className="space-y-3">
            <svg
              viewBox={`0 0 ${width * cell} ${height * cell}`}
              className="w-full max-w-xl rounded-xl border border-line"
              role="img"
              aria-label="The best move the agent has learned in each cell"
            >
              {run.world.layout.kinds.map((kind, s) => (
                <g key={s}>
                  <rect
                    x={(s % width) * cell}
                    y={Math.floor(s / width) * cell}
                    width={cell}
                    height={cell}
                    fill={
                      kind === 'wall'
                        ? palette.css.fg
                        : kind === 'goal'
                          ? palette.css.pos
                          : kind === 'pit'
                            ? palette.css.neg
                            : palette.css.surface
                    }
                    opacity={kind === 'wall' ? 0.35 : 0.6}
                    stroke={palette.css.muted}
                    strokeOpacity={0.25}
                  />
                  <text
                    x={(s % width) * cell + cell / 2}
                    y={Math.floor(s / width) * cell + cell / 2 + 6}
                    textAnchor="middle"
                    className="fill-fg font-mono text-[16px]"
                  >
                    {kind === 'empty' || kind === 'start'
                      ? ARROWS[run.single.policy[s] as number]
                      : kind === 'goal'
                        ? 'G'
                        : kind === 'pit'
                          ? 'X'
                          : ''}
                  </text>
                </g>
              ))}
            </svg>
            <p className="text-sm text-muted">
              Each cell shows the move the agent currently thinks is best, after {scenario.episodes}{' '}
              episodes of one run. Cells far from the goal that the agent has not yet been told
              about still point in arbitrary directions.
            </p>
          </div>
        </Panel>
        <Panel id="lab-how" className="scroll-mt-24" title="How it works" eyebrow="03">
          <div className="space-y-3 text-sm text-muted">
            <p>
              After every real step Dyna-Q does three things: it updates its Q-table from the step
              (as ordinary Q-learning does), it stores the step in a model (&ldquo;in this cell,
              this move gave this reward and led there&rdquo;), and then it picks <em>n</em>{' '}
              remembered (cell, move) pairs at random and updates the Q-table from what the model
              says would happen. The model costs nothing to query, so n can be large.
            </p>
            <p>
              The model here simply remembers the last outcome, which is exact in a deterministic
              world and a single sample in a slippery one. Replaying a single unrepresentative
              sample many times is how planning can hurt, as the last experiment shows. The tests
              check that planning needs far fewer real steps and that the agent still converges to
              the optimal route.
            </p>
          </div>
        </Panel>
        <Panel id="lab-controls" className="scroll-mt-24" title="Settings" eyebrow="04">
          <ParamPanel specs={dynaParams} values={params} onChange={setParam} columns={2} />
        </Panel>
      </div>
    </div>
  );
}
