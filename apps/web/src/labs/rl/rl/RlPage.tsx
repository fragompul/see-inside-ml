import { Pause, Play, RotateCcw, SkipForward, FastForward } from 'lucide-react';
import { useEffect, useMemo, useState } from 'react';
import { Badge, Button, Panel, Segmented } from '@/design-system';
import { usePalette } from '@/shared/canvas/colors';
import { ExperimentBar } from '@/shared/components/ExperimentBar';
import { MiniChart } from '@/shared/components/MiniChart';
import { Controls } from './components/Controls';
import { WorldPanel } from './components/WorldPanel';
import { LESSONS, type Lesson } from './lessons';
import { SPEEDS, useRlStore } from './store';

const STAGES = [
  ['rl-world', 'The world'],
  ['rl-train', 'Learning'],
  ['rl-rules', 'The rules'],
] as const;

const WINDOW = 25;

/** Moving average of `values` over the previous `WINDOW` entries. */
function smooth(values: number[]): number[] {
  const out: number[] = [];
  let sum = 0;
  values.forEach((v, i) => {
    sum += v;
    if (i >= WINDOW) sum -= values[i - WINDOW] as number;
    out.push(sum / Math.min(i + 1, WINDOW));
  });
  return out;
}

export default function RlPage() {
  const s = useRlStore();
  const {
    config,
    history,
    episode,
    totalSteps,
    epsilon,
    running,
    speed,
    notice,
    snapshot,
    quality,
  } = s;
  const palette = usePalette();
  const [activeId, setActiveId] = useState<string | null>(null);

  useEffect(() => {
    useRlStore.getState().init();
    return () => useRlStore.getState().setRunning(false);
  }, []);

  // An experiment stays "active" only while the lab still matches it.
  const active = LESSONS.find((l) => l.id === activeId);
  const stillActive =
    active &&
    (Object.entries(active.config) as [keyof typeof config, unknown][]).every(
      ([key, value]) => JSON.stringify(config[key]) === JSON.stringify(value),
    );

  const apply = (lesson: Lesson) => {
    s.patch(lesson.config);
    if (useRlStore.getState().notice === null) {
      setActiveId(lesson.id);
      s.reset();
      s.setSpeed(100);
      s.setRunning(true);
      window.setTimeout(
        () =>
          document
            .getElementById(lesson.scrollTo)
            ?.scrollIntoView({ behavior: 'smooth', block: 'start' }),
        60,
      );
    }
  };

  const series = useMemo(() => {
    const x = history.map((h) => h.episode);
    return {
      x,
      ret: smooth(history.map((h) => h.ret)),
      goal: smooth(history.map((h) => (h.outcome === 'goal' ? 1 : 0))),
      pit: smooth(history.map((h) => (h.outcome === 'pit' ? 1 : 0))),
      eps: history.map((h) => h.epsilon),
      qx: quality.map((p) => p.episode),
      agreement: quality.map((p) => p.agreement),
    };
  }, [history, quality]);
  const c = palette.css;
  const recent = history.slice(-WINDOW);
  const recentReturn = recent.length ? recent.reduce((a, e) => a + e.ret, 0) / recent.length : null;

  return (
    <div className="mx-auto max-w-6xl px-4 pt-8 pb-20 sm:px-6 lg:px-8">
      <header className="mb-8">
        <p className="eyebrow text-brand">10 · Lab</p>
        <h1 className="mt-2 font-display text-[clamp(1.9rem,3.4vw,2.8rem)] leading-none font-semibold tracking-[-0.03em]">
          Reinforcement <span className="font-serif font-normal italic text-muted">learning</span>
        </h1>
        <p className="mt-3 max-w-2xl text-[0.95rem] text-muted">
          An agent that learns by trial and error, with no examples to copy. Drop it in a gridworld
          you can paint, let it wander, and watch value spread backwards from the goal until it
          knows the way. Then compare how Q-learning and SARSA behave beside a cliff.
        </p>
        <div className="mt-4 flex flex-wrap gap-2">
          <Badge>
            {config.algorithm === 'qlearning'
              ? 'Q-learning'
              : config.algorithm === 'sarsa'
                ? 'SARSA'
                : 'Expected SARSA'}
          </Badge>
          <Badge>
            {config.grid[0]?.length}×{config.grid.length} grid
          </Badge>
          <Badge>{episode.toLocaleString('en-US')} episodes</Badge>
        </div>
      </header>

      <nav
        aria-label="Sections of the lab"
        className="sticky top-16 z-30 -mx-4 mb-6 overflow-x-auto border-b border-line bg-bg/80 px-4 py-2.5 backdrop-blur-xl sm:-mx-6 sm:px-6 lg:-mx-8 lg:px-8"
      >
        <ol className="flex min-w-max items-center gap-1 font-mono text-[0.72rem]">
          {STAGES.map(([id, label], i) => (
            <li key={id} className="flex items-center gap-1">
              {i > 0 && <span className="text-faint">→</span>}
              <a
                href={`#${id}`}
                onClick={(e) => {
                  e.preventDefault();
                  document
                    .getElementById(id)
                    ?.scrollIntoView({ behavior: 'smooth', block: 'start' });
                }}
                className="rounded-full px-3 py-1 text-muted transition-colors hover:bg-brand-soft hover:text-fg"
              >
                <span className="mr-1.5 text-brand">{String(i + 1).padStart(2, '0')}</span>
                {label}
              </a>
            </li>
          ))}
        </ol>
      </nav>

      <ExperimentBar lessons={LESSONS} activeId={stillActive ? activeId : null} onApply={apply} />

      <div className="flex flex-col gap-5">
        <Panel id="rl-world" className="scroll-mt-24" title="A world to explore" eyebrow="01">
          <WorldPanel />
        </Panel>

        <Panel
          id="rl-train"
          className="scroll-mt-24"
          title="Learning by trial and error"
          eyebrow="02"
          action={
            <div className="flex flex-wrap items-center gap-2">
              <Button
                size="sm"
                onClick={s.toggleRunning}
                aria-label={running ? 'Pause training' : 'Train the network'}
                className="w-24"
              >
                {running ? <Pause className="size-4" /> : <Play className="size-4" />}
                {running ? 'Pause' : 'Train'}
              </Button>
              <Button
                size="sm"
                variant="secondary"
                onClick={() => s.stepOnce(1)}
                aria-label="Take one step"
              >
                <SkipForward className="size-4" /> Step
              </Button>
              <Button
                size="sm"
                variant="secondary"
                onClick={() => s.episodes(100)}
                aria-label="Run 100 episodes"
              >
                <FastForward className="size-4" /> 100 episodes
              </Button>
              <Button size="sm" variant="ghost" onClick={s.reset} aria-label="Reset the agent">
                <RotateCcw className="size-4" /> Reset
              </Button>
              <Segmented
                label="Training speed (steps per frame)"
                size="sm"
                fit
                value={String(speed)}
                onValueChange={(v) => s.setSpeed(Number(v))}
                options={SPEEDS.map((v) => ({ value: String(v), label: `${v}×` }))}
              />
            </div>
          }
        >
          <div className="space-y-8">
            <dl className="grid grid-cols-2 gap-x-6 gap-y-3 rounded-xl border border-line bg-raised/30 p-4 sm:grid-cols-3 lg:grid-cols-6">
              <Stat label="Episodes" value={String(episode)} testId="episodes" />
              <Stat label="Steps taken" value={String(totalSteps)} testId="steps" />
              <Stat label="Random moves ε" value={epsilon.toFixed(2)} />
              <Stat
                label={`Return (last ${WINDOW})`}
                value={recentReturn === null ? '–' : recentReturn.toFixed(1)}
              />
              <Stat
                label="Moves that are optimal"
                value={snapshot && episode > 0 ? `${Math.round(snapshot.agreement * 100)}%` : '–'}
                hint="share of cells where the best-looking move is truly optimal"
                testId="agreement"
              />
              <Stat
                label="Value error"
                value={snapshot && episode > 0 ? snapshot.valueError.toFixed(2) : '–'}
                hint="mean distance between its estimates and the true optimal values"
              />
            </dl>

            {notice && (
              <p
                role="alert"
                className="rounded-xl border border-danger/40 bg-danger/10 px-4 py-2 text-sm"
              >
                {notice}
              </p>
            )}

            <div className="grid gap-x-10 gap-y-8 md:grid-cols-2">
              <MiniChart
                title={`Return per episode (average of ${WINDOW})`}
                x={series.x}
                xLabel="episode"
                series={[{ name: 'return', color: c.brand, values: series.ret }]}
                note="Total reward of an episode. It rises as the agent stops wandering and stops falling."
              />
              <MiniChart
                title="How episodes end"
                x={series.x}
                xLabel="episode"
                series={[
                  { name: 'reach the goal', color: c.classes[2] as string, values: series.goal },
                  { name: 'fall in a pit', color: c.classes[4] as string, values: series.pit },
                ]}
                yRange={{ lo: 0, hi: 1 }}
                note="Share of the recent episodes that end at the goal or in a pit (the rest time out)."
              />
              <MiniChart
                title="Closeness to the optimal moves"
                x={series.qx}
                xLabel="episode"
                series={[
                  {
                    name: 'optimal moves',
                    color: c.classes[0] as string,
                    values: series.agreement,
                  },
                ]}
                yRange={{ lo: 0, hi: 1 }}
                note="Compared with value iteration, which knows the rules: the share of cells where the agent's best move is also an optimal one."
              />
              <MiniChart
                title="Exploration rate ε"
                x={series.x}
                xLabel="episode"
                series={[{ name: 'ε', color: c.classes[1] as string, values: series.eps }]}
                yRange={{ lo: 0, hi: 1 }}
                note="Chance of a random move: high at first so it tries things, then lower so it can use what it knows."
              />
            </div>
          </div>
        </Panel>

        <Panel id="rl-rules" className="scroll-mt-24" title="The rules of learning" eyebrow="03">
          <Controls config={config} onPatch={s.patch} />
        </Panel>
      </div>
    </div>
  );
}

function Stat({
  label,
  value,
  hint,
  testId,
}: {
  label: string;
  value: string;
  hint?: string;
  testId?: string;
}) {
  return (
    <div title={hint}>
      <dt className="text-[0.68rem] text-faint">{label}</dt>
      <dd className="font-mono text-lg text-fg tabular-nums" data-testid={testId}>
        {value}
      </dd>
    </div>
  );
}
