import { Pause, Play, RotateCcw, SkipForward } from 'lucide-react';
import { useEffect, useMemo, useState } from 'react';
import { Badge, Button, Panel, Segmented } from '@/design-system';
import { usePalette } from '@/shared/canvas/colors';
import { ExperimentBar } from '@/shared/components/ExperimentBar';
import { MiniChart } from '@/shared/components/MiniChart';
import { ForwardPanel } from './components/ForwardPanel';
import { SamplePanel } from './components/SamplePanel';
import { ScorePanel } from './components/ScorePanel';
import { TrainingControls } from './components/TrainingControls';
import { LESSONS, type Lesson } from './lessons';
import { SPEEDS, useDiffusionStore } from './store';

const STAGES = [
  ['diff-forward', 'Adding noise'],
  ['diff-score', 'What it learned'],
  ['diff-sample', 'Generating'],
  ['diff-train', 'Training'],
] as const;

export default function DiffusionPage() {
  const s = useDiffusionStore();
  const { config, history, iteration, running, speed, notice } = s;
  const palette = usePalette();
  const [activeId, setActiveId] = useState<string | null>(null);

  useEffect(() => {
    useDiffusionStore.getState().init();
    return () => useDiffusionStore.getState().setRunning(false);
  }, []);

  // An experiment stays "active" only while the lab still matches it.
  const active = LESSONS.find((l) => l.id === activeId);
  const stillActive =
    active &&
    (Object.entries(active.config) as [keyof typeof config, unknown][]).every(
      ([key, value]) => config[key] === value,
    );

  const apply = (lesson: Lesson) => {
    s.patch(lesson.config);
    if (useDiffusionStore.getState().notice === null) {
      setActiveId(lesson.id);
      if (lesson.claim.iterations > 0) s.setRunning(true);
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
    const measured = history.filter((h) => h.distance !== null);
    return {
      x: history.map((h) => h.iteration),
      loss: history.map((h) => h.loss),
      dx: measured.map((h) => h.iteration),
      distance: measured.map((h) => h.distance as number),
    };
  }, [history]);

  const last = history[history.length - 1];
  const lastDistance = [...history].reverse().find((h) => h.distance !== null)?.distance ?? null;

  return (
    <div className="mx-auto max-w-6xl px-4 pt-8 pb-20 sm:px-6 lg:px-8">
      <header className="mb-8">
        <p className="eyebrow text-brand">08 · Lab</p>
        <h1 className="mt-2 font-display text-[clamp(1.9rem,3.4vw,2.8rem)] leading-none font-semibold tracking-[-0.03em]">
          Diffusion <span className="font-serif font-normal italic text-muted">models</span>
        </h1>
        <p className="mt-3 max-w-2xl text-[0.95rem] text-muted">
          The idea behind modern image generators, on a plane you can see. Learn to undo a little
          noise, and generating becomes a walk from static to data. Drag through the noising,
          inspect what the network learned and watch points condense into shape.
        </p>
        <div className="mt-4 flex flex-wrap gap-2">
          <Badge>{config.steps} noising steps</Badge>
          <Badge>{config.sampler.toUpperCase()} sampler</Badge>
          <Badge>{iteration} iterations trained</Badge>
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
        <Panel
          id="diff-forward"
          className="scroll-mt-24"
          title="Destroying data with noise"
          eyebrow="01"
        >
          <ForwardPanel />
        </Panel>

        <Panel
          id="diff-score"
          className="scroll-mt-24"
          title="What the network learned"
          eyebrow="02"
        >
          <ScorePanel />
        </Panel>

        <Panel
          id="diff-sample"
          className="scroll-mt-24"
          title="Generating: noise to data"
          eyebrow="03"
        >
          <SamplePanel />
        </Panel>

        <Panel
          id="diff-train"
          className="scroll-mt-24"
          title="Train it"
          eyebrow="04"
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
                onClick={() => s.step(10)}
                aria-label="Run 10 steps"
              >
                <SkipForward className="size-4" /> 10 steps
              </Button>
              <Button size="sm" variant="ghost" onClick={s.reset} aria-label="Reset the network">
                <RotateCcw className="size-4" /> Reset
              </Button>
              <Segmented
                label="Training speed"
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
            <div className="flex flex-wrap items-baseline gap-x-8 gap-y-1 font-mono text-[0.78rem] text-muted tabular-nums">
              <span>
                iteration <span className="text-fg">{iteration}</span>
              </span>
              <span>
                noise-prediction error{' '}
                <span className="text-fg">{last ? last.loss.toFixed(3) : '–'}</span>
              </span>
              <span>
                distance of samples to data{' '}
                <span className="text-fg">
                  {lastDistance !== null ? lastDistance.toFixed(2) : '–'}
                </span>
              </span>
            </div>

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
                title="Noise-prediction error"
                x={series.x}
                series={[{ name: 'MSE', color: palette.css.brand, values: series.loss }]}
                log
                note="The only thing the network is trained on: how well it guesses the noise that was added to a point. A perfect guess is impossible (the noise is random), so it plateaus."
              />
              <MiniChart
                title="Quality of the samples"
                x={series.dx}
                series={[
                  {
                    name: 'distance',
                    color: palette.css.classes[2] as string,
                    values: series.distance,
                  },
                ]}
                log
                note="Sliced Wasserstein distance between generated points and real ones, measured every 100 iterations with 20 DDIM steps (lower is better)."
              />
            </div>

            <TrainingControls config={config} onPatch={s.patch} />
          </div>
        </Panel>
      </div>
    </div>
  );
}
