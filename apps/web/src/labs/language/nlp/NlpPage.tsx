import { Pause, Play, RotateCcw, SkipForward } from 'lucide-react';
import { useEffect, useMemo, useState } from 'react';
import { Badge, Button, Panel, Segmented } from '@/design-system';
import { usePalette } from '@/shared/canvas/colors';
import { ExperimentBar } from '@/shared/components/ExperimentBar';
import { MiniChart } from '@/shared/components/MiniChart';
import { AnalogyPanel } from './components/AnalogyPanel';
import { MapPanel } from './components/MapPanel';
import { PairsPanel } from './components/PairsPanel';
import { SimilarityPanel } from './components/SimilarityPanel';
import { TrainingControls } from './components/TrainingControls';
import { LESSONS, type Lesson } from './lessons';
import { SPEEDS, useNlpStore } from './store';

const STAGES = [
  ['nlp-pairs', 'Text to pairs'],
  ['nlp-map', 'The map'],
  ['nlp-similar', 'Neighbours'],
  ['nlp-analogy', 'Analogies'],
  ['nlp-train', 'Training'],
] as const;

export default function NlpPage() {
  const s = useNlpStore();
  const { config, history, step, epoch, running, speed, notice } = s;
  const palette = usePalette();
  const [activeId, setActiveId] = useState<string | null>(null);

  useEffect(() => {
    useNlpStore.getState().init();
    return () => useNlpStore.getState().setRunning(false);
  }, []);

  // An experiment stays "active" only while the lab still matches it.
  const active = LESSONS.find((l) => l.id === activeId);
  const stillActive =
    active &&
    (Object.entries(active.config) as [keyof typeof config, unknown][]).every(
      ([key, value]) => config[key] === value,
    );

  const apply = (lesson: Lesson) => {
    // experiments start from the default setup, changing only what they are about
    s.patch({ ...DEFAULT_RESET, ...lesson.config });
    if (useNlpStore.getState().notice === null) {
      setActiveId(lesson.id);
      s.reset();
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

  const series = useMemo(
    () => ({
      x: history.map((h) => h.step),
      loss: history.map((h) => h.loss),
      analogy: history.map((h) => h.analogy),
      purity: history.map((h) => h.purity),
    }),
    [history],
  );
  const last = history[history.length - 1];
  const c = palette.css;

  return (
    <div className="mx-auto max-w-6xl px-4 pt-8 pb-20 sm:px-6 lg:px-8">
      <header className="mb-8">
        <p className="eyebrow text-brand">09 · Lab</p>
        <h1 className="mt-2 font-display text-[clamp(1.9rem,3.4vw,2.8rem)] leading-none font-semibold tracking-[-0.03em]">
          Embeddings <span className="font-serif font-normal italic text-muted">& language</span>
        </h1>
        <p className="mt-3 max-w-2xl text-[0.95rem] text-muted">
          How a machine can learn what words mean from nothing but which words appear together.
          Train word vectors on a small invented world, explore the map they form, and do arithmetic
          with meaning: king − man + woman.
        </p>
        <div className="mt-4 flex flex-wrap gap-2">
          <Badge>{config.dim}-dimensional vectors</Badge>
          <Badge>window {config.window}</Badge>
          <Badge>{step.toLocaleString('en-US')} steps</Badge>
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
          id="nlp-pairs"
          className="scroll-mt-24"
          title="From text to training pairs"
          eyebrow="01"
        >
          <PairsPanel />
        </Panel>
        <Panel id="nlp-map" className="scroll-mt-24" title="The map of meaning" eyebrow="02">
          <MapPanel />
        </Panel>
        <Panel id="nlp-similar" className="scroll-mt-24" title="Nearest neighbours" eyebrow="03">
          <SimilarityPanel />
        </Panel>
        <Panel
          id="nlp-analogy"
          className="scroll-mt-24"
          title="Arithmetic with meaning"
          eyebrow="04"
        >
          <AnalogyPanel />
        </Panel>

        <Panel
          id="nlp-train"
          className="scroll-mt-24"
          title="Train it"
          eyebrow="05"
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
                onClick={() => s.stepOnce(25)}
                aria-label="Train 25 steps"
              >
                <SkipForward className="size-4" /> 25 steps
              </Button>
              <Button size="sm" variant="ghost" onClick={s.reset} aria-label="Reset the vectors">
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
                step <span className="text-fg">{step.toLocaleString('en-US')}</span> · epoch{' '}
                <span className="text-fg">{epoch}</span>
              </span>
              <span>
                loss <span className="text-fg">{last ? last.loss.toFixed(2) : '–'}</span>
              </span>
              <span>
                analogies right{' '}
                <span className="text-fg">{last ? `${Math.round(last.analogy * 100)}%` : '–'}</span>
              </span>
              <span>
                neighbours in the same family{' '}
                <span className="text-fg">{last ? `${Math.round(last.purity * 100)}%` : '–'}</span>
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
                title="Training loss"
                x={series.x}
                xLabel="step"
                series={[{ name: 'loss', color: c.brand, values: series.loss }]}
                note="How badly the model separates real context words from random ones (lower is better)."
              />
              <MiniChart
                title="What the vectors have learned"
                x={series.x}
                xLabel="step"
                series={[
                  {
                    name: 'same-family neighbours',
                    color: c.classes[2] as string,
                    values: series.purity,
                  },
                  {
                    name: 'analogies right',
                    color: c.classes[0] as string,
                    values: series.analogy,
                  },
                ]}
                yRange={{ lo: 0, hi: 1 }}
                note="Families form first; analogies, which need exact directions, take much longer."
              />
            </div>

            <TrainingControls config={config} onPatch={s.patch} />
          </div>
        </Panel>
      </div>
    </div>
  );
}

/** The settings experiments start from, so that one experiment does not leak into the next. */
const DEFAULT_RESET = { dim: 16, window: 4, negatives: 5, lr: 0.05, batchSize: 64 } as const;
