import { cellInfo, type EpochStats } from '@sim/deep';
import { Pause, Play, RotateCcw, SkipForward } from 'lucide-react';
import { useEffect, useMemo, useState } from 'react';
import { Badge, Button, Panel, Segmented } from '@/design-system';
import { LossChart } from '@/shared/components/LossChart';
import { ConfusionMatrix } from '@/shared/components/ConfusionMatrix';
import { ExperimentBar } from '@/shared/components/ExperimentBar';
import { taskInfo } from '@sim/deep';
import { GradientPanel } from './components/GradientPanel';
import { LengthPanel } from './components/LengthPanel';
import { SequencePanel } from './components/SequencePanel';
import { TrainingControls } from './components/TrainingControls';
import { UnrolledPanel } from './components/UnrolledPanel';
import { LESSONS, type Lesson } from './lessons';
import { SPEEDS, useRnnStore } from './store';

const STAGES = [
  ['rnn-read', 'Sequence'],
  ['rnn-unrolled', 'Unrolled'],
  ['rnn-gradient', 'Vanishing gradients'],
  ['rnn-train', 'Training'],
] as const;

export default function RnnPage() {
  const s = useRnnStore();
  const { config, history, epoch, progress, running, speed, notice } = s;
  const [activeId, setActiveId] = useState<string | null>(null);

  useEffect(() => {
    useRnnStore.getState().init();
    return () => useRnnStore.getState().setRunning(false);
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
    if (useRnnStore.getState().notice === null) {
      setActiveId(lesson.id);
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

  const stats: EpochStats[] = useMemo(
    () =>
      history.map((h) => ({
        epoch: h.epoch,
        lr: config.lr,
        trainLoss: h.trainLoss,
        testLoss: h.testLoss,
        trainMetric: h.trainAccuracy,
        testMetric: h.testAccuracy,
      })),
    [history, config.lr],
  );
  const last = history[history.length - 1];
  const info = taskInfo[config.task];

  return (
    <div className="mx-auto max-w-6xl px-4 pt-8 pb-20 sm:px-6 lg:px-8">
      <header className="mb-8">
        <p className="eyebrow text-brand">04 · Lab</p>
        <h1 className="mt-2 font-display text-[clamp(1.9rem,3.4vw,2.8rem)] leading-none font-semibold tracking-[-0.03em]">
          Recurrent <span className="font-serif font-normal italic text-muted">networks</span>
        </h1>
        <p className="mt-3 max-w-2xl text-[0.95rem] text-muted">
          Networks with memory. They read a sequence one token at a time, carrying a hidden state
          forward. Unroll one in time, watch the gates of an LSTM and a GRU open and close, and see
          why a plain RNN forgets.
        </p>
        <div className="mt-4 flex flex-wrap gap-2">
          <Badge>{cellInfo[config.cell].label}</Badge>
          <Badge>{config.hidden} hidden units</Badge>
          <Badge>{info.label}</Badge>
          <Badge>{epoch} epochs trained</Badge>
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
        <Panel id="rnn-read" className="scroll-mt-24" title="A task and a sequence" eyebrow="01">
          <SequencePanel />
        </Panel>

        <Panel
          id="rnn-unrolled"
          className="scroll-mt-24"
          title="The network unrolled in time"
          eyebrow="02"
        >
          <UnrolledPanel />
        </Panel>

        <Panel
          id="rnn-gradient"
          className="scroll-mt-24"
          title="Why long memory is hard"
          eyebrow="03"
        >
          <GradientPanel />
        </Panel>

        <Panel
          id="rnn-train"
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
                onClick={s.stepEpoch}
                aria-label="Train one epoch"
              >
                <SkipForward className="size-4" /> Epoch
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
            <div>
              <div className="mb-2 flex items-baseline justify-between font-mono text-[0.72rem] text-muted tabular-nums">
                <span>
                  epoch {epoch + 1} · {Math.round(progress * 100)}%
                </span>
                {last && (
                  <span>
                    test accuracy{' '}
                    <span className="text-fg">{(last.testAccuracy * 100).toFixed(1)}%</span>
                    {' · '}train {(last.trainAccuracy * 100).toFixed(1)}%
                  </span>
                )}
              </div>
              <div className="h-1.5 overflow-hidden rounded-full bg-raised">
                <div
                  className="h-full rounded-full bg-brand"
                  style={{ width: `${progress * 100}%` }}
                />
              </div>
            </div>

            {notice && (
              <p
                role="alert"
                className="rounded-xl border border-danger/40 bg-danger/10 px-4 py-2 text-sm"
              >
                {notice}
              </p>
            )}

            <div className="grid gap-10 lg:grid-cols-[minmax(0,1.2fr)_minmax(0,1fr)]">
              <div>
                <p className="eyebrow mb-3">Learning curves</p>
                <LossChart history={stats} problem="classification" />
              </div>
              <div>
                <p className="eyebrow mb-3">Where it gets confused (test sequences)</p>
                {s.confusion && (
                  <div className="mx-auto max-w-[22rem]">
                    <ConfusionMatrix
                      matrix={s.confusion}
                      labels={info.classSymbols}
                      noun="sequence"
                    />
                  </div>
                )}
              </div>
            </div>

            <div>
              <p className="eyebrow mb-3">Other sequence lengths</p>
              <LengthPanel />
            </div>

            <TrainingControls config={config} onPatch={s.patch} />
          </div>
        </Panel>
      </div>
    </div>
  );
}
