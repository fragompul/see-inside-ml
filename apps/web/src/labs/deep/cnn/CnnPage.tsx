import { type EpochStats } from '@sim/deep';
import { Pause, Play, RotateCcw, SkipForward } from 'lucide-react';
import { useEffect, useMemo } from 'react';
import { Badge, Button, Panel, Segmented } from '@/design-system';
import { LossChart } from '@/shared/components/LossChart';
import { ConfusionMatrix } from '@/shared/components/ConfusionMatrix';
import { AdversarialPanel } from './components/AdversarialPanel';
import { AttentionPanel } from './components/AttentionPanel';
import { ConvExplainer } from './components/ConvExplainer';
import { FeatureMaps } from './components/FeatureMaps';
import { ReadPanel } from './components/ReadPanel';
import { TrainingControls } from './components/TrainingControls';
import { SPEEDS, useCnnStore } from './store';

const STAGES = [
  ['cnn-read', 'Draw & read'],
  ['cnn-conv', 'Convolution'],
  ['cnn-maps', 'Feature maps'],
  ['cnn-look', 'Where it looks'],
  ['cnn-attack', 'Fooling it'],
  ['cnn-train', 'Training'],
] as const;

export default function CnnPage() {
  const s = useCnnStore();
  const { config, history, epoch, progress, running, speed, notice } = s;

  useEffect(() => {
    useCnnStore.getState().init();
    return () => useCnnStore.getState().setRunning(false);
  }, []);

  // The shared loss chart speaks `EpochStats`; accuracy goes in the "metric" slots.
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

  return (
    <div className="mx-auto max-w-6xl px-4 pt-8 pb-20 sm:px-6 lg:px-8">
      <header className="mb-8">
        <p className="eyebrow text-brand">03 · Lab</p>
        <h1 className="mt-2 font-display text-[clamp(1.9rem,3.4vw,2.8rem)] leading-none font-semibold tracking-[-0.03em]">
          Convolutional <span className="font-serif font-normal italic text-muted">networks</span>
        </h1>
        <p className="mt-3 max-w-2xl text-[0.95rem] text-muted">
          A small image classifier that learns to read handwritten digits. Draw one, see which
          patterns its filters pick out, and train the network yourself, all in your browser.
        </p>
        <div className="mt-4 flex flex-wrap gap-2">
          <Badge>16 × 16 pixels</Badge>
          <Badge>
            {config.filters1} + {config.filters2} filters
          </Badge>
          <Badge>
            {config.kernel}×{config.kernel} kernels
          </Badge>
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

      <div className="flex flex-col gap-5">
        <Panel
          id="cnn-read"
          className="scroll-mt-24"
          title="Draw a digit, read its answer"
          eyebrow="01"
        >
          <ReadPanel />
        </Panel>

        <Panel id="cnn-conv" className="scroll-mt-24" title="Convolution by hand" eyebrow="02">
          <ConvExplainer />
        </Panel>

        <Panel id="cnn-maps" className="scroll-mt-24" title="What each layer sees" eyebrow="03">
          <FeatureMaps />
        </Panel>

        <Panel id="cnn-look" className="scroll-mt-24" title="Where does it look?" eyebrow="04">
          <AttentionPanel />
        </Panel>

        <Panel id="cnn-attack" className="scroll-mt-24" title="Fool the network" eyebrow="05">
          <AdversarialPanel />
        </Panel>

        <Panel
          id="cnn-train"
          className="scroll-mt-24"
          title="Train it"
          eyebrow="06"
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
                <p className="eyebrow mb-3">Where it gets confused (test images)</p>
                {s.confusion && <ConfusionMatrix matrix={s.confusion} />}
              </div>
            </div>

            <TrainingControls config={config} onPatch={s.patch} />
          </div>
        </Panel>
      </div>
    </div>
  );
}
