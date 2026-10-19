import { type EpochStats } from '@sim/deep';
import { Pause, Play, RotateCcw, SkipForward } from 'lucide-react';
import { useEffect, useMemo, useState } from 'react';
import { Badge, Button, Panel, Segmented } from '@/design-system';
import { ExperimentBar } from '@/shared/components/ExperimentBar';
import { LossChart } from '@/shared/components/LossChart';
import { ImaginePanel } from './components/ImaginePanel';
import { LatentMap } from './components/LatentMap';
import { ReconstructPanel } from './components/ReconstructPanel';
import { TrainingControls } from './components/TrainingControls';
import { LESSONS, type Lesson } from './lessons';
import { SPEEDS, useAeStore } from './store';

const STAGES = [
  ['ae-read', 'Reconstruct'],
  ['ae-latent', 'Latent space'],
  ['ae-imagine', 'Interpolate & imagine'],
  ['ae-train', 'Training'],
] as const;

export default function AutoencoderPage() {
  const s = useAeStore();
  const { config, history, epoch, progress, running, speed, notice } = s;
  const [activeId, setActiveId] = useState<string | null>(null);

  useEffect(() => {
    useAeStore.getState().init();
    return () => useAeStore.getState().setRunning(false);
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
    if (useAeStore.getState().notice === null) {
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

  // The shared loss chart speaks `EpochStats`: total loss in the loss slots, reconstruction
  // error in the "metric" slots.
  const stats: EpochStats[] = useMemo(
    () =>
      history.map((h) => ({
        epoch: h.epoch,
        lr: config.lr,
        trainLoss: h.trainTotal,
        testLoss: h.testTotal,
        trainMetric: h.trainRecon,
        testMetric: h.testRecon,
      })),
    [history, config.lr],
  );
  const last = history[history.length - 1];
  const vae = config.mode === 'vae';

  return (
    <div className="mx-auto max-w-6xl px-4 pt-8 pb-20 sm:px-6 lg:px-8">
      <header className="mb-8">
        <p className="eyebrow text-brand">06 · Lab</p>
        <h1 className="mt-2 font-display text-[clamp(1.9rem,3.4vw,2.8rem)] leading-none font-semibold tracking-[-0.03em]">
          Autoencoders <span className="font-serif font-normal italic text-muted">& VAEs</span>
        </h1>
        <p className="mt-3 max-w-2xl text-[0.95rem] text-muted">
          Networks that learn by copying their own input through a tiny bottleneck. What they keep
          in those few numbers becomes a map of the data: explore it, walk through it, and turn it
          into a generator.
        </p>
        <div className="mt-4 flex flex-wrap gap-2">
          <Badge>{vae ? 'Variational autoencoder' : 'Autoencoder'}</Badge>
          <Badge>{config.latent} numbers in the code</Badge>
          {config.corruption > 0 && <Badge>denoising</Badge>}
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
        <Panel
          id="ae-read"
          className="scroll-mt-24"
          title="Squeeze a picture, rebuild it"
          eyebrow="01"
        >
          <ReconstructPanel />
        </Panel>

        <Panel id="ae-latent" className="scroll-mt-24" title="The latent space" eyebrow="02">
          <LatentMap />
        </Panel>

        <Panel
          id="ae-imagine"
          className="scroll-mt-24"
          title="Interpolation and imagination"
          eyebrow="03"
        >
          <ImaginePanel />
        </Panel>

        <Panel
          id="ae-train"
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
              <div className="mb-2 flex flex-wrap items-baseline justify-between gap-x-6 font-mono text-[0.72rem] text-muted tabular-nums">
                <span>
                  epoch {epoch + 1} · {Math.round(progress * 100)}%
                </span>
                {last && (
                  <span>
                    test reconstruction <span className="text-fg">{last.testRecon.toFixed(1)}</span>
                    {vae && (
                      <>
                        {' · '}KL <span className="text-fg">{last.testKl.toFixed(2)}</span>
                      </>
                    )}
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

            <div>
              <p className="eyebrow mb-3">Learning curves</p>
              <LossChart history={stats} problem="regression" metricName="reconstruction" />
              <p className="mt-2 text-[0.74rem] text-faint">
                {vae
                  ? 'Loss = reconstruction error + β × KL. Reconstruction error is the pixel-wise cross-entropy of the rebuilt picture; KL measures how far the codes drift from a standard normal.'
                  : 'Loss = reconstruction error: the pixel-wise cross-entropy between the rebuilt picture and the original (lower is better).'}
              </p>
            </div>

            <TrainingControls config={config} onPatch={s.patch} />
          </div>
        </Panel>
      </div>
    </div>
  );
}
