import { GAN_DATASETS, ganDatasetInfo } from '@sim/deep';
import { Pause, Play, RotateCcw, SkipForward } from 'lucide-react';
import { useEffect, useMemo, useState } from 'react';
import { Badge, Button, Panel, Segmented, Switch, Tex } from '@/design-system';
import { usePalette } from '@/shared/canvas/colors';
import { ExperimentBar } from '@/shared/components/ExperimentBar';
import { MiniChart } from '@/shared/components/MiniChart';
import { DuelCanvas } from './components/DuelCanvas';
import { TrainingControls } from './components/TrainingControls';
import { LESSONS, type Lesson } from './lessons';
import { SPEEDS, useGanStore } from './store';

const STAGES = [
  ['gan-duel', 'The duel'],
  ['gan-charts', 'Is it working?'],
  ['gan-rules', 'The rules'],
] as const;

export default function GanPage() {
  const s = useGanStore();
  const { config, history, iteration, running, speed, notice } = s;
  const palette = usePalette();
  const [activeId, setActiveId] = useState<string | null>(null);
  const [showD, setShowD] = useState(true);
  const [showWarp, setShowWarp] = useState(false);

  useEffect(() => {
    useGanStore.getState().init();
    return () => useGanStore.getState().setRunning(false);
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
    if (useGanStore.getState().notice === null) {
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

  const c = palette.css;
  const series = useMemo(() => {
    const pick = (f: (h: (typeof history)[number]) => number) => history.map(f);
    return {
      x: history.map((h) => h.iteration),
      lossD: pick((h) => h.lossD),
      lossG: pick((h) => h.lossG),
      dReal: pick((h) => h.dReal),
      dFake: pick((h) => h.dFake),
      distance: pick((h) => h.distance),
      near: pick((h) => h.near),
      signal: pick((h) => h.signal),
    };
  }, [history]);

  const last = history[history.length - 1];
  const info = ganDatasetInfo[config.dataset];

  return (
    <div className="mx-auto max-w-6xl px-4 pt-8 pb-20 sm:px-6 lg:px-8">
      <header className="mb-8">
        <p className="eyebrow text-brand">07 · Lab</p>
        <h1 className="mt-2 font-display text-[clamp(1.9rem,3.4vw,2.8rem)] leading-none font-semibold tracking-[-0.03em]">
          Generative adversarial{' '}
          <span className="font-serif font-normal italic text-muted">networks</span>
        </h1>
        <p className="mt-3 max-w-2xl text-[0.95rem] text-muted">
          A forger and a detective train each other. The generator turns random noise into points;
          the discriminator tries to tell them from real ones. Watch the duel on a plane, break it
          on purpose, and see why GANs are famously hard to train.
        </p>
        <div className="mt-4 flex flex-wrap gap-2">
          <Badge>{info.label}</Badge>
          <Badge>{config.noiseDim}-d noise</Badge>
          <Badge>{iteration} iterations</Badge>
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
          id="gan-duel"
          className="scroll-mt-24"
          title="Forger against detective"
          eyebrow="01"
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
              <Button size="sm" variant="ghost" onClick={s.reset} aria-label="Reset the networks">
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
          <div className="space-y-6">
            <div className="space-y-2">
              <Segmented
                label="Data to imitate"
                size="sm"
                value={config.dataset}
                onValueChange={(dataset) => s.patch({ dataset })}
                options={GAN_DATASETS.map((d) => ({ value: d, label: ganDatasetInfo[d].label }))}
              />
              <p className="text-[0.82rem] text-muted">{info.note}</p>
            </div>

            <div className="grid gap-8 lg:grid-cols-[auto_minmax(0,1fr)]">
              <DuelCanvas
                real={s.real}
                fake={s.fake}
                discriminator={s.discriminator}
                warp={s.warp}
                showDiscriminator={showD}
                showWarp={showWarp}
              />

              <div className="min-w-0 space-y-5">
                <ul className="space-y-2 text-[0.84rem]">
                  <Legend
                    swatch={<span className="size-3 rounded-full border-[1.5px] border-fg" />}
                  >
                    <b className="font-medium text-fg">Real data</b> (300 examples of the target)
                  </Legend>
                  <Legend swatch={<span className="size-3 rounded-full bg-brand" />}>
                    <b className="font-medium text-fg">Generated points</b>, always from the same
                    300 noise vectors, so you see them move
                  </Legend>
                  <Legend
                    swatch={
                      <span className="h-3 w-6 rounded bg-gradient-to-r from-neg via-surface to-pos" />
                    }
                  >
                    <b className="font-medium text-fg">The detective&apos;s opinion</b>: frost =
                    &ldquo;fake&rdquo;, ember = &ldquo;real&rdquo;, grey = unsure
                  </Legend>
                </ul>

                <div className="space-y-3">
                  <Switch
                    label="Show the discriminator's opinion"
                    checked={showD}
                    onCheckedChange={setShowD}
                  />
                  <Switch
                    label="Show how the generator bends its input"
                    checked={showWarp}
                    onCheckedChange={setShowWarp}
                  />
                </div>

                <dl className="grid grid-cols-2 gap-x-6 gap-y-3 rounded-xl border border-line bg-raised/30 p-4 sm:grid-cols-3">
                  <Stat label="Iteration" value={String(iteration)} />
                  <Stat label="D says real (on real)" value={last ? last.dReal.toFixed(2) : '–'} />
                  <Stat label="D says real (on fake)" value={last ? last.dFake.toFixed(2) : '–'} />
                  <Stat
                    label="Distance to data"
                    value={last ? last.distance.toFixed(2) : '–'}
                    hint="sliced Wasserstein, lower is better"
                  />
                  <Stat
                    label="Points on the data"
                    value={last ? `${Math.round(last.near * 100)}%` : '–'}
                  />
                  {config.dataset === 'gaussians8' && (
                    <Stat
                      label="Clusters found"
                      value={last && last.modes !== null ? `${last.modes} / 8` : '–'}
                    />
                  )}
                </dl>
                {iteration === 0 && (
                  <p className="rounded-xl border border-line bg-raised/40 px-4 py-3 text-[0.82rem] text-muted">
                    The networks start random: the generator&apos;s points huddle near the middle
                    and the detective guesses. Press Train, or pick an experiment above.
                  </p>
                )}
              </div>
            </div>
          </div>
        </Panel>

        <Panel id="gan-charts" className="scroll-mt-24" title="Is it working?" eyebrow="02">
          <div className="grid gap-x-10 gap-y-8 md:grid-cols-2">
            <MiniChart
              title="What the detective thinks"
              x={series.x}
              series={[
                { name: 'D(real)', color: c.classes[0] as string, values: series.dReal },
                { name: 'D(fake)', color: c.classes[1] as string, values: series.dFake },
              ]}
              yRange={{ lo: 0, hi: 1 }}
              guide={0.5}
              note="A balanced duel hovers around 0.5 for both: the detective cannot tell them apart. A big gap means one player is winning."
            />
            <MiniChart
              title="Distance to the real data"
              x={series.x}
              series={[{ name: 'distance', color: c.brand, values: series.distance }]}
              log
              note="Sliced Wasserstein distance between the generated points and the real ones (lower is better). Unlike the losses, it really measures quality."
            />
            <MiniChart
              title="Losses"
              x={series.x}
              series={[
                { name: 'detective', color: c.classes[1] as string, values: series.lossD },
                { name: 'forger', color: c.classes[0] as string, values: series.lossG },
              ]}
              note="Both losses move against each other, so they say little about quality: a GAN can be improving while neither goes down."
            />
            <MiniChart
              title="Gradient to the generator"
              x={series.x}
              series={[{ name: 'signal', color: c.classes[2] as string, values: series.signal }]}
              log
              note="How strongly the detective pushes the forger. With the minimax loss it fades whenever the detective is winning."
            />
          </div>
        </Panel>

        <Panel id="gan-rules" className="scroll-mt-24" title="The rules of the game" eyebrow="03">
          <div className="space-y-8">
            <div className="grid gap-6 lg:grid-cols-2">
              <div className="space-y-3">
                <p className="max-w-xl text-[0.88rem] text-muted">
                  The discriminator <Tex tex="D" /> outputs the probability that a point is real. It
                  learns to push that to 1 for real points and 0 for generated ones. The generator{' '}
                  <Tex tex="G" /> turns noise <Tex tex="z" /> into a point and learns to make{' '}
                  <Tex tex="D" /> say 1 about its output.
                </p>
              </div>
              <div className="space-y-2 rounded-xl border border-line bg-raised/30 px-4 py-3 text-[0.95rem]">
                <div className="overflow-x-auto">
                  <Tex
                    tex={String.raw`\mathcal{L}_D=-\,\mathbb{E}_{x}\log D(x)\;-\;\mathbb{E}_{z}\log\!\big(1-D(G(z))\big)`}
                  />
                </div>
                <div className="overflow-x-auto">
                  <Tex
                    tex={
                      config.lossG === 'nonsaturating'
                        ? String.raw`\mathcal{L}_G=-\,\mathbb{E}_{z}\log D(G(z))`
                        : String.raw`\mathcal{L}_G=\;\mathbb{E}_{z}\log\!\big(1-D(G(z))\big)`
                    }
                  />
                </div>
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
            <TrainingControls config={config} onPatch={s.patch} />
          </div>
        </Panel>
      </div>
    </div>
  );
}

function Legend({ swatch, children }: { swatch: React.ReactNode; children: React.ReactNode }) {
  return (
    <li className="flex items-start gap-3 text-muted">
      <span className="mt-1 flex w-6 shrink-0 justify-center">{swatch}</span>
      <span>{children}</span>
    </li>
  );
}

function Stat({ label, value, hint }: { label: string; value: string; hint?: string }) {
  return (
    <div title={hint}>
      <dt className="text-[0.68rem] text-faint">{label}</dt>
      <dd className="font-mono text-lg text-fg tabular-nums">{value}</dd>
    </div>
  );
}
