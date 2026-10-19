import { CLASSIFICATION_SETS, classificationSetInfo } from '@sim/core';
import { Dices } from 'lucide-react';
import { useDeferredValue, useMemo, useState } from 'react';
import { Badge, Button, Panel, Segmented, Slider } from '@/design-system';
import { ExperimentBar } from '@/shared/components/ExperimentBar';
import { LabHeader, RegionThumb, StageNav, fix, pct } from '@/shared/lab-kit';
import {
  SHOWDOWN_MODELS,
  defaultShowdown,
  experimentSetup,
  runShowdown,
  showdownExperiments,
  type ShowdownSetup,
} from './showdown';

const FAMILY_LABEL: Record<(typeof SHOWDOWN_MODELS)[number]['family'], string> = {
  linear: 'linear',
  probabilistic: 'Gaussian',
  instance: 'memory-based',
  tree: 'tree',
  ensemble: 'ensemble',
  kernel: 'kernel',
};

export default function ShowdownPage() {
  const [setup, setSetup] = useState<ShowdownSetup>(() => defaultShowdown());
  const [activeId, setActiveId] = useState<string | null>(null);
  const deferred = useDeferredValue(setup);
  const stale = deferred !== setup;
  const result = useMemo(() => runShowdown(deferred), [deferred]);

  const sorted = [...result.rows].sort(
    (a, b) => (b.testAccuracy ?? b.trainAccuracy) - (a.testAccuracy ?? a.trainAccuracy),
  );
  const winner = result.rows.find((r) => r.model.id === result.winner);
  const active = showdownExperiments.find((e) => e.id === activeId);
  const stillActive = active && JSON.stringify(experimentSetup(active)) === JSON.stringify(setup);

  return (
    <div className="mx-auto max-w-6xl px-4 pt-8 pb-20 sm:px-6 lg:px-8">
      <LabHeader
        eyebrow="Classic · supervised"
        title="Model showdown"
        accent="ten classifiers, one dataset"
        blurb="Every model sees the same points and the same train/test split, with default settings and no tuning. No model wins everywhere: the data decides."
        badges={
          <>
            <Badge>{classificationSetInfo[setup.dataset].label}</Badge>
            <Badge>{setup.n} points</Badge>
            <Badge>{SHOWDOWN_MODELS.length} models</Badge>
          </>
        }
      />
      <StageNav
        stages={[
          ['lab-arena', 'The arena'],
          ['lab-table', 'Scoreboard'],
          ['lab-controls', 'Data'],
        ]}
      />
      <ExperimentBar
        lessons={showdownExperiments}
        activeId={stillActive ? activeId : null}
        onApply={(e) => {
          setSetup(experimentSetup(e));
          setActiveId(e.id);
        }}
      />

      <div className="flex flex-col gap-5">
        <Panel id="lab-arena" className="scroll-mt-24" title="The arena" eyebrow="01">
          <div className="space-y-5" aria-busy={stale}>
            <Segmented
              label="Dataset"
              size="sm"
              fit
              value={setup.dataset}
              onValueChange={(dataset) => setSetup((s) => ({ ...s, dataset }))}
              options={CLASSIFICATION_SETS.map((d) => ({
                value: d,
                label: classificationSetInfo[d].label,
              }))}
            />
            <p className="-mt-2 text-[0.8rem] text-muted">
              {classificationSetInfo[setup.dataset].note}
            </p>
            <div
              className={`grid grid-cols-2 gap-5 sm:grid-cols-3 lg:grid-cols-5 ${stale ? 'opacity-60' : ''}`}
            >
              {result.rows.map((r) => (
                <div
                  key={r.model.id}
                  data-testid={`card-${r.model.id}`}
                  className={`space-y-2 rounded-xl border p-2.5 ${r.model.id === result.winner ? 'border-brand bg-brand-soft' : 'border-line'}`}
                >
                  <RegionThumb regions={r.regions} size={132} label={r.model.label} />
                  <p className="text-center font-mono text-[0.72rem] tabular-nums text-muted">
                    train {pct(r.trainAccuracy)} · test{' '}
                    <span className="text-fg">{pct(r.testAccuracy)}</span>
                  </p>
                </div>
              ))}
            </div>
            {winner && (
              <p className="text-sm text-muted" data-testid="winner">
                Best on held-out points: <strong className="text-fg">{winner.model.label}</strong> (
                {pct(winner.testAccuracy)}). Colours are the predicted class; paler = less
                confident.
              </p>
            )}
          </div>
        </Panel>

        <Panel id="lab-table" className="scroll-mt-24" title="Scoreboard" eyebrow="02">
          <div className="space-y-4">
            <div className="overflow-x-auto">
              <table className="w-full min-w-[36rem] text-left font-mono text-[0.8rem] tabular-nums">
                <thead className="text-muted">
                  <tr>
                    {['#', 'model', 'family', 'train', 'test', 'test log-loss', 'fit (ms)'].map(
                      (h) => (
                        <th key={h} className="pr-4 pb-2 font-medium">
                          {h}
                        </th>
                      ),
                    )}
                  </tr>
                </thead>
                <tbody>
                  {sorted.map((r, i) => (
                    <tr key={r.model.id} className="border-t border-line">
                      <td className="py-1.5 pr-4 text-faint">{i + 1}</td>
                      <td className="pr-4">{r.model.label}</td>
                      <td className="pr-4 text-muted">{FAMILY_LABEL[r.model.family]}</td>
                      <td className="pr-4">{pct(r.trainAccuracy)}</td>
                      <td className="pr-4">{pct(r.testAccuracy)}</td>
                      <td className="pr-4">{r.testLoss === null ? '–' : fix(r.testLoss)}</td>
                      <td>{r.fitMs.toFixed(0)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <p className="text-sm text-muted">
              A big gap between train and test means overfitting. Log-loss also judges the
              confidence: a model can be right yet badly over-sure. Fit time is measured in your
              browser.
            </p>
          </div>
        </Panel>

        <Panel id="lab-controls" className="scroll-mt-24" title="Data settings" eyebrow="03">
          <div className="space-y-5">
            <div className="grid gap-x-10 gap-y-6 sm:grid-cols-2 lg:grid-cols-3">
              <Slider
                label="Points"
                min={60}
                max={500}
                step={10}
                value={setup.n}
                onValueChange={(n) => setSetup((s) => ({ ...s, n }))}
              />
              <Slider
                label="Noise"
                min={0}
                max={1}
                step={0.05}
                value={setup.noise}
                onValueChange={(noise) =>
                  setSetup((s) => ({ ...s, noise: Number(noise.toFixed(2)) }))
                }
                format={(v) => v.toFixed(2)}
              />
            </div>
            <Button
              size="sm"
              variant="secondary"
              onClick={() => setSetup((s) => ({ ...s, seed: Math.floor(Math.random() * 1e6) }))}
            >
              <Dices className="size-4" /> New random sample
            </Button>
          </div>
        </Panel>
      </div>
    </div>
  );
}
