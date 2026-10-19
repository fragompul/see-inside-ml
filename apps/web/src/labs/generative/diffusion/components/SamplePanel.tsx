import { SAMPLERS, samplerInfo } from '@sim/deep';
import { Pause, Play, RotateCcw } from 'lucide-react';
import { useEffect, useMemo, useState } from 'react';
import { Button, Segmented, Slider, Switch } from '@/design-system';
import { useDiffusionStore } from '../store';
import { PlaneCanvas } from './PlaneCanvas';

const TRAILS = 40;

/**
 * Generating data by running the movie backwards: noise in, a data cloud out. A different sampler
 * or step count changes the length of the path, so the player restarts (it is keyed on it).
 */
export function SamplePanel() {
  const length = useDiffusionStore((s) => s.trajectory?.times.length ?? 0);
  return <SamplePlayer key={length} />;
}

function SamplePlayer() {
  const { config, trajectory, reference, iteration } = useDiffusionStore();
  const patch = useDiffusionStore((s) => s.patch);
  const [step, setStep] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [ghost, setGhost] = useState(true);

  const last = Math.max(0, (trajectory?.times.length ?? 1) - 1);
  useEffect(() => {
    if (!playing) return;
    const id = window.setInterval(
      () => {
        setStep((s) => {
          if (s >= last) {
            setPlaying(false);
            return s;
          }
          return s + 1;
        });
      },
      last > 30 ? 60 : 130,
    );
    return () => window.clearInterval(id);
  }, [playing, last]);

  const current = useMemo(() => {
    if (!trajectory) return null;
    const n = trajectory.points;
    return trajectory.data.subarray(step * n * 2, (step + 1) * n * 2);
  }, [trajectory, step]);

  const t = trajectory?.times[step] ?? config.steps;

  return (
    <div className="space-y-6">
      <p className="max-w-3xl text-[0.9rem] text-muted">
        To generate data, start from pure noise and run the movie backwards: at each step the
        network guesses the noise in the current points, a little of it is removed, and the cloud
        slowly condenses into the data. Press play, or drag through the steps yourself.
      </p>
      <div className="grid gap-8 lg:grid-cols-[auto_minmax(0,1fr)]">
        <PlaneCanvas
          clouds={[
            ...(ghost && reference
              ? [{ points: reference, color: 'fg' as const, hollow: true, alpha: 0.28 }]
              : []),
            ...(current ? [{ points: current, color: 'brand' as const }] : []),
          ]}
          trails={
            trajectory
              ? { data: trajectory.data, points: trajectory.points, upto: step, count: TRAILS }
              : null
          }
          ariaLabel="Points being denoised step by step; lines show the paths of some of them"
        />
        <div className="min-w-0 space-y-6">
          <div className="flex flex-wrap items-center gap-2">
            <Button
              size="sm"
              onClick={() => {
                if (!playing && step >= last) setStep(0);
                setPlaying(!playing);
              }}
            >
              {playing ? <Pause className="size-4" /> : <Play className="size-4" />}
              {playing ? 'Pause' : step >= last ? 'Replay' : 'Generate'}
            </Button>
            <Button
              size="sm"
              variant="ghost"
              aria-label="Start over from the noise"
              onClick={() => {
                setPlaying(false);
                setStep(0);
              }}
            >
              <RotateCcw className="size-4" /> Start over
            </Button>
          </div>
          <Slider
            label="Denoising step"
            min={0}
            max={last}
            value={step}
            onValueChange={(v) => {
              setPlaying(false);
              setStep(v);
            }}
            format={(v) => `${v} / ${last} · noise level t = ${trajectory?.times[v] ?? '–'}`}
          />
          <Switch
            label="Show the real data (faint rings)"
            checked={ghost}
            onCheckedChange={setGhost}
          />

          <div className="space-y-3 rounded-xl border border-line bg-raised/30 p-4">
            <p className="eyebrow">Sampler</p>
            <Segmented
              label="Sampler"
              size="sm"
              value={config.sampler}
              onValueChange={(sampler) => patch({ sampler })}
              options={SAMPLERS.map((s) => ({ value: s, label: samplerInfo[s].label }))}
            />
            <p className="text-[0.76rem] text-faint">{samplerInfo[config.sampler].note}</p>
            {config.sampler === 'ddim' ? (
              <Slider
                label="Number of denoising steps"
                min={1}
                max={config.steps}
                value={Math.min(config.sampleSteps, config.steps)}
                onValueChange={(sampleSteps) => patch({ sampleSteps })}
              />
            ) : (
              <p className="text-[0.76rem] text-muted">
                DDPM always takes all {config.steps} steps ({config.steps} network evaluations).
              </p>
            )}
          </div>
          <p className="font-mono text-[0.76rem] text-muted tabular-nums">
            t = {t} · {last} network evaluations per sample
          </p>
          {iteration === 0 && (
            <p className="rounded-xl border border-line bg-raised/40 px-4 py-3 text-[0.82rem] text-muted">
              Untrained, the network predicts garbage, so the cloud wanders without finding the
              data. Train it first.
            </p>
          )}
        </div>
      </div>
    </div>
  );
}
