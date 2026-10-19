import { makeSchedule, noiseScheduleInfo } from '@sim/deep';
import { Pause, Play } from 'lucide-react';
import { useEffect, useMemo, useState } from 'react';
import { Button, Slider, Tex } from '@/design-system';
import { usePalette } from '@/shared/canvas/colors';
import { MiniChart } from '@/shared/components/MiniChart';
import { useDiffusionStore } from '../store';
import { PlaneCanvas } from './PlaneCanvas';

/** Destroying data with noise, one slider: the forward process. */
export function ForwardPanel() {
  const palette = usePalette();
  const { config, time, noised, reference } = useDiffusionStore();
  const setTime = useDiffusionStore((s) => s.setTime);
  const [playing, setPlaying] = useState(false);

  const schedule = useMemo(
    () => makeSchedule(config.schedule, config.steps),
    [config.schedule, config.steps],
  );
  const signal = schedule.sqrtAlphaBar[time] as number;
  const noise = schedule.sqrtOneMinusAlphaBar[time] as number;

  // play the noising from 0 to T, then stop
  useEffect(() => {
    if (!playing) return;
    const id = window.setInterval(() => {
      const t = useDiffusionStore.getState().time;
      if (t >= useDiffusionStore.getState().config.steps) setPlaying(false);
      else useDiffusionStore.getState().setTime(t + 1);
    }, 90);
    return () => window.clearInterval(id);
  }, [playing]);

  const curve = useMemo(
    () => ({
      x: Array.from({ length: config.steps + 1 }, (_, t) => t),
      signal: Array.from(schedule.sqrtAlphaBar),
      noise: Array.from(schedule.sqrtOneMinusAlphaBar),
    }),
    [schedule, config.steps],
  );

  return (
    <div className="space-y-6">
      <p className="max-w-3xl text-[0.9rem] text-muted">
        A diffusion model starts by learning what <em>destroying</em> data looks like. Take real
        points and add a little Gaussian noise, again and again: after enough steps nothing of the
        original shape is left, just a round cloud. Drag the time slider to watch it happen. The
        reverse of this movie is how the model will create data.
      </p>
      <div className="grid gap-8 lg:grid-cols-[auto_minmax(0,1fr)]">
        <PlaneCanvas
          clouds={[
            ...(reference
              ? [{ points: reference, color: 'fg' as const, hollow: true, alpha: 0.25 }]
              : []),
            ...(noised ? [{ points: noised, color: 'brand' as const }] : []),
          ]}
          ariaLabel="Data points with noise added up to the chosen time; the faint rings are the clean data"
        />
        <div className="min-w-0 space-y-6">
          <div className="flex items-end gap-3">
            <div className="grow">
              <Slider
                label="Time t (0 = clean data, T = pure noise)"
                min={0}
                max={config.steps}
                value={time}
                onValueChange={(t) => {
                  setPlaying(false);
                  setTime(t);
                }}
                format={(v) => `${v} / ${config.steps}`}
              />
            </div>
            <Button
              size="sm"
              variant="secondary"
              aria-label={playing ? 'Pause the noising' : 'Play the noising'}
              onClick={() => {
                if (!playing && time >= config.steps) setTime(0);
                setPlaying(!playing);
              }}
            >
              {playing ? <Pause className="size-4" /> : <Play className="size-4" />}
            </Button>
          </div>

          <div className="rounded-xl border border-line bg-raised/30 px-4 py-3">
            <div className="overflow-x-auto text-[0.95rem]">
              <Tex
                tex={String.raw`x_t=\sqrt{\bar\alpha_t}\;x_0+\sqrt{1-\bar\alpha_t}\;\varepsilon,\qquad \varepsilon\sim\mathcal N(0,I)`}
              />
            </div>
            <p className="mt-2 font-mono text-[0.78rem] text-muted tabular-nums">
              signal kept <span className="text-fg">{(signal * 100).toFixed(0)}%</span> · noise{' '}
              <span className="text-fg">{(noise * 100).toFixed(0)}%</span>
            </p>
          </div>

          <MiniChart
            title={`${noiseScheduleInfo[config.schedule].label} schedule`}
            x={curve.x}
            xLabel="t"
            series={[
              { name: 'signal √ᾱ', color: palette.css.brand, values: curve.signal },
              {
                name: 'noise √(1−ᾱ)',
                color: palette.css.classes[0] as string,
                values: curve.noise,
              },
            ]}
            yRange={{ lo: 0, hi: 1 }}
            note={noiseScheduleInfo[config.schedule].note}
          />
        </div>
      </div>
    </div>
  );
}
