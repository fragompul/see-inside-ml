import { Slider, Switch } from '@/design-system';
import { useState } from 'react';
import { useDiffusionStore } from '../store';
import { PlaneCanvas } from './PlaneCanvas';

/** What the denoiser has learned: at every spot, which way is "towards the data". */
export function ScorePanel() {
  const { config, time, noised, field, iteration } = useDiffusionStore();
  const setTime = useDiffusionStore((s) => s.setTime);
  const [arrows, setArrows] = useState(true);

  return (
    <div className="space-y-6">
      <p className="max-w-3xl text-[0.9rem] text-muted">
        The model is a small network that is shown a noisy point and the time, and has to guess the
        noise that was added. Subtract that guess and you move towards cleaner data. Each arrow
        below is the network&apos;s answer at that spot, drawn as the direction to move to make the
        point more likely. At high noise the arrows point to the middle; at low noise they snap onto
        the shape of the data.
      </p>
      <div className="grid gap-8 lg:grid-cols-[auto_minmax(0,1fr)]">
        <PlaneCanvas
          clouds={noised ? [{ points: noised, color: 'brand', alpha: 0.55, radius: 2.6 }] : []}
          arrows={arrows && field ? { points: field.points, vectors: field.score } : null}
          ariaLabel="Noisy points with the denoiser's arrows over the plane"
        />
        <div className="min-w-0 space-y-5">
          <Slider
            label="Noise level (time t)"
            min={1}
            max={config.steps}
            value={Math.max(1, time)}
            onValueChange={setTime}
            format={(v) => `${v} / ${config.steps}`}
          />
          <Switch label="Show the network's arrows" checked={arrows} onCheckedChange={setArrows} />
          <p className="text-[0.8rem] text-muted">
            The points are the data noised to this time. Arrows sit on a regular grid; a longer
            arrow means a stronger pull (the longest fits one grid cell).
          </p>
          {iteration === 0 && (
            <p className="rounded-xl border border-line bg-raised/40 px-4 py-3 text-[0.82rem] text-muted">
              The network has not been trained yet, so its arrows are random. Train it in section 04
              (or run an experiment above) and come back.
            </p>
          )}
        </div>
      </div>
    </div>
  );
}
