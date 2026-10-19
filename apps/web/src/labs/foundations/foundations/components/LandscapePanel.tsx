import { OPTIMIZER_NAMES, optimizerInfo } from '@sim/core';
import { DESCENT_LIMITS, NEURON_DATASETS, neuronDatasetInfo } from '@sim/deep';
import { Pause, Play, RotateCcw, SkipForward } from 'lucide-react';
import { useEffect, useMemo, useRef, type PointerEvent } from 'react';
import { Button, Segmented, Slider, Tex } from '@/design-system';
import { mix, usePalette } from '@/shared/canvas/colors';
import { MiniChart } from '@/shared/components/MiniChart';
import { LR_LADDER, nearestIndex } from '@/shared/lab/ladders';
import { LAND_GRID, RANGE, SPEEDS, useFoundationsStore } from '../store';

const SIZE = 440;
const px = (v: number) => ((v + RANGE) / (2 * RANGE)) * SIZE;
const unpx = (p: number) => (p / SIZE) * 2 * RANGE - RANGE;
const trim = (v: number) => String(Number(v.toPrecision(3)));

/** The loss as a surface over the two weights, with a ball that an optimiser rolls downhill. */
export function LandscapePanel() {
  const palette = usePalette();
  const s = useFoundationsStore();
  const { landscape, path, descent, dataset, running, speed, gradientNorm } = s;
  const ref = useRef<HTMLCanvasElement>(null);

  // loss as colour: low (valley) = frost, high = ember, on a log scale so the valley is visible
  const heat = useMemo(() => {
    if (!landscape) return null;
    const canvas = document.createElement('canvas');
    canvas.width = LAND_GRID;
    canvas.height = LAND_GRID;
    const ctx = canvas.getContext('2d');
    if (!ctx) return null;
    let lo = Infinity;
    let hi = -Infinity;
    for (const v of landscape) {
      lo = Math.min(lo, v);
      hi = Math.max(hi, v);
    }
    const img = ctx.createImageData(LAND_GRID, LAND_GRID);
    for (let i = 0; i < landscape.length; i++) {
      const t = Math.log((landscape[i] as number) - lo + 0.05) - Math.log(0.05);
      const k = Math.min(1, t / (Math.log(hi - lo + 0.05) - Math.log(0.05) + 1e-9));
      const c = mix(palette.neg, palette.pos, Math.sqrt(k));
      img.data[i * 4] = c[0];
      img.data[i * 4 + 1] = c[1];
      img.data[i * 4 + 2] = c[2];
      img.data[i * 4 + 3] = 255;
    }
    ctx.putImageData(img, 0, 0);
    return canvas;
  }, [landscape, palette]);

  useEffect(() => {
    const canvas = ref.current;
    const ctx = canvas?.getContext('2d');
    if (!canvas || !ctx) return;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    canvas.width = SIZE * dpr;
    canvas.height = SIZE * dpr;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.fillStyle = palette.css.surface;
    ctx.fillRect(0, 0, SIZE, SIZE);
    if (heat) {
      ctx.imageSmoothingEnabled = true;
      ctx.globalAlpha = 0.6;
      ctx.drawImage(heat, 0, 0, SIZE, SIZE);
      ctx.globalAlpha = 1;
    }
    ctx.strokeStyle = palette.line;
    ctx.globalAlpha = 0.5;
    ctx.beginPath();
    ctx.moveTo(px(0), 0);
    ctx.lineTo(px(0), SIZE);
    ctx.moveTo(0, SIZE - px(0));
    ctx.lineTo(SIZE, SIZE - px(0));
    ctx.stroke();
    ctx.globalAlpha = 1;

    if (path.length > 1) {
      ctx.strokeStyle = palette.css.fg;
      ctx.lineWidth = 2;
      ctx.lineJoin = 'round';
      ctx.beginPath();
      path.forEach((p, i) => {
        const x = px(Math.max(-RANGE, Math.min(RANGE, p.w1)));
        const y = SIZE - px(Math.max(-RANGE, Math.min(RANGE, p.w2)));
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      });
      ctx.stroke();
    }
    const here = path[path.length - 1];
    if (here) {
      const x = px(Math.max(-RANGE, Math.min(RANGE, here.w1)));
      const y = SIZE - px(Math.max(-RANGE, Math.min(RANGE, here.w2)));
      ctx.fillStyle = palette.css.brand;
      ctx.strokeStyle = palette.css.surface;
      ctx.lineWidth = 3;
      ctx.shadowColor = palette.css.brand;
      ctx.shadowBlur = 12;
      ctx.beginPath();
      ctx.arc(x, y, 9, 0, Math.PI * 2);
      ctx.fill();
      ctx.shadowBlur = 0;
      ctx.stroke();
    }
    const start = path[0];
    if (start) {
      ctx.strokeStyle = palette.css.fg;
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.arc(px(start.w1), SIZE - px(start.w2), 5, 0, Math.PI * 2);
      ctx.stroke();
    }
  }, [heat, path, palette]);

  const place = (event: PointerEvent<HTMLCanvasElement>) => {
    const rect = event.currentTarget.getBoundingClientRect();
    s.place(
      unpx(((event.clientX - rect.left) / rect.width) * SIZE),
      -unpx(((event.clientY - rect.top) / rect.height) * SIZE),
    );
  };

  const series = useMemo(
    () => ({ x: path.map((_, i) => i), loss: path.map((p) => p.loss) }),
    [path],
  );
  const here = path[path.length - 1];

  return (
    <div className="space-y-6">
      <p className="max-w-3xl text-[0.9rem] text-muted">
        Learning means choosing the weights that make the loss (how wrong the neuron is on the data)
        as small as possible. Picture the loss as a landscape over the two weights, with valleys
        where the neuron is good. Gradient descent drops a ball on it and repeatedly moves it a
        little in the steepest downhill direction. Click anywhere to drop the ball somewhere else.
      </p>

      <div className="space-y-2">
        <Segmented
          label="Dataset"
          size="sm"
          value={dataset}
          onValueChange={s.setDataset}
          options={NEURON_DATASETS.map((d) => ({ value: d, label: neuronDatasetInfo[d].label }))}
        />
      </div>

      <div className="grid gap-8 lg:grid-cols-[auto_minmax(0,1fr)]">
        <div className="space-y-2">
          <canvas
            ref={ref}
            role="img"
            aria-label="The loss landscape over the two weights, with the path of the ball. Click to drop the ball."
            style={{ width: SIZE, height: 'auto', aspectRatio: '1' }}
            className="block max-w-full cursor-crosshair rounded-xl border border-line-strong"
            onPointerDown={place}
          />
          <div className="flex justify-between font-mono text-[0.68rem] text-faint">
            <span>w₁ → (−8 … 8)</span>
            <span>w₂ ↑</span>
          </div>
          <p className="text-[0.74rem] text-faint">
            Frost = low loss (a good neuron), ember = high loss. The ring marks where the ball
            started.
          </p>
        </div>

        <div className="min-w-0 space-y-5">
          <div className="flex flex-wrap items-center gap-2">
            <Button
              size="sm"
              onClick={s.toggleRunning}
              aria-label={running ? 'Pause the ball' : 'Roll the ball'}
              className="w-24"
            >
              {running ? <Pause className="size-4" /> : <Play className="size-4" />}
              {running ? 'Pause' : 'Roll'}
            </Button>
            <Button
              size="sm"
              variant="secondary"
              onClick={() => s.stepOnce(1)}
              aria-label="Take one step"
            >
              <SkipForward className="size-4" /> Step
            </Button>
            <Button size="sm" variant="ghost" onClick={s.reset}>
              <RotateCcw className="size-4" /> Back to start
            </Button>
            <Segmented
              label="Speed"
              size="sm"
              fit
              value={String(speed)}
              onValueChange={(v) => s.setSpeed(Number(v))}
              options={SPEEDS.map((v) => ({ value: String(v), label: `${v}×` }))}
            />
          </div>

          <div className="overflow-x-auto rounded-xl border border-line bg-raised/30 px-4 py-3 text-[0.95rem]">
            <Tex tex={String.raw`w\leftarrow w-\eta\,\dfrac{\partial L}{\partial w}`} />
          </div>

          <dl className="grid grid-cols-2 gap-x-6 gap-y-3 rounded-xl border border-line p-4 sm:grid-cols-4">
            <Stat label="Step" value={String(path.length - 1)} testId="step" />
            <Stat label="Loss" value={here ? here.loss.toFixed(3) : '–'} testId="loss" />
            <Stat
              testId="weights"
              label="Weights (w₁, w₂)"
              value={here ? `${here.w1.toFixed(1)}, ${here.w2.toFixed(1)}` : '–'}
            />
            <Stat label="Steepness |∇L|" value={gradientNorm.toFixed(3)} />
          </dl>

          <div className="grid gap-x-8 gap-y-5 sm:grid-cols-2">
            <div className="space-y-2 sm:col-span-2">
              <span className="text-[0.82rem] text-muted">Optimizer</span>
              <Segmented
                label="Optimizer"
                size="sm"
                value={descent.optimizer}
                onValueChange={(optimizer) => s.patchDescent({ optimizer })}
                options={OPTIMIZER_NAMES.map((o) => ({ value: o, label: optimizerInfo[o].label }))}
              />
              <p className="text-[0.74rem] text-faint">{optimizerInfo[descent.optimizer].note}</p>
            </div>
            <Slider
              label="Learning rate η"
              min={0}
              max={LR_LADDER.length - 1}
              value={nearestIndex(LR_LADDER, descent.lr)}
              onValueChange={(i) => s.patchDescent({ lr: LR_LADDER[i] as number })}
              format={() => trim(descent.lr)}
            />
            <Slider
              label="Bias b (fixed)"
              min={DESCENT_LIMITS.bias[0]}
              max={DESCENT_LIMITS.bias[1]}
              step={0.5}
              value={descent.bias}
              onValueChange={(bias) => s.patchDescent({ bias })}
              format={(v) => v.toFixed(1)}
            />
          </div>
          {s.notice && (
            <p
              role="alert"
              className="rounded-xl border border-danger/40 bg-danger/10 px-4 py-2 text-sm"
            >
              {s.notice}
            </p>
          )}
        </div>
      </div>

      <div className="grid gap-8 md:grid-cols-2">
        <MiniChart
          title="Loss along the path"
          x={series.x}
          xLabel="step"
          series={[{ name: 'loss', color: palette.css.brand, values: series.loss }]}
          note="A good run falls quickly and then flattens out at the bottom of a valley. A flat line high up means the ball is stuck on a plateau."
        />
        <div className="flex items-start">
          <Button size="sm" variant="secondary" onClick={s.useLearned}>
            Use these weights in the neuron above
          </Button>
        </div>
      </div>
    </div>
  );
}

function Stat({ label, value, testId }: { label: string; value: string; testId?: string }) {
  return (
    <div>
      <dt className="text-[0.68rem] text-faint">{label}</dt>
      <dd className="font-mono text-lg text-fg tabular-nums" data-testid={testId}>
        {value}
      </dd>
    </div>
  );
}
