import { IMAGE_SIZE } from '@sim/deep';
import { ArrowRight, Dices, Eraser } from 'lucide-react';
import { Button, Slider, Switch, cn } from '@/design-system';
import { PixelCanvas } from '@/shared/components/PixelCanvas';
import { networkInput, useAeStore } from '../store';

const PAD = 224;
const SHOW = 160;

/** Original → code → reconstruction, with the error map and the code's numbers. */
export function ReconstructPanel() {
  const s = useAeStore();
  const { config, code, recon, input, image, source } = s;
  const blank = source === 'blank' || !code || !recon;
  const vae = config.mode === 'vae';

  // measured against the clean picture, so with noise on it tells whether the noise was removed
  const clean = networkInput(image, s.center, 0);
  const error = blank ? null : Float64Array.from(clean, (v, i) => (recon[i] as number) - v);
  const mse = error ? error.reduce((a, e) => a + e * e, 0) / error.length : null;

  return (
    <div className="space-y-6">
      <p className="max-w-3xl text-[0.9rem] text-muted">
        An autoencoder is two networks back to back. The <em>encoder</em> squeezes a picture into a
        handful of numbers (the <em>code</em>); the <em>decoder</em> paints a picture from those
        numbers alone. It is trained to make the output match the input, so the code has to keep
        whatever matters. Draw a digit or pick one and see how well it survives the squeeze.
      </p>

      <div className="flex flex-wrap items-center gap-2" role="group" aria-label="Pick a digit">
        <span className="eyebrow mr-1">Show me a</span>
        {Array.from({ length: 10 }, (_, d) => (
          <button
            key={d}
            type="button"
            aria-label={`Show a ${d}`}
            aria-pressed={s.sampleLabel === d}
            onClick={() => s.loadDigit(d)}
            className={cn(
              'size-8 cursor-pointer rounded-lg border font-mono text-sm transition-colors',
              s.sampleLabel === d
                ? 'border-brand bg-brand text-brand-fg'
                : 'border-line-strong text-muted hover:border-brand hover:text-fg',
            )}
          >
            {d}
          </button>
        ))}
        <Button size="sm" variant="secondary" onClick={s.randomSample}>
          <Dices className="size-4" /> Random
        </Button>
        <Button size="sm" variant="ghost" onClick={s.clear}>
          <Eraser className="size-4" /> Clear
        </Button>
      </div>

      <div className="flex flex-col items-center gap-6 lg:flex-row lg:items-start lg:justify-between">
        <figure className="space-y-2">
          <div className="relative">
            <PixelCanvas
              values={image}
              side={IMAGE_SIZE}
              size={PAD}
              grid
              onDown={(x, y) => s.beginStroke(x, y)}
              onMove={(x, y) => s.moveStroke(x, y)}
              onUp={s.endStroke}
              ariaLabel="Drawing pad: draw a digit with the mouse or your finger"
            />
            {source === 'blank' && (
              <p className="pointer-events-none absolute inset-0 grid place-items-center font-serif text-lg text-faint italic">
                Draw a digit here
              </p>
            )}
          </div>
          <figcaption className="eyebrow text-center">Original (draw on it)</figcaption>
        </figure>

        <ArrowRight
          aria-hidden
          className="hidden size-5 shrink-0 self-center text-faint lg:block"
        />

        <div className="w-full max-w-[18rem] space-y-3 self-center">
          <p className="eyebrow">
            Code · {config.latent} number{config.latent > 1 ? 's' : ''}
            {vae && <span className="text-faint"> (mean ± spread)</span>}
          </p>
          <ul className="space-y-1.5" aria-label="Code of the picture">
            {Array.from({ length: config.latent }, (_, d) => (
              <CodeBar
                key={d}
                label={`z${d + 1}`}
                value={code?.mu[d] ?? 0}
                spread={code?.logvar ? Math.exp(0.5 * (code.logvar[d] as number)) : 0}
                active={!blank}
              />
            ))}
          </ul>
          <p className="text-[0.74rem] leading-snug text-faint">
            {config.latent} numbers instead of {IMAGE_SIZE * IMAGE_SIZE} pixels: a{' '}
            {Math.round((IMAGE_SIZE * IMAGE_SIZE) / config.latent)}× compression.
          </p>
        </div>

        <ArrowRight
          aria-hidden
          className="hidden size-5 shrink-0 self-center text-faint lg:block"
        />

        <div className="flex gap-4">
          <figure className="space-y-2">
            <PixelCanvas
              values={blank ? new Float64Array(IMAGE_SIZE * IMAGE_SIZE) : recon}
              side={IMAGE_SIZE}
              size={SHOW}
              ariaLabel="Reconstruction painted by the decoder"
            />
            <figcaption className="eyebrow text-center">Reconstruction</figcaption>
          </figure>
          <figure className="space-y-2">
            <PixelCanvas
              values={error ?? new Float64Array(IMAGE_SIZE * IMAGE_SIZE)}
              side={IMAGE_SIZE}
              size={SHOW}
              mode="diverging"
              scale={0.6}
              ariaLabel="Difference between the reconstruction and the input"
            />
            <figcaption className="eyebrow text-center">
              Error {mse !== null && <span className="text-fg">{mse.toFixed(3)}</span>}
            </figcaption>
          </figure>
        </div>
      </div>

      <div className="grid gap-x-10 gap-y-4 sm:grid-cols-2">
        <div>
          <Slider
            label="Noise added to the input"
            min={0}
            max={0.6}
            step={0.05}
            value={s.inputNoise}
            onValueChange={s.setInputNoise}
            format={(v) => v.toFixed(2)}
          />
          <p className="mt-1.5 text-[0.74rem] leading-snug text-faint">
            {s.inputNoise > 0
              ? 'The encoder now receives the noisy picture shown on the right. The error map compares the reconstruction with the clean picture, so it tells you whether the noise was removed. Networks trained with noisy inputs learn to do that (see Training).'
              : 'Add noise to see whether the network copies its input or has learned what digits look like.'}
          </p>
        </div>
        <div className="space-y-3">
          <Switch
            label="Centre and scale my drawing"
            checked={s.center}
            onCheckedChange={s.setCenter}
          />
          {s.inputNoise > 0 && (
            <div className="flex items-center gap-3">
              <PixelCanvas
                values={input}
                side={IMAGE_SIZE}
                size={72}
                ariaLabel="Noisy input as the encoder reads it"
              />
              <span className="text-[0.74rem] text-faint">What the encoder receives</span>
            </div>
          )}
        </div>
      </div>

      {s.epoch === 0 && (
        <p className="rounded-xl border border-line bg-raised/40 px-4 py-3 text-[0.82rem] text-muted">
          This network has not been trained yet, so its reconstructions are blurry guesses. Try an
          experiment above, or scroll down to train it.
        </p>
      )}
    </div>
  );
}

function CodeBar({
  label,
  value,
  spread,
  active,
}: {
  label: string;
  value: number;
  spread: number;
  active: boolean;
}) {
  const RANGE = 4; // bars span ±4
  const pct = (v: number) =>
    `${((Math.max(-RANGE, Math.min(RANGE, v)) + RANGE) / (2 * RANGE)) * 100}%`;
  return (
    <li className="flex items-center gap-3">
      <span className="w-6 font-mono text-[0.72rem] text-muted">{label}</span>
      <div className="relative h-4 grow rounded bg-raised/70">
        <span className="absolute inset-y-0 left-1/2 w-px bg-line-strong" />
        {active && (
          <>
            {spread > 0 && (
              <span
                className="absolute inset-y-1 rounded bg-brand/25"
                style={{
                  left: pct(value - spread),
                  width: `calc(${pct(value + spread)} - ${pct(value - spread)})`,
                }}
              />
            )}
            <span
              className="absolute top-0 size-4 -translate-x-1/2 rounded-full bg-brand shadow transition-[left] duration-150"
              style={{ left: pct(value) }}
            />
          </>
        )}
      </div>
      <span className="w-12 text-right font-mono text-[0.72rem] text-fg tabular-nums">
        {active ? value.toFixed(2) : '–'}
      </span>
    </li>
  );
}
