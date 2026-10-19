import { IMAGE_SIZE } from '@sim/deep';
import { Sparkles } from 'lucide-react';
import { useMemo } from 'react';
import { Button, cn } from '@/design-system';
import { PixelCanvas } from '@/shared/components/PixelCanvas';
import { IMAGINED, STRIP_STEPS, useAeStore } from '../store';

const PIXELS = IMAGE_SIZE * IMAGE_SIZE;
const THUMB = 64;

/** Morphing between two digits (pixels vs codes) and images the decoder imagines from noise. */
export function ImaginePanel() {
  const { pair, strip, imagined, config, epoch } = useAeStore();
  const setPairDigit = useAeStore((s) => s.setPairDigit);
  const imagine = useAeStore((s) => s.imagine);
  const vae = config.mode === 'vae';

  const blend = useMemo(() => {
    if (!pair) return null;
    return Array.from({ length: STRIP_STEPS }, (_, k) => {
      const t = k / (STRIP_STEPS - 1);
      return Float64Array.from(
        pair.a.image,
        (v, i) => v * (1 - t) + (pair.b.image[i] as number) * t,
      );
    });
  }, [pair]);

  if (!pair || !strip) return null;

  return (
    <div className="space-y-8">
      <div className="space-y-4">
        <p className="max-w-3xl text-[0.9rem] text-muted">
          Blend two digits in pixel space and you get a ghostly double exposure. Walk from one code
          to the other and decode every step, and a good model shows <em>plausible digits</em> all
          along the way: it learned a map of what digits look like, not just a way to copy them.
        </p>
        <div className="grid gap-4 sm:grid-cols-2">
          <DigitPicker label="From" value={pair.a.label} onPick={(d) => setPairDigit('a', d)} />
          <DigitPicker label="To" value={pair.b.label} onPick={(d) => setPairDigit('b', d)} />
        </div>
        <Strip title="Fading pixels" images={blend} />
        <Strip
          title="Walking through the code"
          images={Array.from({ length: STRIP_STEPS }, (_, k) =>
            strip.subarray(k * PIXELS, (k + 1) * PIXELS),
          )}
          highlight
        />
      </div>

      <div className="space-y-4">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div className="max-w-2xl">
            <p className="eyebrow mb-1">Imagination</p>
            <p className="text-[0.9rem] text-muted">
              Draw {IMAGINED} random codes from a standard normal distribution and decode them.{' '}
              {vae
                ? 'A VAE was trained to make exactly these codes meaningful, so most results look like digits.'
                : 'A plain autoencoder never learned which codes are valid, so many results are smudges: only the regions it actually used decode well.'}
            </p>
          </div>
          <Button size="sm" variant="secondary" onClick={imagine}>
            <Sparkles className="size-4" /> Draw new codes
          </Button>
        </div>
        {epoch === 0 && (
          <p className="text-[0.78rem] text-faint">
            Train the model first to see anything but noise.
          </p>
        )}
        <ul
          className="grid grid-cols-3 gap-2 sm:grid-cols-4 lg:grid-cols-6"
          aria-label="Images imagined from random codes"
        >
          {Array.from({ length: IMAGINED }, (_, k) => (
            <li key={k}>
              <PixelCanvas
                values={
                  imagined
                    ? imagined.subarray(k * PIXELS, (k + 1) * PIXELS)
                    : new Float64Array(PIXELS)
                }
                side={IMAGE_SIZE}
                size={THUMB * 1.4}
                ariaLabel={`Imagined image ${k + 1}`}
                className="w-full"
              />
            </li>
          ))}
        </ul>
      </div>
    </div>
  );
}

function DigitPicker({
  label,
  value,
  onPick,
}: {
  label: string;
  value: number;
  onPick: (digit: number | null) => void;
}) {
  return (
    <div className="flex flex-wrap items-center gap-1.5" role="group" aria-label={`${label} digit`}>
      <span className="eyebrow mr-1 w-10">{label}</span>
      {Array.from({ length: 10 }, (_, d) => (
        <button
          key={d}
          type="button"
          aria-label={`${label} a ${d}`}
          aria-pressed={value === d}
          onClick={() => onPick(d)}
          className={cn(
            'size-7 cursor-pointer rounded-md border font-mono text-xs transition-colors',
            value === d
              ? 'border-brand bg-brand text-brand-fg'
              : 'border-line-strong text-muted hover:border-brand hover:text-fg',
          )}
        >
          {d}
        </button>
      ))}
      <button
        type="button"
        onClick={() => onPick(null)}
        className="cursor-pointer rounded-md border border-line-strong px-2 py-1 text-xs text-muted hover:border-brand hover:text-fg"
      >
        any
      </button>
    </div>
  );
}

function Strip({
  title,
  images,
  highlight = false,
}: {
  title: string;
  images: Float64Array[] | null;
  highlight?: boolean;
}) {
  if (!images) return null;
  return (
    <div>
      <p className="eyebrow mb-2">{title}</p>
      <ul className="flex gap-1.5 overflow-x-auto pb-1" aria-label={title}>
        {images.map((image, k) => (
          <li key={k} className="shrink-0">
            <PixelCanvas
              values={image}
              side={IMAGE_SIZE}
              size={THUMB}
              ariaLabel={`${title}, step ${k + 1} of ${images.length}`}
              className={cn(highlight && (k === 0 || k === images.length - 1) && 'border-brand')}
            />
          </li>
        ))}
      </ul>
    </div>
  );
}
