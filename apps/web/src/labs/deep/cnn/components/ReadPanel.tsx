import { DIGIT_CLASSES, IMAGE_SIZE, argmax } from '@sim/deep';
import { Dices, Eraser } from 'lucide-react';
import { Button, Switch, cn } from '@/design-system';
import { effectiveImage, useCnnStore } from '../store';
import { PixelCanvas } from '@/shared/components/PixelCanvas';

const PAD_SIZE = 288;

/** Drawing pad, what the network sees and its answer as ten probability bars. */
export function ReadPanel() {
  const s = useCnnStore();
  const trace = s.trace;
  const probs = trace?.probs ?? [];
  const blank = s.source === 'blank';
  const best = probs.length ? argmax(probs) : -1;
  const input = effectiveImage(s.image, s.center);

  return (
    <div className="grid gap-8 lg:grid-cols-[auto_minmax(0,1fr)] lg:gap-10">
      <div className="flex flex-col items-center gap-4 lg:items-start">
        <div className="relative">
          <PixelCanvas
            values={s.image}
            side={IMAGE_SIZE}
            size={PAD_SIZE}
            grid
            overlay={s.showOcclusion ? s.occlusion : null}
            onDown={(x, y) => s.beginStroke(x, y)}
            onMove={(x, y) => s.moveStroke(x, y)}
            onUp={s.endStroke}
            ariaLabel="Drawing pad: draw a digit from 0 to 9 with the mouse or your finger"
          />
          {blank && (
            <p className="pointer-events-none absolute inset-0 grid place-items-center px-8 text-center font-serif text-xl text-faint italic">
              Draw a digit here
            </p>
          )}
        </div>
        <div className="flex flex-wrap justify-center gap-2 lg:justify-start">
          <Button size="sm" variant="secondary" onClick={s.randomSample}>
            <Dices className="size-4" /> Random test digit
          </Button>
          <Button size="sm" variant="ghost" onClick={s.clear}>
            <Eraser className="size-4" /> Clear
          </Button>
        </div>
        <div className="w-full max-w-[288px] space-y-3">
          <Switch
            label="Centre and scale my drawing"
            checked={s.center}
            onCheckedChange={s.setCenter}
          />
          <Switch
            label="Show what the decision depends on"
            checked={s.showOcclusion}
            onCheckedChange={s.setShowOcclusion}
          />
        </div>
      </div>

      <div className="min-w-0 space-y-5">
        <div className="flex items-end gap-5">
          <div>
            <p className="eyebrow mb-2">The network sees</p>
            <PixelCanvas
              values={input}
              side={IMAGE_SIZE}
              size={96}
              ariaLabel="Input image as the network reads it"
            />
          </div>
          <div className="min-w-0 pb-1">
            <p className="eyebrow mb-1">Its answer</p>
            {blank ? (
              <p className="font-serif text-2xl text-faint italic">Nothing to read yet</p>
            ) : (
              <p className="font-display text-5xl leading-none font-semibold tracking-tight tabular-nums">
                {best}
                <span className="ml-3 font-mono text-base font-normal text-muted">
                  {((probs[best] ?? 0) * 100).toFixed(0)}% sure
                </span>
              </p>
            )}
            {s.sampleLabel !== null && (
              <p className="mt-1.5 text-[0.8rem] text-muted">
                This test image is a <span className="font-mono text-fg">{s.sampleLabel}</span>
                {best === s.sampleLabel ? (
                  <span className="text-ok"> · correct</span>
                ) : (
                  <span className="text-neg"> · the network is wrong here</span>
                )}
              </p>
            )}
          </div>
        </div>

        <ol className="space-y-1.5" aria-label="Probability of each digit">
          {Array.from({ length: DIGIT_CLASSES }, (_, digit) => {
            const p = blank ? 0 : (probs[digit] ?? 0);
            const top = !blank && digit === best;
            return (
              <li key={digit} className="flex items-center gap-3">
                <span
                  className={cn(
                    'w-4 text-right font-mono text-sm tabular-nums',
                    top ? 'font-semibold text-fg' : 'text-muted',
                  )}
                >
                  {digit}
                </span>
                <div className="h-5 grow overflow-hidden rounded-md bg-raised/70">
                  <div
                    className={cn(
                      'h-full rounded-md transition-[width] duration-200 ease-out',
                      top ? 'bg-brand' : 'bg-line-strong',
                    )}
                    style={{ width: `${p * 100}%` }}
                  />
                </div>
                <span className="w-12 text-right font-mono text-[0.74rem] text-muted tabular-nums">
                  {(p * 100).toFixed(1)}%
                </span>
              </li>
            );
          })}
        </ol>
        {s.showOcclusion && !blank && (
          <p className="text-[0.78rem] text-muted">
            Ember marks the pixels that matter most: covering them with a small blank patch lowers
            the network&apos;s confidence in its answer the most.
          </p>
        )}
        {s.epoch === 0 && (
          <p className="rounded-xl border border-line bg-raised/40 px-4 py-3 text-[0.82rem] text-muted">
            This network has not learned anything yet, so its answers are guesses. Scroll down to
            train it, then come back and draw.
          </p>
        )}
      </div>
    </div>
  );
}
