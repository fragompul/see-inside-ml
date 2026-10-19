import { Ruler } from 'lucide-react';
import { Button, cn } from '@/design-system';
import { useRnnStore } from '../store';

/** How the trained network copes with sequences longer or shorter than the ones it practised on. */
export function LengthPanel() {
  const sweep = useRnnStore((s) => s.lengthSweep);
  const trained = useRnnStore((s) => s.config.seqLen);
  const epoch = useRnnStore((s) => s.epoch);
  const measure = useRnnStore((s) => s.measureLengths);

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <p className="max-w-2xl text-[0.86rem] text-muted">
          The weights are shared across steps, so the same network can read sequences of any length.
          Does what it learned carry over? Measure its accuracy on fresh sequences of other lengths.
        </p>
        <Button size="sm" variant="secondary" onClick={measure} disabled={epoch === 0}>
          <Ruler className="size-4" /> Measure other lengths
        </Button>
      </div>
      {epoch === 0 && <p className="text-[0.78rem] text-faint">Train the network first.</p>}
      {sweep && (
        <div
          className="flex items-end gap-2 sm:gap-3"
          role="img"
          aria-label="Accuracy for each sequence length"
        >
          {sweep.map(({ length, accuracy }) => (
            <div key={length} className="flex min-w-0 flex-1 flex-col items-center gap-1.5">
              <span className="font-mono text-[0.7rem] text-muted tabular-nums">
                {Math.round(accuracy * 100)}%
              </span>
              <div className="flex h-28 w-full items-end rounded-md bg-raised/60">
                <div
                  className={cn(
                    'w-full rounded-md transition-[height] duration-300',
                    length === trained ? 'bg-brand' : 'bg-line-strong',
                  )}
                  style={{ height: `${Math.max(2, accuracy * 100)}%` }}
                />
              </div>
              <span
                className={cn(
                  'font-mono text-[0.7rem]',
                  length === trained ? 'text-fg' : 'text-faint',
                )}
              >
                {length}
              </span>
            </div>
          ))}
        </div>
      )}
      {sweep && (
        <p className="text-[0.76rem] text-faint">
          Purple = the length it was trained on. Accuracy usually falls away from it: a network only
          reliably learns what its training data demanded.
        </p>
      )}
    </div>
  );
}
