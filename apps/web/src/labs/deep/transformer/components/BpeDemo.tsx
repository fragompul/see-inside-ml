import { type BpeTokenizer, type MergeStep } from '@sim/deep';
import { ChevronLeft, ChevronRight, Play } from 'lucide-react';
import { useEffect, useMemo, useState } from 'react';
import { Button, cn } from '@/design-system';
import { tokenHue, tokenLabel } from '../tokens';

const EXAMPLES = ['Zebra', 'unbelievable', 'cat', 'Anna', 'ball', 'café'];
const STEP_MS = 650;

/**
 * Byte-pair encoding, live. Type any word: it starts as raw bytes and the most frequent learned
 * pairs are fused one at a time. Words the model saw often collapse into a single token; words it
 * never saw stay in pieces.
 */
export function BpeDemo({ tokenizer }: { tokenizer: BpeTokenizer }) {
  const [word, setWord] = useState('Zebra');
  const [space, setSpace] = useState(true);
  const text = (space ? ' ' : '') + word;
  const steps = useMemo(
    () => (word.trim() === '' ? [] : tokenizer.mergeSteps(text)),
    [tokenizer, text, word],
  );

  return (
    <div className="rounded-xl border border-line bg-raised/30 p-4">
      <div className="flex flex-wrap items-end gap-x-6 gap-y-3">
        <div className="min-w-48 flex-1">
          <label htmlFor="bpe-word" className="mb-1.5 block text-[0.78rem] text-muted">
            Try a word, even one the model has never seen
          </label>
          <input
            id="bpe-word"
            value={word}
            maxLength={24}
            spellCheck={false}
            onChange={(e) => setWord(e.target.value.replace(/\s/g, ''))}
            className="h-9 w-full rounded-lg border border-line-strong bg-raised/50 px-3 font-mono text-[0.9rem] text-fg transition-colors outline-none focus:border-brand"
          />
        </div>
        <label className="flex cursor-pointer items-center gap-2 pb-2 text-[0.78rem] text-muted">
          <input
            type="checkbox"
            checked={space}
            onChange={(e) => setSpace(e.target.checked)}
            className="accent-[var(--brand)]"
          />
          leading space
        </label>
      </div>

      <div className="mt-2 flex flex-wrap gap-1.5">
        {EXAMPLES.map((example) => (
          <button
            key={example}
            type="button"
            onClick={() => setWord(example)}
            className="cursor-pointer rounded-full border border-line-strong px-2.5 py-0.5 font-mono text-[0.72rem] text-muted transition-colors hover:border-brand hover:text-fg"
          >
            {example}
          </button>
        ))}
      </div>

      {steps.length > 0 ? (
        // Keyed by the text: typing a new word mounts a fresh player at step 0.
        <MergePlayer key={text} tokenizer={tokenizer} steps={steps} length={text.length} />
      ) : (
        <p className="mt-4 text-sm text-muted">Type a word to watch it being tokenized.</p>
      )}
    </div>
  );
}

function MergePlayer({
  tokenizer,
  steps,
  length,
}: {
  tokenizer: BpeTokenizer;
  steps: MergeStep[];
  length: number;
}) {
  const [step, setStep] = useState(0);
  const [playing, setPlaying] = useState(false);
  const last = steps.length - 1;
  const shown = Math.min(step, last);
  const current = steps[shown] as MergeStep;
  const final = (steps[last] as MergeStep).ids;

  useEffect(() => {
    if (!playing) return;
    const timer = window.setInterval(() => {
      setStep((s) => {
        if (s >= last) {
          setPlaying(false);
          return s;
        }
        return s + 1;
      });
    }, STEP_MS);
    return () => window.clearInterval(timer);
  }, [playing, last]);

  const label = (id: number) => tokenLabel(tokenizer.tokenString(id), id);
  const go = (to: number) => {
    setPlaying(false);
    setStep(Math.max(0, Math.min(last, to)));
  };

  return (
    <div className="mt-5">
      <div className="flex min-h-[3.6rem] flex-wrap items-start gap-1.5" aria-live="polite">
        {current.ids.map((id, i) => {
          const fresh = current.merged?.into === id;
          return (
            <span key={`${shown}-${i}`} className="flex flex-col items-center gap-0.5">
              <span
                className={cn(
                  'tok rounded-md px-2 py-1 font-mono text-[0.95rem] whitespace-pre transition-all',
                  fresh && 'tok-solid text-bg shadow-[0_0_0_3px_var(--brand-soft)]',
                )}
                style={{ ['--h' as string]: tokenHue(id) }}
              >
                {label(id)}
              </span>
              <span className="font-mono text-[0.6rem] text-faint tabular-nums">{id}</span>
            </span>
          );
        })}
      </div>

      <p className="mt-3 min-h-5 text-[0.82rem] text-muted">
        {current.merged ? (
          <>
            Merge {shown}: <span className="font-mono text-fg">{label(current.merged.left)}</span> +{' '}
            <span className="font-mono text-fg">{label(current.merged.right)}</span> →{' '}
            <span className="font-mono text-fg">{label(current.merged.into)}</span>{' '}
            <span className="text-faint">(learned merge #{current.merged.into - 255})</span>
          </>
        ) : (
          <>
            Start: {current.ids.length} raw bytes (one token per byte, so any text can be encoded).
          </>
        )}
      </p>

      <div className="mt-3 flex flex-wrap items-center gap-2">
        <Button
          size="sm"
          variant="secondary"
          onClick={() => go(shown - 1)}
          disabled={shown === 0}
          aria-label="Previous merge"
        >
          <ChevronLeft className="size-4" />
        </Button>
        <Button
          size="sm"
          variant="secondary"
          onClick={() => go(shown + 1)}
          disabled={shown === last}
          aria-label="Next merge"
        >
          <ChevronRight className="size-4" />
        </Button>
        <Button
          size="sm"
          variant="ghost"
          onClick={() => {
            setStep(0);
            setPlaying(true);
          }}
        >
          <Play className="size-3.5" /> Replay
        </Button>
        <p className="ml-auto font-mono text-[0.72rem] text-muted">
          {final.length === 1
            ? 'a single token: the model knows this word'
            : `${final.length} tokens: ${final.length > length / 2 ? 'mostly unseen letters' : 'a few familiar pieces'}`}
        </p>
      </div>
    </div>
  );
}
