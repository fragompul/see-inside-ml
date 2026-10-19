import { type BpeTokenizer, type NextTokenDistribution } from '@sim/deep';
import { cn } from '@/design-system';
import { visible } from '../tokens';

const ROWS = 14;

const pct = (p: number) =>
  p >= 0.1 ? `${(p * 100).toFixed(0)}%` : p >= 0.001 ? `${(p * 100).toFixed(1)}%` : '<0.1%';

interface Props {
  distribution: NextTokenDistribution;
  tokenizer: BpeTokenizer;
  /** Called when the user clicks a candidate: they overrule the sampler. */
  onPick: (id: number) => void;
  /** What the prediction is for, e.g. "after “ She”". */
  context: string;
  vocabSize: number;
}

/**
 * The model's actual answer: a probability for every token in the vocabulary. Bars show what
 * the sampler will draw from after temperature, top-k and top-p; dimmed bars were filtered out.
 */
export function NextTokenPanel({ distribution, tokenizer, onPick, context, vocabSize }: Props) {
  const rows = distribution.candidates.slice(0, ROWS);
  const top = rows[0]?.prob ?? 1;
  const filtered = distribution.keptCount < vocabSize;
  const shownMass = rows.reduce((s, c) => s + c.prob, 0);

  return (
    <div>
      <div className="mb-4 flex flex-wrap items-baseline justify-between gap-x-6 gap-y-1">
        <p className="text-[0.85rem] text-muted">
          Next-token probabilities <span className="text-fg">{context}</span>
        </p>
        <p className="font-mono text-[0.72rem] text-muted tabular-nums">
          uncertainty {distribution.entropyBits.toFixed(2)} bits ·{' '}
          {filtered
            ? `${distribution.keptCount} of ${vocabSize} tokens kept`
            : `${vocabSize} candidates`}
        </p>
      </div>

      <ol className="space-y-1.5" aria-label="Most likely next tokens">
        {rows.map((c, rank) => (
          <li key={c.id}>
            <button
              type="button"
              onClick={() => onPick(c.id)}
              title={`logit ${c.logit.toFixed(2)} · click to choose this token`}
              className={cn(
                'group grid w-full cursor-pointer grid-cols-[1.5rem_minmax(5rem,9rem)_1fr_3.2rem] items-center gap-3 rounded-lg px-2 py-1 text-left transition-colors hover:bg-brand-soft',
                !c.kept && 'opacity-45',
              )}
            >
              <span className="font-mono text-[0.68rem] text-faint tabular-nums">{rank + 1}</span>
              <span
                className={cn(
                  'tok truncate rounded-md px-1.5 py-0.5 font-mono text-[0.84rem] whitespace-pre',
                  !c.kept && 'line-through',
                )}
                style={{ ['--h' as string]: (c.id * 53) % 360 }}
              >
                {visible(tokenizer.tokenString(c.id))}
              </span>
              <span className="relative h-2.5 overflow-hidden rounded-full bg-line">
                <span
                  className={cn(
                    'absolute inset-y-0 left-0 rounded-full transition-[width] duration-300',
                    c.kept ? 'bg-gradient-to-r from-neg via-brand to-pos' : 'bg-faint',
                  )}
                  style={{ width: `${Math.max(1.5, (c.prob / top) * 100)}%` }}
                />
              </span>
              <span className="text-right font-mono text-[0.78rem] tabular-nums">
                {pct(filtered && c.kept ? c.adjusted : c.prob)}
              </span>
            </button>
          </li>
        ))}
      </ol>

      <p className="mt-3 font-mono text-[0.68rem] text-faint">
        top {ROWS} cover {(shownMass * 100).toFixed(1)}% of the probability
        {filtered ? ' · percentages are renormalised over the kept tokens' : ''} · click a row to
        choose it yourself
      </p>
    </div>
  );
}
