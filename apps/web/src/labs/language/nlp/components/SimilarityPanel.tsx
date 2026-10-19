import { useState } from 'react';
import { Select } from '@/design-system';
import { CONTENT_WORDS } from '../colors';
import { useNlpStore } from '../store';

const options = CONTENT_WORDS.map((w) => ({ value: w, label: w }));

/** Nearest neighbours of a word, and the cosine similarity of any two words. */
export function SimilarityPanel() {
  // subscribing to `version` re-renders the panel with every publish, so it follows training
  const { selected, neighbours } = useNlpStore();
  const select = useNlpStore((s) => s.select);
  const similarity = useNlpStore((s) => s.similarity);
  const [other, setOther] = useState('queen');
  const value = similarity(selected, other);

  return (
    <div className="space-y-8">
      <p className="max-w-3xl text-[0.9rem] text-muted">
        Closeness is measured with <em>cosine similarity</em>: the angle between two vectors, from 1
        (pointing the same way) through 0 (unrelated) to −1 (opposite). Pick a word to see which
        words the model considers its closest relatives.
      </p>
      <div className="grid gap-10 lg:grid-cols-2">
        <div className="space-y-4">
          <Select label="Word" value={selected} onValueChange={select} options={options} />
          <ol className="space-y-2" aria-label={`Nearest words to ${selected}`}>
            {neighbours.map((n) => (
              <li key={n.word} className="flex items-center gap-3">
                <button
                  type="button"
                  onClick={() => select(n.word)}
                  className="w-20 cursor-pointer text-left font-mono text-[0.88rem] text-fg hover:underline"
                >
                  {n.word}
                </button>
                <div className="relative h-4 grow overflow-hidden rounded bg-raised/70">
                  <div
                    className="absolute inset-y-0 rounded transition-[width,left] duration-200"
                    style={
                      n.similarity >= 0
                        ? {
                            left: '50%',
                            width: `${n.similarity * 50}%`,
                            background: 'var(--brand)',
                          }
                        : {
                            left: `${50 + n.similarity * 50}%`,
                            width: `${-n.similarity * 50}%`,
                            background: 'var(--neg)',
                          }
                    }
                  />
                  <span className="absolute inset-y-0 left-1/2 w-px bg-line-strong" />
                </div>
                <span className="w-12 text-right font-mono text-[0.76rem] text-muted tabular-nums">
                  {n.similarity.toFixed(2)}
                </span>
              </li>
            ))}
          </ol>
        </div>

        <div className="space-y-4">
          <p className="eyebrow">Compare two words</p>
          <div className="grid grid-cols-2 gap-4">
            <Select label="First" value={selected} onValueChange={select} options={options} />
            <Select label="Second" value={other} onValueChange={setOther} options={options} />
          </div>
          <div className="rounded-xl border border-line bg-raised/30 px-4 py-4">
            <p className="font-mono text-[0.9rem] text-muted">
              similarity(<span className="text-fg">{selected}</span>,{' '}
              <span className="text-fg">{other}</span>)
            </p>
            <p
              className="mt-1 font-display text-4xl font-semibold tabular-nums"
              data-testid="pair-similarity"
            >
              {value.toFixed(2)}
            </p>
            <div className="relative mt-3 h-2 rounded-full bg-raised">
              <span className="absolute inset-y-0 left-1/2 w-px bg-line-strong" />
              <span
                className="absolute top-1/2 size-4 -translate-x-1/2 -translate-y-1/2 rounded-full bg-brand shadow transition-[left] duration-200"
                style={{ left: `${(value + 1) * 50}%` }}
              />
            </div>
            <div className="mt-1 flex justify-between font-mono text-[0.64rem] text-faint">
              <span>−1 opposite</span>
              <span>0 unrelated</span>
              <span>1 same</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
