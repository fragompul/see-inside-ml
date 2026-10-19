import { VOCABULARY } from '@sim/deep';
import { ChevronLeft, ChevronRight, Dices } from 'lucide-react';
import { useState } from 'react';
import { Button, cn } from '@/design-system';
import { useNlpStore } from '../store';

/** How a sentence becomes (centre word, context word) training pairs, plus random negatives. */
export function PairsPanel() {
  const sentence = useNlpStore((s) => s.sentence);
  const setSentence = useNlpStore((s) => s.setSentence);
  const config = useNlpStore((s) => s.config);
  const corpus = useNlpStore((s) => s.corpus);
  const pairCount = useNlpStore((s) => s.pairCount);
  return (
    <PairsDemo
      key={sentence}
      words={corpus()[sentence] ?? []}
      index={sentence}
      total={corpus().length}
      window={config.window}
      negatives={config.negatives}
      pairs={pairCount()}
      onMove={(delta) => setSentence(sentence + delta)}
      onRandom={() => setSentence(Math.floor(Math.random() * corpus().length))}
    />
  );
}

function PairsDemo({
  words,
  index,
  total,
  window,
  negatives,
  pairs,
  onMove,
  onRandom,
}: {
  words: string[];
  index: number;
  total: number;
  window: number;
  negatives: number;
  pairs: number;
  onMove: (delta: number) => void;
  onRandom: () => void;
}) {
  // the centre word resets with the sentence because this component is keyed on it
  const [centre, setCentre] = useState(Math.min(2, Math.max(0, words.length - 1)));
  const [negs, setNegs] = useState<string[]>(() => draw(negatives));

  const contexts = words
    .map((w, i) => ({ w, i, offset: i - centre }))
    .filter((c) => c.offset !== 0 && Math.abs(c.offset) <= window);

  return (
    <div className="space-y-6">
      <p className="max-w-3xl text-[0.9rem] text-muted">
        A word is defined by the company it keeps. To learn that, the model reads a corpus sentence
        by sentence, takes each word in turn as the <em>centre</em> and pairs it with every word
        within a few places on either side (the <em>context</em>). Those pairs are all the training
        data: no labels, no dictionary. This toy world has {total.toLocaleString('en-US')} sentences
        about kings and queens, capitals, animals and food, which give{' '}
        {pairs.toLocaleString('en-US')} pairs at the current window.
      </p>

      <div className="flex flex-wrap items-center gap-2">
        <Button
          size="sm"
          variant="secondary"
          aria-label="Previous sentence"
          onClick={() => onMove(-1)}
        >
          <ChevronLeft className="size-4" />
        </Button>
        <span className="font-mono text-[0.78rem] text-muted tabular-nums">
          sentence {index + 1} / {total}
        </span>
        <Button size="sm" variant="secondary" aria-label="Next sentence" onClick={() => onMove(1)}>
          <ChevronRight className="size-4" />
        </Button>
        <Button size="sm" variant="ghost" onClick={onRandom}>
          <Dices className="size-4" /> Random sentence
        </Button>
      </div>

      <div>
        <p className="eyebrow mb-3">Click a word to make it the centre (window = {window})</p>
        <ol className="flex flex-wrap gap-1.5" aria-label="Words of the sentence">
          {words.map((w, i) => {
            const offset = i - centre;
            const isCentre = offset === 0;
            const inWindow = !isCentre && Math.abs(offset) <= window;
            return (
              <li key={i}>
                <button
                  type="button"
                  aria-pressed={isCentre}
                  aria-label={`Word ${i + 1}: ${w}`}
                  onClick={() => setCentre(i)}
                  className={cn(
                    'cursor-pointer rounded-lg border px-3 py-1.5 font-mono text-[0.9rem] transition-colors',
                    isCentre
                      ? 'border-brand bg-brand text-brand-fg'
                      : inWindow
                        ? 'border-brand/60 bg-brand-soft text-fg'
                        : 'border-line-strong text-faint hover:text-fg',
                  )}
                >
                  {w}
                </button>
              </li>
            );
          })}
        </ol>
      </div>

      <div className="grid gap-8 md:grid-cols-2">
        <div>
          <p className="eyebrow mb-2">
            Positive pairs <span className="text-faint">(centre → real context)</span>
          </p>
          <ul className="space-y-1 font-mono text-[0.84rem]" aria-label="Positive training pairs">
            {contexts.length === 0 && <li className="text-muted">No words in the window.</li>}
            {contexts.map((c) => (
              <li key={c.i} className="flex items-center gap-2">
                <span className="text-fg">{words[centre]}</span>
                <span className="text-faint">→</span>
                <span className="text-fg">{c.w}</span>
                <span className="text-[0.7rem] text-faint">
                  {c.offset < 0 ? `${-c.offset} before` : `${c.offset} after`}
                </span>
              </li>
            ))}
          </ul>
        </div>
        <div>
          <p className="eyebrow mb-2">
            Negative samples <span className="text-faint">(centre ↛ random words)</span>
          </p>
          <ul
            className="flex flex-wrap gap-1.5 font-mono text-[0.84rem]"
            aria-label="Random negative words"
          >
            {negs.map((w, i) => (
              <li key={i} className="rounded-md border border-line-strong px-2 py-1 text-muted">
                {w}
              </li>
            ))}
          </ul>
          <p className="mt-2 text-[0.76rem] leading-snug text-faint">
            For every real pair the model is also shown {negatives} random words and must learn that{' '}
            <em>those</em> do not belong with the centre word. Without them it could score perfectly
            by making every vector identical.
          </p>
          <Button
            size="sm"
            variant="ghost"
            className="mt-2"
            onClick={() => setNegs(draw(negatives))}
          >
            <Dices className="size-4" /> Draw other random words
          </Button>
        </div>
      </div>
    </div>
  );
}

const draw = (n: number): string[] =>
  Array.from(
    { length: n },
    () => VOCABULARY[Math.floor(Math.random() * VOCABULARY.length)] as string,
  );
