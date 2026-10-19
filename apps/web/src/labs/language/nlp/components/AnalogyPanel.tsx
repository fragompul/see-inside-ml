import { ALL_ANALOGIES } from '@sim/deep';
import { Select, cn } from '@/design-system';
import { CONTENT_WORDS } from '../colors';
import { useNlpStore } from '../store';
import { EmbeddingMap } from './EmbeddingMap';

const options = CONTENT_WORDS.map((w) => ({ value: w, label: w }));

const PRESETS: { label: string; q: [string, string, string] }[] = [
  { label: 'man : king :: woman : ?', q: ['man', 'king', 'woman'] },
  { label: 'france : paris :: italy : ?', q: ['france', 'paris', 'italy'] },
  { label: 'japan : tokyo :: egypt : ?', q: ['japan', 'tokyo', 'egypt'] },
  { label: 'boy : girl :: man : ?', q: ['boy', 'girl', 'man'] },
];

/** "a is to b as c is to ?": vector arithmetic on the embeddings, drawn on the map. */
export function AnalogyPanel() {
  const { map, query, answers, groups } = useNlpStore();
  const setQuery = useNlpStore((s) => s.setQuery);
  const top = answers[0]?.word;
  const expected = ALL_ANALOGIES.find(
    (q) => q[0] === query.a && q[1] === query.b && q[2] === query.c,
  )?.[3];

  return (
    <div className="space-y-6">
      <p className="max-w-3xl text-[0.9rem] text-muted">
        If directions in the vector space mean something, then the step from <em>man</em> to{' '}
        <em>king</em> should be the same arrow as from <em>woman</em> to <em>queen</em>. So to
        answer &ldquo;a is to b as c is to ?&rdquo;, compute{' '}
        <b className="font-mono font-medium text-fg">b − a + c</b> and look for the word nearest to
        the result.
      </p>

      <div className="flex flex-wrap gap-2" role="group" aria-label="Example analogies">
        {PRESETS.map((p) => (
          <button
            key={p.label}
            type="button"
            onClick={() => setQuery({ a: p.q[0], b: p.q[1], c: p.q[2] })}
            className={cn(
              'cursor-pointer rounded-full border px-3 py-1 font-mono text-[0.76rem] transition-colors',
              query.a === p.q[0] && query.b === p.q[1] && query.c === p.q[2]
                ? 'border-brand bg-brand text-brand-fg'
                : 'border-line-strong text-muted hover:border-brand hover:text-fg',
            )}
          >
            {p.label}
          </button>
        ))}
      </div>

      <div className="grid gap-8 lg:grid-cols-[minmax(0,40rem)_minmax(0,1fr)]">
        {map && (
          <EmbeddingMap
            map={map}
            arrows={[
              { from: query.a, to: query.b },
              ...(top ? [{ from: query.c, to: top, dashed: true }] : []),
            ]}
            marked={[query.a, query.b, query.c, ...(top ? [top] : [])]}
          />
        )}
        <div className="min-w-0 space-y-5">
          <div className="grid grid-cols-3 gap-3">
            <Select
              label="a"
              value={query.a}
              onValueChange={(a) => setQuery({ a })}
              options={options}
            />
            <Select
              label="b"
              value={query.b}
              onValueChange={(b) => setQuery({ b })}
              options={options}
            />
            <Select
              label="c"
              value={query.c}
              onValueChange={(c) => setQuery({ c })}
              options={options}
            />
          </div>
          <div className="rounded-xl border border-line bg-raised/30 px-4 py-4">
            <p className="font-mono text-[0.84rem] text-muted">
              {query.a} : {query.b} :: {query.c} : <span className="text-fg">?</span>
            </p>
            <p className="mt-1 font-display text-4xl font-semibold" data-testid="analogy-answer">
              {top ?? '–'}
            </p>
            {expected && (
              <p className={cn('mt-1 text-[0.8rem]', top === expected ? 'text-ok' : 'text-neg')}>
                {top === expected ? 'Correct' : `Expected ${expected}`}
              </p>
            )}
            <ol
              className="mt-3 space-y-1 font-mono text-[0.8rem] text-muted"
              aria-label="Candidate answers"
            >
              {answers.map((n) => (
                <li key={n.word} className="flex justify-between gap-4">
                  <span className={n.word === top ? 'text-fg' : undefined}>{n.word}</span>
                  <span className="tabular-nums">{n.similarity.toFixed(2)}</span>
                </li>
              ))}
            </ol>
          </div>
          <p className="text-[0.76rem] leading-snug text-faint">
            The solid arrow is a → b; the dashed one is the same step applied to c. The words a, b
            and c are never allowed as the answer.
          </p>

          <div>
            <p className="eyebrow mb-2">Accuracy on {ALL_ANALOGIES.length} analogy questions</p>
            <ul className="space-y-2" aria-label="Analogy accuracy by kind">
              {groups.map((g) => (
                <li key={g.id} className="flex items-center gap-3">
                  <span className="w-44 shrink-0 text-[0.76rem] text-muted">{g.label}</span>
                  <div className="h-3 grow overflow-hidden rounded bg-raised/70">
                    <div
                      className="h-full rounded bg-brand transition-[width] duration-300"
                      style={{ width: `${g.accuracy * 100}%` }}
                    />
                  </div>
                  <span className="w-10 text-right font-mono text-[0.74rem] text-muted tabular-nums">
                    {Math.round(g.accuracy * 100)}%
                  </span>
                </li>
              ))}
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
}
