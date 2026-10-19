import { type ForwardTrace } from '@sim/deep';
import { useMemo, useState } from 'react';
import { Segmented, Tex, cn } from '@/design-system';
import { tokenHue, visible } from '../tokens';
import { AttentionArcs } from './AttentionArcs';
import { AttentionMatrix, type CellInfo } from './AttentionMatrix';

interface Props {
  trace: ForwardTrace;
  /** Display string per token of the visible window. */
  labels: string[];
  headDim: number;
  layer: number;
  head: number;
  /** Query position in focus. */
  position: number;
  onLayer: (layer: number) => void;
  onHead: (head: number) => void;
  onPosition: (position: number) => void;
}

const pct = (w: number) => (w >= 0.995 ? '100%' : w >= 0.01 ? `${(w * 100).toFixed(0)}%` : '<1%');

/**
 * Self-attention made visible: which earlier tokens does each token read from, per head.
 * Every row of the matrix is a probability distribution over the tokens at or before it.
 */
export function AttentionPanel({
  trace,
  labels,
  headDim,
  layer,
  head,
  position,
  onLayer,
  onHead,
  onPosition,
}: Props) {
  const [hover, setHover] = useState<CellInfo | null>(null);
  const block = trace.layers[layer];
  const heads = block?.probs.length ?? 0;
  const probs = block?.probs[head];
  const T = trace.ids.length;
  const query = Math.min(position, T - 1);

  const ranked = useMemo(() => {
    if (!probs) return [];
    return Array.from({ length: query + 1 }, (_, j) => ({ j, w: probs.get(query, j) }))
      .sort((a, b) => b.w - a.w)
      .slice(0, 5);
  }, [probs, query]);

  if (!block || !probs) return null;

  const cellInfo = hover ?? { query, key: ranked[0]?.j ?? query };
  const rawScore =
    (block.scores[head]?.get(cellInfo.query, cellInfo.key) ?? 0) * Math.sqrt(headDim);
  const scaled = block.scores[head]?.get(cellInfo.query, cellInfo.key) ?? 0;
  const weight = probs.get(cellInfo.query, cellInfo.key);

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-center justify-between gap-4">
        <Segmented
          label="Layer"
          size="sm"
          fit
          value={String(layer)}
          onValueChange={(v) => onLayer(Number(v))}
          options={trace.layers.map((_, i) => ({ value: String(i), label: `Layer ${i + 1}` }))}
        />
        <p className="max-w-xl text-[0.85rem] text-muted">
          Each head learns its own way of looking back. Compare them: some track the previous word,
          some jump to names, some barely move.
        </p>
      </div>

      <div
        className="grid grid-cols-2 gap-3 sm:grid-cols-4"
        role="group"
        aria-label="Attention heads"
      >
        {Array.from({ length: heads }, (_, h) => (
          <button
            key={h}
            type="button"
            aria-pressed={h === head}
            onClick={() => onHead(h)}
            className={cn(
              'cursor-pointer rounded-xl border p-2.5 text-left transition-all duration-200',
              h === head
                ? 'border-brand bg-brand-soft shadow-[0_0_0_3px_var(--brand-soft)]'
                : 'border-line bg-raised/30 hover:border-line-strong',
            )}
          >
            <p className="mb-2 flex items-baseline justify-between font-mono text-[0.7rem]">
              <span className={h === head ? 'text-fg' : 'text-muted'}>Head {h + 1}</span>
              <span className="text-muted">→ {labelOfTop(block.probs[h]!, query, labels)}</span>
            </p>
            <AttentionMatrix
              probs={block.probs[h]!}
              labels={labels}
              selected={query}
              onSelect={onPosition}
              compact
              maxCell={9}
            />
          </button>
        ))}
      </div>

      <div>
        <p className="eyebrow mb-3">
          Where “<span className="normal-case">{visible(labels[query] ?? '')}</span>” looks · layer{' '}
          {layer + 1}, head {head + 1}
        </p>
        <AttentionArcs
          probs={probs}
          labels={labels}
          ids={trace.ids}
          query={query}
          onSelect={onPosition}
        />
      </div>

      <div className="grid grid-cols-[minmax(0,1fr)] gap-8 lg:grid-cols-[minmax(0,1.2fr)_minmax(0,1fr)]">
        <div>
          <p className="eyebrow mb-3">
            Layer {layer + 1} · head {head + 1}
          </p>
          <AttentionMatrix
            probs={probs}
            labels={labels}
            selected={query}
            onSelect={onPosition}
            onHover={setHover}
          />
          <p className="mt-3 text-[0.75rem] text-muted">
            Rows: the token that is reading. Columns: what it can read. The hatched triangle is the
            causal mask. Click a row to focus a token.
          </p>
        </div>

        <div className="space-y-5">
          <div>
            <p className="eyebrow mb-3">
              “<span className="normal-case">{visible(labels[query] ?? '')}</span>” reads mostly
              from
            </p>
            <ul className="space-y-2">
              {ranked.map(({ j, w }) => (
                <li
                  key={j}
                  className="grid grid-cols-[minmax(4rem,7rem)_1fr_2.6rem] items-center gap-3"
                >
                  <span
                    className="tok truncate rounded-md px-1.5 py-0.5 font-mono text-[0.8rem] whitespace-pre"
                    style={{ ['--h' as string]: tokenHue(j) }}
                  >
                    {visible(labels[j] ?? '')}
                  </span>
                  <span className="h-2 overflow-hidden rounded-full bg-line">
                    <span
                      className="block h-full rounded-full bg-brand transition-[width] duration-300"
                      style={{ width: `${Math.max(2, w * 100)}%` }}
                    />
                  </span>
                  <span className="text-right font-mono text-[0.76rem] tabular-nums">{pct(w)}</span>
                </li>
              ))}
            </ul>
          </div>

          <div className="rounded-xl border border-line bg-raised/40 p-4">
            <p className="eyebrow mb-2">{hover ? 'Cell under the cursor' : 'Strongest link'}</p>
            <p className="text-[0.82rem] text-muted">
              <span className="font-mono text-fg">{visible(labels[cellInfo.query] ?? '')}</span>{' '}
              asks a question (its <em>query</em>);{' '}
              <span className="font-mono text-fg">{visible(labels[cellInfo.key] ?? '')}</span>{' '}
              offers an answer (its <em>key</em>). Their dot product says how well they match:
            </p>
            <div className="mt-2 overflow-x-auto text-[0.9rem] text-fg">
              <Tex
                block
                tex={`\\frac{q\\cdot k}{\\sqrt{d_h}} = \\frac{${fix(rawScore)}}{\\sqrt{${headDim}}} = ${fix(scaled)}\\;\\xrightarrow{\\text{softmax}}\\; ${pct(weight).replace('%', '\\%')}`}
              />
            </div>
          </div>

          <div className="overflow-x-auto text-[0.82rem] text-muted">
            <Tex
              tex="\mathrm{Attention}(Q,K,V)=\mathrm{softmax}\!\left(\frac{QK^{\top}}{\sqrt{d_h}}+\text{mask}\right)V"
              block
            />
            <p className="mt-1">
              The softmax turns the scores of each row into weights that sum to 100%; the head then
              returns the weighted average of the tokens' <em>values</em>.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}

const fix = (v: number) => (Math.abs(v) < 0.005 ? '0' : v.toFixed(2).replace('-', '-'));

/** Label of the token that a head attends to most from `query` (tiny summary above each head). */
function labelOfTop(
  probs: ForwardTrace['layers'][number]['probs'][number],
  query: number,
  labels: string[],
): string {
  let best = 0;
  for (let j = 1; j <= query; j++) if (probs.get(query, j) > probs.get(query, best)) best = j;
  return (
    visible(labels[best] ?? '')
      .trim()
      .slice(0, 8) || '·'
  );
}
