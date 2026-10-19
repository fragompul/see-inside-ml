import { type Matrix } from '@sim/core';
import { type BpeTokenizer, type ForwardTrace } from '@sim/deep';
import { useState } from 'react';
import { cn } from '@/design-system';
import { tokenHue, visible } from '../tokens';
import { EmbeddingSpace } from './EmbeddingSpace';
import { Heat, maxAbs } from '@/shared/components/Heat';

interface Props {
  trace: ForwardTrace;
  tokenizer: BpeTokenizer;
  tokenEmbeddings: Matrix;
  position: number;
  onPosition: (position: number) => void;
}

const row = (m: Matrix, i: number) => m.data.subarray(i * m.cols, (i + 1) * m.cols);

export function EmbeddingsPanel({
  trace,
  tokenizer,
  tokenEmbeddings,
  position,
  onPosition,
}: Props) {
  const [hoverDim, setHoverDim] = useState<number | null>(null);
  const d = trace.embed.cols;
  const T = trace.ids.length;
  const p = Math.min(position, T - 1);
  const tok = row(trace.tok, p);
  const pos = row(trace.pos, p);
  const sum = row(trace.embed, p);
  const scale = maxAbs(tok, pos, sum);
  const matrixScale = maxAbs(trace.embed.data);
  const label = tokenizer.tokenString(trace.ids[p] as number);

  const strip = (title: string, hint: string, values: ArrayLike<number>) => (
    <div>
      <div className="mb-1.5 flex items-baseline justify-between gap-3">
        <p className="text-[0.82rem] text-fg">{title}</p>
        <p className="font-mono text-[0.68rem] text-faint">{hint}</p>
      </div>
      <Heat
        values={values}
        cols={d}
        scale={scale}
        cellW={7}
        cellH={20}
        onHover={setHoverDim}
        ariaLabel={`${title}: ${d} numbers`}
      />
    </div>
  );

  return (
    <div className="grid grid-cols-[minmax(0,1fr)] gap-10 lg:grid-cols-[minmax(0,1.05fr)_minmax(0,1fr)]">
      <div className="space-y-6">
        <p className="text-[0.88rem] leading-relaxed text-muted">
          A token id means nothing to a neural network, so each token is looked up in a table of {d}
          -number vectors (its <em className="text-fg">embedding</em>). A second table adds the
          token's <em className="text-fg">position</em>, because attention alone cannot tell word
          order. Their sum is what enters the first block.
        </p>

        <div className="space-y-4 overflow-x-auto rounded-xl border border-line bg-raised/30 p-4">
          <p className="eyebrow">
            Token {p + 1}: <span className="normal-case text-fg">“{visible(label)}”</span>
          </p>
          {strip('Token embedding', `table[${trace.ids[p]}]`, tok)}
          <p className="text-center font-mono text-sm text-muted">+</p>
          {strip('Position embedding', `table[${p}]`, pos)}
          <p className="text-center font-mono text-sm text-muted">=</p>
          {strip('Input to block 1', 'x₀', sum)}
          <p className="h-4 font-mono text-[0.72rem] text-muted">
            {hoverDim !== null
              ? `dimension ${hoverDim}: token ${(tok[hoverDim] as number).toFixed(3)} + position ${(pos[hoverDim] as number).toFixed(3)} = ${(sum[hoverDim] as number).toFixed(3)}`
              : 'colours: ember = positive, frost = negative. Hover a cell.'}
          </p>
        </div>

        <div>
          <p className="eyebrow mb-3">Everything the first block receives</p>
          <div className="overflow-x-auto">
            <div className="inline-flex">
              <ul className="shrink-0 pr-2" aria-label="Token labels">
                {trace.ids.map((id, i) => (
                  <li key={i}>
                    <button
                      type="button"
                      onClick={() => onPosition(i)}
                      style={{ height: 16, ['--h' as string]: tokenHue(i) }}
                      className={cn(
                        'block w-20 cursor-pointer truncate pr-1 text-right font-mono text-[0.68rem] leading-4 whitespace-pre transition-colors',
                        i === p ? 'text-fg' : 'text-muted hover:text-fg',
                      )}
                    >
                      {visible(tokenizer.tokenString(id))}
                    </button>
                  </li>
                ))}
              </ul>
              <div className="relative">
                <Heat
                  values={trace.embed.data}
                  cols={d}
                  scale={matrixScale}
                  cellW={6}
                  cellH={16}
                  onSelect={(index) => onPosition(Math.floor(index / d))}
                  ariaLabel="Embedding of every token (rows) across all dimensions (columns)"
                />
                <span
                  aria-hidden
                  className="pointer-events-none absolute inset-x-0 rounded-sm border border-brand shadow-[0_0_0_2px_var(--brand-soft)]"
                  style={{ top: p * 16, height: 16 }}
                />
              </div>
            </div>
          </div>
        </div>
      </div>

      <div>
        <p className="mb-3 text-[0.88rem] leading-relaxed text-muted">
          <span className="text-fg">The embedding table as a map.</span> Nobody told the model which
          words are names or animals, but training pulled similar words together.
        </p>
        <EmbeddingSpace embeddings={tokenEmbeddings} tokenizer={tokenizer} highlight={trace.ids} />
      </div>
    </div>
  );
}
