import { Matrix } from '@sim/core';
import { nearestByCosine, pca, type BpeTokenizer } from '@sim/deep';
import { useMemo, useState } from 'react';
import { CATEGORY_COLOR, CATEGORY_LABEL, categoryOf, type Category } from '../categories';
import { visible } from '../tokens';

interface Props {
  /** Token embedding table (vocab × d). */
  embeddings: Matrix;
  tokenizer: BpeTokenizer;
  /** Tokens of the current text, always labelled on the map. */
  highlight: readonly number[];
}

const W = 560;
const H = 400;
const PAD = 26;
/** Ids below this are raw bytes that never occur on their own in the stories: leave them out. */
const FIRST_WORD_ID = 256;

/**
 * All word-level tokens placed in 2-D by PCA of their embeddings. Nothing told the model that
 * "Anna" and "Tom" are both names, yet similar tokens end up as neighbours.
 */
export function EmbeddingSpace({ embeddings, tokenizer, highlight }: Props) {
  const [hover, setHover] = useState<number | null>(null);
  const [showAll, setShowAll] = useState(false);

  const layout = useMemo(() => {
    const ids = Array.from(
      { length: embeddings.rows - FIRST_WORD_ID },
      (_, i) => FIRST_WORD_ID + i,
    ).filter((id) => {
      if (id === tokenizer.eotId) return false;
      // By default only real vocabulary words: pieces and rarely used tokens are pushed far away
      // by the tied output layer and would dominate the picture.
      return showAll || categoryOf(tokenizer.tokenString(id)) !== 'other';
    });
    const sub = new Matrix(ids.length, embeddings.cols);
    ids.forEach((id, r) =>
      sub.data.set(
        embeddings.data.subarray(id * embeddings.cols, (id + 1) * embeddings.cols),
        r * embeddings.cols,
      ),
    );
    const { projection, explained } = pca(sub, 2);
    // Robust bounds: a few barely-trained tokens sit far away and would squash everything else, so
    // the frame covers the 3rd-97th percentile and outliers are pinned to the border.
    const bounds = (axis: 0 | 1): [number, number] => {
      const values = Array.from({ length: ids.length }, (_, r) => projection.get(r, axis)).sort(
        (a, b) => a - b,
      );
      const at = (q: number) =>
        values[
          Math.min(values.length - 1, Math.max(0, Math.round(q * (values.length - 1))))
        ] as number;
      return [at(0.03), at(0.97)];
    };
    const [minX, maxX] = bounds(0);
    const [minY, maxY] = bounds(1);
    const clamp01 = (t: number) => Math.min(1, Math.max(0, t));
    const points = ids.map((id, r) => ({
      id,
      row: r,
      x: PAD + clamp01((projection.get(r, 0) - minX) / (maxX - minX || 1)) * (W - 2 * PAD),
      y: H - PAD - clamp01((projection.get(r, 1) - minY) / (maxY - minY || 1)) * (H - 2 * PAD),
      category: categoryOf(tokenizer.tokenString(id)),
    }));
    return { ids, sub, points, explained };
  }, [embeddings, tokenizer, showAll]);

  const byId = useMemo(() => new Map(layout.points.map((p) => [p.id, p])), [layout.points]);
  const neighbours = useMemo(() => {
    if (hover === null) return [];
    const point = byId.get(hover);
    if (!point) return [];
    return nearestByCosine(layout.sub, point.row, 5).map((n) => ({
      point: layout.points[n.index]!,
      similarity: n.similarity,
    }));
  }, [hover, byId, layout]);

  const categories = useMemo(() => {
    const seen = new Set<Category>(layout.points.map((p) => p.category));
    return (Object.keys(CATEGORY_LABEL) as Category[]).filter((c) => seen.has(c));
  }, [layout.points]);

  const labelled = new Set<number>(highlight.filter((id) => byId.has(id)));
  if (hover !== null) labelled.add(hover);
  for (const n of neighbours) labelled.add(n.point.id);

  return (
    <div>
      <label className="mb-3 flex cursor-pointer items-center gap-2 text-[0.78rem] text-muted">
        <input
          type="checkbox"
          checked={showAll}
          onChange={(e) => setShowAll(e.target.checked)}
          className="accent-[var(--brand)]"
        />
        include word pieces and rarely used tokens
      </label>
      <ul className="mb-3 flex flex-wrap gap-x-4 gap-y-1.5" aria-label="Legend">
        {categories.map((c) => (
          <li key={c} className="flex items-center gap-1.5 font-mono text-[0.68rem] text-muted">
            <span className="size-2 rounded-full" style={{ background: CATEGORY_COLOR[c] }} />
            {CATEGORY_LABEL[c]}
          </li>
        ))}
      </ul>

      <svg
        viewBox={`0 0 ${W} ${H}`}
        className="w-full rounded-xl border border-line bg-raised/30"
        role="img"
        aria-label="Map of the token embeddings, coloured by word type"
        onPointerLeave={() => setHover(null)}
      >
        {neighbours.map(({ point }) => {
          const origin = byId.get(hover as number)!;
          return (
            <line
              key={point.id}
              x1={origin.x}
              y1={origin.y}
              x2={point.x}
              y2={point.y}
              stroke="var(--brand)"
              strokeOpacity={0.55}
              strokeDasharray="3 3"
            />
          );
        })}
        {layout.points.map((p) => {
          const isHighlight = labelled.has(p.id);
          return (
            <circle
              key={p.id}
              cx={p.x}
              cy={p.y}
              r={p.id === hover ? 7 : isHighlight ? 5.5 : 3.6}
              fill={CATEGORY_COLOR[p.category]}
              fillOpacity={hover !== null && !isHighlight ? 0.28 : 0.9}
              stroke={isHighlight ? 'var(--fg)' : 'none'}
              strokeWidth={1.3}
              onPointerEnter={() => setHover(p.id)}
              style={{ cursor: 'pointer', transition: 'r 0.15s, fill-opacity 0.2s' }}
            />
          );
        })}
        {[...labelled].map((id) => {
          const p = byId.get(id);
          if (!p) return null;
          const right = p.x < W - 120;
          return (
            <text
              key={`l-${id}`}
              x={p.x + (right ? 9 : -9)}
              y={p.y + 4}
              textAnchor={right ? 'start' : 'end'}
              className="pointer-events-none fill-fg font-mono text-[11px]"
              style={{ paintOrder: 'stroke', stroke: 'var(--bg)', strokeWidth: 3 }}
            >
              {visible(tokenizer.tokenString(id)).replace('·', '')}
            </text>
          );
        })}
      </svg>

      <div className="mt-3 min-h-[3.2rem] text-[0.8rem] text-muted">
        {hover !== null && neighbours.length > 0 ? (
          <p>
            Closest to{' '}
            <span className="font-mono text-fg">{tokenizer.tokenString(hover).trim()}</span>:{' '}
            {neighbours.map((n, i) => (
              <span key={n.point.id}>
                {i > 0 && ', '}
                <span className="font-mono text-fg">
                  {tokenizer.tokenString(n.point.id).trim()}
                </span>{' '}
                <span className="text-faint">{n.similarity.toFixed(2)}</span>
              </span>
            ))}
          </p>
        ) : (
          <p>
            Two axes can show only{' '}
            {(((layout.explained[0] ?? 0) + (layout.explained[1] ?? 0)) * 100).toFixed(0)}% of how
            the {embeddings.cols} dimensions differ. Hover a token to see its nearest neighbours in
            the full space (cosine similarity).
          </p>
        )}
      </div>
    </div>
  );
}
