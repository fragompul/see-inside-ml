import { type Matrix } from '@sim/core';
import { type ForwardTrace, type NextTokenDistribution } from '@sim/deep';

/**
 * Geometry of the 3-D model, kept free of Three.js so it can be unit-tested.
 *
 * World layout: the residual stream is a tunnel running along -Z. Every stage of the forward pass
 * is a "slab" in the XY plane: one column per token (x) and one cell per vector dimension (y).
 * Attention heads sit under their stage as T × T matrices; the next-token probabilities stand as
 * bars at the far end.
 */

/** Horizontal size of one token column. */
export const COLUMN_W = 0.17;
/** Total height of every slab, whatever its number of rows (64 for the stream, 256 for the MLP). */
export const SLAB_H = 3.2;
/** Distance between consecutive slabs along Z. */
export const SLAB_GAP = 1.55;
/** Side of one attention-head matrix. */
export const HEAD_SIZE = 1.5;
export const HEAD_GAP = 0.3;
export const BARS = 10;
export const BAR_SPACING = 0.34;

export type SlabKind = 'stream' | 'delta' | 'hidden' | 'norm';

export interface Slab {
  id: string;
  title: string;
  kind: SlabKind;
  /** Transformer block this belongs to (undefined for the embedding and the final norm). */
  layer?: number;
  /** Number of dimensions (cells per token column). */
  rows: number;
  tokens: number;
  z: number;
  /** Token-major values, `tokens × rows`. */
  values: Float64Array;
}

export interface Stage {
  id: string;
  title: string;
  /** Slabs highlighted while this stage is active. */
  slabIds: string[];
  /** Whether this stage shows the attention heads of `layer`. */
  heads?: number;
  /** Whether this stage shows the output bars. */
  output?: boolean;
  /** Where the camera should look. */
  z: number;
  /** Short explanation shown beside the scene. */
  text: string;
}

const slab = (
  id: string,
  title: string,
  kind: SlabKind,
  z: number,
  matrix: Matrix,
  layer?: number,
): Slab => ({
  id,
  title,
  kind,
  layer,
  rows: matrix.cols,
  tokens: matrix.rows,
  z,
  values: matrix.data,
});

/** The stages of the forward pass as slabs, in order, front (z = 0) to back. */
export function buildSlabs(trace: ForwardTrace): Slab[] {
  const slabs: Slab[] = [];
  let index = 0;
  const next = () => -SLAB_GAP * index++;
  slabs.push(slab('embed', 'Token + position embedding', 'stream', next(), trace.embed));
  trace.layers.forEach((layer, l) => {
    const n = l + 1;
    slabs.push(
      slab(`attn-${l}`, `Attention output · layer ${n}`, 'delta', next(), layer.attnOut, l),
      slab(
        `mid-${l}`,
        `Residual stream after attention · layer ${n}`,
        'stream',
        next(),
        layer.residMid,
        l,
      ),
      slab(`mlp-${l}`, `MLP hidden (after GELU) · layer ${n}`, 'hidden', next(), layer.mlpAct, l),
      slab(`mlpout-${l}`, `MLP output · layer ${n}`, 'delta', next(), layer.mlpOut, l),
      slab(`out-${l}`, `Residual stream after block ${n}`, 'stream', next(), layer.residOut, l),
    );
  });
  slabs.push(slab('final', 'Final LayerNorm', 'norm', next(), trace.finalLn));
  return slabs;
}

/** z of the plane where the output bars stand (one gap behind the last slab). */
export function outputZ(slabs: readonly Slab[]): number {
  const last = slabs[slabs.length - 1];
  return (last?.z ?? 0) - SLAB_GAP;
}

/** Guided tour: one stage per idea, each pointing at the slabs it is about. */
export function buildStages(trace: ForwardTrace, slabs: readonly Slab[]): Stage[] {
  const zOf = (id: string) => slabs.find((s) => s.id === id)?.z ?? 0;
  const stages: Stage[] = [
    {
      id: 'embed',
      title: 'Embedding',
      slabIds: ['embed'],
      z: zOf('embed'),
      text: 'Every token becomes a column of numbers (its embedding plus a position vector). Each coloured cell is one number; ember is positive, frost negative.',
    },
  ];
  trace.layers.forEach((_, l) => {
    const n = l + 1;
    stages.push(
      {
        id: `attention-${l}`,
        title: `Attention · layer ${n}`,
        slabIds: [`attn-${l}`],
        heads: l,
        z: zOf(`attn-${l}`),
        text: `Four heads decide, for every token, which earlier tokens to read from (the small matrices underneath). What they gather is written into the slab as new numbers.`,
      },
      {
        id: `add-${l}`,
        title: `Add to the stream · layer ${n}`,
        slabIds: [`mid-${l}`],
        z: zOf(`mid-${l}`),
        text: 'The attention result is added to the token’s existing vector, not substituted for it. This running sum is the residual stream.',
      },
      {
        id: `mlp-${l}`,
        title: `MLP · layer ${n}`,
        slabIds: [`mlp-${l}`, `mlpout-${l}`],
        z: zOf(`mlp-${l}`),
        text: 'Each token is widened to 256 numbers, passed through GELU (most cells go quiet), and narrowed back to 64. No token looks at another here: it is per-token thinking.',
      },
      {
        id: `block-${l}`,
        title: `Block ${n} output`,
        slabIds: [`out-${l}`],
        z: zOf(`out-${l}`),
        text: 'The MLP result is added too. This vector enters the next block, or the output stage after the last one.',
      },
    );
  });
  stages.push(
    {
      id: 'final',
      title: 'Final LayerNorm',
      slabIds: ['final'],
      z: zOf('final'),
      text: 'One last normalisation prepares the vectors to be compared with the whole vocabulary.',
    },
    {
      id: 'output',
      title: 'Next-token probabilities',
      slabIds: [],
      output: true,
      z: outputZ(slabs),
      text: 'The last token’s vector is compared with every token’s embedding; softmax turns those scores into the probabilities standing here. The tallest bar is the model’s best guess.',
    },
  );
  return stages;
}

// ───────────────────────────── instancing helpers ─────────────────────────────

export interface CellPosition {
  x: number;
  y: number;
  z: number;
  w: number;
  h: number;
}

/** Row height for a slab: more rows means thinner cells so every slab is SLAB_H tall. */
export const rowHeight = (rows: number) => SLAB_H / rows;

/** Centre x of a token column for a sequence of `tokens` tokens. */
export const columnX = (token: number, tokens: number) => (token - (tokens - 1) / 2) * COLUMN_W;

/** Instance index of (token, row) — token-major, so a column is contiguous. */
export const cellIndex = (token: number, row: number, rows: number) => token * rows + row;

/** Inverse of {@link cellIndex}. */
export function cellOf(index: number, rows: number): { token: number; row: number } {
  return { token: Math.floor(index / rows), row: index % rows };
}

/** World position and size of one slab cell. */
export function cellPosition(
  s: Pick<Slab, 'rows' | 'tokens' | 'z'>,
  token: number,
  row: number,
): CellPosition {
  const h = rowHeight(s.rows);
  return {
    x: columnX(token, s.tokens),
    y: (row - (s.rows - 1) / 2) * h,
    z: s.z,
    w: COLUMN_W * 0.86,
    h: h * 0.86,
  };
}

export interface HeadCell {
  query: number;
  key: number;
  x: number;
  y: number;
  weight: number;
}

/** Centre of head `head`'s matrix under a slab that is `slabZ` deep. */
export function headOrigin(head: number, heads: number): { x: number; y: number } {
  const pitch = HEAD_SIZE + HEAD_GAP;
  return {
    x: (head - (heads - 1) / 2) * pitch,
    y: -(SLAB_H / 2 + 0.55 + HEAD_SIZE / 2),
  };
}

/** The visible (unmasked) cells of one head's T × T attention matrix. */
export function headCells(probs: Matrix, head: number, heads: number): HeadCell[] {
  const T = probs.rows;
  const cell = HEAD_SIZE / Math.max(T, 1);
  const origin = headOrigin(head, heads);
  const cells: HeadCell[] = [];
  for (let q = 0; q < T; q++) {
    for (let k = 0; k <= q; k++) {
      cells.push({
        query: q,
        key: k,
        x: origin.x + (k - (T - 1) / 2) * cell,
        y: origin.y + ((T - 1) / 2 - q) * cell,
        weight: probs.get(q, k),
      });
    }
  }
  return cells;
}

export interface Bar {
  id: number;
  x: number;
  height: number;
  prob: number;
  kept: boolean;
}

/** Next-token probabilities as bars: tallest = the single most likely token. */
export function outputBars(distribution: NextTokenDistribution, count = BARS): Bar[] {
  const top = distribution.candidates.slice(0, count);
  const max = top[0]?.prob ?? 1;
  return top.map((c, i) => ({
    id: c.id,
    x: (i - (top.length - 1) / 2) * BAR_SPACING,
    height: Math.max(0.03, (c.prob / max) * SLAB_H),
    prob: c.prob,
    kept: c.kept,
  }));
}

/** Largest absolute value (never 0), used to scale a slab's colours. */
export function slabScale(values: ArrayLike<number>): number {
  let max = 1e-9;
  for (let i = 0; i < values.length; i++) max = Math.max(max, Math.abs(values[i] as number));
  return max;
}
