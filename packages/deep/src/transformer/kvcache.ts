import { createRng } from '@sim/core';
import { rotate } from './positional';

/**
 * The cost of generating text with attention, with and without a key/value cache.
 * A tiny one-layer decoder is run both ways to prove the cache changes the cost and not the answer,
 * and counts every multiply-add so that the closed-form cost model used by the lab can be checked.
 */
export interface CostShape {
  /** Model width. */
  d: number;
  /** Width of the keys and values that are stored per token (d for multi-head attention, less for grouped). */
  dkv: number;
  vocab: number;
}

export interface Macs {
  /** Projections (query, key, value, output) and the final vocabulary scores. */
  projection: number;
  /** Query-key scores and the weighted sum of values. */
  attention: number;
}

const add = (a: Macs, b: Macs): Macs => ({
  projection: a.projection + b.projection,
  attention: a.attention + b.attention,
});
const ZERO: Macs = { projection: 0, attention: 0 };

/**
 * Multiply-adds for one decoding step that writes one token after reading `length` tokens, of which
 * `fresh` still need their keys and values computed. The query, output projection and vocabulary scores
 * are for the last token only; the scores and weighted sum run over all `length` keys.
 */
export function stepMacs(shape: CostShape, length: number, fresh: number): Macs {
  const { d, dkv, vocab } = shape;
  return {
    projection: fresh * 2 * d * dkv + 2 * d * d + vocab * d,
    attention: 2 * length * d,
  };
}

/**
 * Total multiply-adds to write `steps` tokens after a prompt of `prompt` tokens. With a cache every
 * token's keys and values are computed once. Without it every step recomputes them for the whole text.
 */
export function generationMacs(
  shape: CostShape,
  prompt: number,
  steps: number,
  cached: boolean,
): Macs {
  let total = ZERO;
  for (let s = 0; s < steps; s++) {
    const length = prompt + s;
    total = add(total, stepMacs(shape, length, cached ? (s === 0 ? length : 1) : length));
  }
  return total;
}

/** Bytes of keys and values kept for a text of `context` tokens. */
export const cacheBytes = (layers: number, dkv: number, context: number, bytes: number): number =>
  2 * layers * dkv * context * bytes;

// ---------------------------------------------------------------------------------------------
// A tiny decoder, run for real

export interface TinyDecoder {
  shape: CostShape;
  heads: number;
  embed: Float64Array[];
  wq: Float64Array[];
  wk: Float64Array[];
  wv: Float64Array[];
  wo: Float64Array[];
  unembed: Float64Array[];
}

export function makeDecoder(d: number, heads: number, vocab: number, seed: number): TinyDecoder {
  const rng = createRng(seed);
  const matrix = (rows: number, cols: number, std: number) =>
    Array.from({ length: rows }, () =>
      Float64Array.from({ length: cols }, () => rng.normal() * std),
    );
  const std = 1 / Math.sqrt(d);
  return {
    shape: { d, dkv: d, vocab },
    heads,
    embed: matrix(vocab, d, 1),
    wq: matrix(d, d, std),
    wk: matrix(d, d, std),
    wv: matrix(d, d, std),
    wo: matrix(d, d, std),
    unembed: matrix(vocab, d, std),
  };
}

const mulVec = (rows: Float64Array[], x: ArrayLike<number>): Float64Array =>
  Float64Array.from(rows, (row) => {
    let s = 0;
    for (let i = 0; i < row.length; i++) s += (row[i] as number) * (x[i] as number);
    return s;
  });

export interface DecodeResult {
  tokens: number[];
  /** Scores for the next token, at the final step. */
  lastLogits: Float64Array;
  macs: Macs;
}

/** Key and value of the token at `position`, with the position rotated into the key (as RoPE does). */
function keyValue(m: TinyDecoder, token: number, position: number) {
  const x = m.embed[token] as Float64Array;
  const dk = m.shape.d / m.heads;
  const k = mulVec(m.wk, x);
  const rotated = new Float64Array(m.shape.d);
  for (let h = 0; h < m.heads; h++)
    rotated.set(rotate(k.subarray(h * dk, (h + 1) * dk), position), h * dk);
  return { k: rotated, v: mulVec(m.wv, x) };
}

/** Greedy generation. `cached` decides whether earlier keys and values are remembered or recomputed. */
export function decodeText(
  m: TinyDecoder,
  prompt: number[],
  steps: number,
  cached: boolean,
): DecodeResult {
  const { d, vocab } = m.shape;
  const dk = d / m.heads;
  const tokens = [...prompt];
  let macs = ZERO;
  const keys: Float64Array[] = [];
  const values: Float64Array[] = [];
  let lastLogits: Float64Array = new Float64Array(vocab);
  for (let s = 0; s < steps; s++) {
    const length = tokens.length;
    if (!cached) {
      keys.length = 0;
      values.length = 0;
    }
    const fresh = length - keys.length;
    for (let t = keys.length; t < length; t++) {
      const kv = keyValue(m, tokens[t] as number, t);
      keys.push(kv.k);
      values.push(kv.v);
    }
    macs = add(macs, stepMacs(m.shape, length, fresh));
    // the last token attends over every key so far
    const x = m.embed[tokens[length - 1] as number] as Float64Array;
    const qFull = mulVec(m.wq, x);
    const out = new Float64Array(d);
    for (let h = 0; h < m.heads; h++) {
      const q = rotate(qFull.subarray(h * dk, (h + 1) * dk), length - 1);
      const scores = keys.map((k) => {
        let sum = 0;
        for (let i = 0; i < dk; i++) sum += (q[i] as number) * (k[h * dk + i] as number);
        return sum / Math.sqrt(dk);
      });
      const top = Math.max(...scores);
      const weights = scores.map((v) => Math.exp(v - top));
      const z = weights.reduce((a, b) => a + b, 0);
      for (let t = 0; t < length; t++)
        for (let i = 0; i < dk; i++)
          out[h * dk + i] =
            (out[h * dk + i] as number) +
            ((weights[t] as number) / z) * ((values[t] as Float64Array)[h * dk + i] as number);
    }
    const hidden = mulVec(m.wo, out);
    for (let i = 0; i < d; i++) hidden[i] = (hidden[i] as number) + (x[i] as number);
    lastLogits = mulVec(m.unembed, hidden);
    let best = 0;
    for (let v = 1; v < vocab; v++)
      if ((lastLogits[v] as number) > (lastLogits[best] as number)) best = v;
    tokens.push(best);
  }
  return { tokens, lastLogits, macs };
}
