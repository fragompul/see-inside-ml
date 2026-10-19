import { createRng } from '@sim/core';

/**
 * Position encodings: how an attention layer, which sees a set of vectors, learns where each one sits.
 *  - sinusoidal: a fixed vector per position, added to the embedding (Vaswani et al.)
 *  - rope: rotate the query and key by an angle proportional to the position (Su et al.)
 *  - alibi: add a penalty proportional to the distance to every attention score (Press et al.)
 *  - learned: a lookup table with one trained vector per position, so nothing exists past its length
 */
export type PositionScheme = 'sinusoidal' | 'rope' | 'alibi' | 'learned';

export const POSITION_SCHEMES: PositionScheme[] = ['sinusoidal', 'rope', 'alibi', 'learned'];

/** Angular frequency of the i-th (sin, cos) pair: 10000^(-2i/d). */
export const frequency = (pair: number, dim: number): number => 10000 ** (-(2 * pair) / dim);

export function sinusoidalEncoding(position: number, dim: number): Float64Array {
  const out = new Float64Array(dim);
  for (let i = 0; i < dim / 2; i++) {
    const angle = position * frequency(i, dim);
    out[2 * i] = Math.sin(angle);
    out[2 * i + 1] = Math.cos(angle);
  }
  return out;
}

/** Rotates each (even, odd) pair of x by position × its frequency. */
export function rotate(x: ArrayLike<number>, position: number): Float64Array {
  const dim = x.length;
  const out = new Float64Array(dim);
  for (let i = 0; i < dim / 2; i++) {
    const angle = position * frequency(i, dim);
    const c = Math.cos(angle);
    const s = Math.sin(angle);
    const a = x[2 * i] as number;
    const b = x[2 * i + 1] as number;
    out[2 * i] = a * c - b * s;
    out[2 * i + 1] = a * s + b * c;
  }
  return out;
}

export const dot = (a: ArrayLike<number>, b: ArrayLike<number>): number => {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += (a[i] as number) * (b[i] as number);
  return s;
};

/** The ALiBi slope of head h out of H: 2^(-8(h+1)/H). */
export const alibiSlope = (head: number, heads: number): number => 2 ** (-(8 * (head + 1)) / heads);

/** A table of random vectors with one row per trained position, as at initialisation. */
export function learnedTable(length: number, dim: number, seed: number): Float64Array[] {
  const rng = createRng(seed);
  return Array.from({ length }, () => {
    const row = new Float64Array(dim);
    for (let i = 0; i < dim; i++) row[i] = rng.normal();
    return row;
  });
}

export interface PositionMatrix {
  /** Row-major length × length: how strongly position i relates to position j. NaN where undefined. */
  values: Float64Array;
  length: number;
}

/**
 * The position-only part of the attention score between every pair of positions, for tokens whose content is
 * identical, so that nothing but position can matter. For sinusoidal and RoPE this is the similarity of the
 * two position codes (scaled to 1 at distance 0); for ALiBi it is the bias for one head (zero on the diagonal,
 * increasingly negative with distance; causal pairs only); for a learned table it is the similarity of two
 * random rows, undefined (NaN) beyond the trained length.
 */
export function positionMatrix(
  scheme: PositionScheme,
  length: number,
  dim: number,
  trained: number,
  seed = 1,
): PositionMatrix {
  const values = new Float64Array(length * length);
  const ones = new Float64Array(dim).fill(1);
  const table = scheme === 'learned' ? learnedTable(Math.min(trained, length), dim, seed) : [];
  for (let i = 0; i < length; i++)
    for (let j = 0; j < length; j++) {
      let v: number;
      if (scheme === 'sinusoidal')
        v = dot(sinusoidalEncoding(i, dim), sinusoidalEncoding(j, dim)) / (dim / 2);
      else if (scheme === 'rope') v = dot(rotate(ones, i), rotate(ones, j)) / dim;
      else if (scheme === 'alibi') v = j > i ? Number.NaN : -alibiSlope(0, 8) * (i - j);
      else
        v =
          i < table.length && j < table.length
            ? dot(table[i] as Float64Array, table[j] as Float64Array) / dim
            : Number.NaN;
      values[i * length + j] = v;
    }
  return { values, length };
}

/** Mean of the defined entries at the given offsets (|i - j| in [lo, hi]). */
export function meanAtOffsets(m: PositionMatrix, lo: number, hi: number): number {
  let sum = 0;
  let n = 0;
  for (let i = 0; i < m.length; i++)
    for (let j = 0; j < m.length; j++) {
      const d = Math.abs(i - j);
      const v = m.values[i * m.length + j] as number;
      if (d >= lo && d <= hi && !Number.isNaN(v)) {
        sum += v;
        n++;
      }
    }
  return n ? sum / n : Number.NaN;
}

/** The largest spread of values along any diagonal: 0 when the score depends only on the offset. */
export function diagonalSpread(m: PositionMatrix): number {
  let worst = 0;
  for (let k = -(m.length - 1); k < m.length; k++) {
    let lo = Infinity;
    let hi = -Infinity;
    for (let i = 0; i < m.length; i++) {
      const j = i + k;
      if (j < 0 || j >= m.length) continue;
      const v = m.values[i * m.length + j] as number;
      if (Number.isNaN(v)) continue;
      lo = Math.min(lo, v);
      hi = Math.max(hi, v);
    }
    if (hi >= lo) worst = Math.max(worst, hi - lo);
  }
  return worst;
}
