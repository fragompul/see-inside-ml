/** Pure axis helpers for the small line charts. */

/** Evenly thins an index range to at most `max` points, always keeping the first and last. */
export function thin(length: number, max = 240): number[] {
  if (length <= max) return Array.from({ length }, (_, i) => i);
  const step = (length - 1) / (max - 1);
  return Array.from({ length: max }, (_, i) => Math.round(i * step));
}

export interface Range {
  lo: number;
  hi: number;
}

/** A readable linear range around the data: starts at 0 when the data is non-negative. */
export function linearRange(values: ArrayLike<number>, fixed?: Partial<Range>): Range {
  let min = Infinity;
  let max = -Infinity;
  for (let i = 0; i < values.length; i++) {
    const v = values[i] as number;
    if (Number.isFinite(v)) {
      min = Math.min(min, v);
      max = Math.max(max, v);
    }
  }
  if (!Number.isFinite(min)) return { lo: fixed?.lo ?? 0, hi: fixed?.hi ?? 1 };
  const lo = fixed?.lo ?? (min >= 0 ? 0 : niceFloor(min));
  let hi = fixed?.hi ?? niceCeil(max);
  if (hi <= lo) hi = lo + 1;
  return { lo, hi };
}

/** Whole-decade log range holding every positive value (at least one decade tall). */
export function logRange(values: ArrayLike<number>): Range {
  let min = Infinity;
  let max = -Infinity;
  for (let i = 0; i < values.length; i++) {
    const v = values[i] as number;
    if (v > 0 && Number.isFinite(v)) {
      min = Math.min(min, v);
      max = Math.max(max, v);
    }
  }
  if (!Number.isFinite(min)) return { lo: -1, hi: 0 };
  const lo = Math.floor(Math.log10(min));
  const hi = Math.max(Math.ceil(Math.log10(max)), lo + 1);
  return { lo, hi };
}

function niceCeil(v: number): number {
  if (v <= 0) return v === 0 ? 1 : -niceFloor(-v);
  const exp = Math.floor(Math.log10(v));
  const base = 10 ** exp;
  const f = v / base;
  return (f <= 1 ? 1 : f <= 2 ? 2 : f <= 5 ? 5 : 10) * base;
}

function niceFloor(v: number): number {
  return v >= 0 ? 0 : -niceCeil(-v);
}

/** Position of `value` in `[lo, hi]` (0..1, clamped); on a log axis `lo`/`hi` are exponents. */
export function position(value: number, { lo, hi }: Range, log: boolean): number {
  const v = log ? (value > 0 ? Math.log10(value) : lo) : value;
  return Math.max(0, Math.min(1, (v - lo) / (hi - lo)));
}
