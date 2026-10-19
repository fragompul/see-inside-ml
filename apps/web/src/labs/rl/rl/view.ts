import { ACTION_COUNT } from '@sim/deep';

/** Pure helpers for drawing the gridworld (no DOM). */

/** Value range over cells that are neither walls nor terminal (so one colour scale fits them all). */
export function valueRange(
  values: ArrayLike<number>,
  skip: (cell: number) => boolean,
): { lo: number; hi: number } {
  let lo = Infinity;
  let hi = -Infinity;
  for (let s = 0; s < values.length; s++) {
    if (skip(s)) continue;
    lo = Math.min(lo, values[s] as number);
    hi = Math.max(hi, values[s] as number);
  }
  if (!Number.isFinite(lo)) return { lo: 0, hi: 1 };
  if (hi - lo < 1e-9) return { lo: lo - 0.5, hi: hi + 0.5 };
  return { lo, hi };
}

/** Position of `value` inside a range as 0..1 (clamped). */
export const normalise = (value: number, { lo, hi }: { lo: number; hi: number }): number =>
  Math.max(0, Math.min(1, (value - lo) / (hi - lo)));

/** The greedy action(s) of a cell: every action within a hair of the best; empty when all are equal (nothing learned). */
export function bestActions(q: ArrayLike<number>, cell: number): number[] {
  const values = Array.from(
    { length: ACTION_COUNT },
    (_, a) => q[cell * ACTION_COUNT + a] as number,
  );
  const top = Math.max(...values);
  const bottom = Math.min(...values);
  if (top - bottom < 1e-9) return [];
  return values.flatMap((v, a) => (v >= top - 1e-9 ? [a] : []));
}

/**
 * The triangle of an action inside a square cell: its apex at the centre and its base on the side
 * the action points to. Returns `x1,y1 x2,y2 x3,y3` for an SVG polygon.
 */
export function actionTriangle(action: number, size: number, inset = 0): string {
  const c = size / 2;
  const a = inset;
  const b = size - inset;
  const corners = [
    [a, a, b, a], // up: top side
    [b, a, b, b], // right: right side
    [b, b, a, b], // down: bottom side
    [a, b, a, a], // left: left side
  ][action] as number[];
  return `${c},${c} ${corners[0]},${corners[1]} ${corners[2]},${corners[3]}`;
}

/** Centre of the arrow glyph of an action: unit vector (dx, dy) in screen coordinates. */
export const ACTION_VECTOR: readonly (readonly [number, number])[] = [
  [0, -1],
  [1, 0],
  [0, 1],
  [-1, 0],
];

/** Cell index → [column, row]. */
export const cellXY = (cell: number, width: number): [number, number] => [
  cell % width,
  Math.floor(cell / width),
];

/** Pixel position inside a `width × height` grid drawn `size` pixels per cell → cell index, or null outside. */
export function cellAt(
  x: number,
  y: number,
  width: number,
  height: number,
  size: number,
): number | null {
  const col = Math.floor(x / size);
  const row = Math.floor(y / size);
  return col < 0 || row < 0 || col >= width || row >= height ? null : row * width + col;
}
