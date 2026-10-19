import { Matrix, type Table } from '@sim/core';

/** Pure edits of a table, used when the learner adds, drags or erases points on a plot. */

export function addRow(t: Table, features: readonly number[], target: number): Table {
  const d = t.x.cols;
  const x = new Matrix(t.x.rows + 1, d);
  x.data.set(t.x.data, 0);
  x.data.set(features, t.x.rows * d);
  const y = new Float64Array(t.y.length + 1);
  y.set(t.y, 0);
  y[t.y.length] = target;
  return { ...t, x, y };
}

export function removeRow(t: Table, index: number): Table {
  const d = t.x.cols;
  const keep = Array.from({ length: t.x.rows }, (_, i) => i).filter((i) => i !== index);
  const x = new Matrix(keep.length, d);
  const y = new Float64Array(keep.length);
  keep.forEach((src, k) => {
    x.data.set(t.x.data.subarray(src * d, (src + 1) * d), k * d);
    y[k] = t.y[src] as number;
  });
  return { ...t, x, y };
}

export function moveRow(
  t: Table,
  index: number,
  features: readonly number[],
  target?: number,
): Table {
  const x = t.x.clone();
  x.data.set(features, index * t.x.cols);
  const y = Float64Array.from(t.y);
  if (target !== undefined) y[index] = target;
  return { ...t, x, y };
}

/** Index of the row closest to (x, y) in pixel space, if within `radius` pixels. */
export function nearestPoint(
  px: ArrayLike<number>,
  py: ArrayLike<number>,
  x: number,
  y: number,
  radius: number,
): number | null {
  let best: number | null = null;
  let bestD = radius * radius;
  for (let i = 0; i < px.length; i++) {
    const d = ((px[i] as number) - x) ** 2 + ((py[i] as number) - y) ** 2;
    if (d <= bestD) {
      bestD = d;
      best = i;
    }
  }
  return best;
}
