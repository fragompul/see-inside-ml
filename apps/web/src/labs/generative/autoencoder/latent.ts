/** Pure geometry of the latent-space map (no DOM): scaling, picking and the decoder grid. */

/**
 * Half-width of the square window `[-e, e]²` that holds (almost) every code in the two plotted
 * dimensions: the 98th percentile of |coordinate| with some margin, rounded up to a half unit.
 */
export function plotExtent(
  mu: ArrayLike<number>,
  count: number,
  latent: number,
  dims: readonly [number, number],
): number {
  const values: number[] = [];
  for (let i = 0; i < count; i++)
    for (const d of dims) values.push(Math.abs(mu[i * latent + d] as number));
  if (values.length === 0) return 2;
  values.sort((a, b) => a - b);
  const q = values[Math.min(values.length - 1, Math.floor(values.length * 0.98))] as number;
  return Math.max(2, Math.ceil(q * 1.15 * 2) / 2);
}

export { fromPixel, toPixel } from '@/shared/canvas/plane';

/**
 * Codes of an `n × n` grid covering the window, row by row from the top (largest y first), with
 * every other latent dimension held at the probe's value. Result is `n² × latent`, row-major.
 */
export function gridCodes(
  extent: number,
  n: number,
  probe: ArrayLike<number>,
  dimX: number,
  dimY: number,
): Float64Array {
  const latent = probe.length;
  const out = new Float64Array(n * n * latent);
  for (let row = 0; row < n; row++) {
    for (let col = 0; col < n; col++) {
      const base = (row * n + col) * latent;
      for (let d = 0; d < latent; d++) out[base + d] = probe[d] as number;
      out[base + dimX] = ((col + 0.5) / n) * 2 * extent - extent;
      out[base + dimY] = extent - ((row + 0.5) / n) * 2 * extent;
    }
  }
  return out;
}

/** Ten evenly spread hues, lighter on dark backgrounds. */
export const digitColor = (digit: number, dark: boolean): string =>
  `hsl(${(digit * 36 + 12) % 360} ${dark ? 78 : 70}% ${dark ? 62 : 44}%)`;
