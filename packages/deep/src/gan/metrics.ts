import { createRng } from '@sim/core';
import { MODE_CENTERS, MODE_STD } from './datasets';

/**
 * Sliced Wasserstein-1 distance between two equal-size point clouds (`n × 2`): project both onto
 * many fixed directions, sort, and average the distance between the sorted projections. It is 0 for
 * identical clouds and grows with every kind of mismatch, including a collapsed generator.
 */
export function slicedWasserstein(a: Float64Array, b: Float64Array, projections = 48): number {
  const n = a.length / 2;
  if (n === 0 || a.length !== b.length) throw new RangeError('Need two clouds of the same size');
  const rng = createRng(1234); // the same directions every time, so values are comparable
  let total = 0;
  const pa = new Float64Array(n);
  const pb = new Float64Array(n);
  for (let k = 0; k < projections; k++) {
    const angle = rng.next() * Math.PI;
    const dx = Math.cos(angle);
    const dy = Math.sin(angle);
    for (let i = 0; i < n; i++) {
      pa[i] = (a[2 * i] as number) * dx + (a[2 * i + 1] as number) * dy;
      pb[i] = (b[2 * i] as number) * dx + (b[2 * i + 1] as number) * dy;
    }
    pa.sort();
    pb.sort();
    let d = 0;
    for (let i = 0; i < n; i++) d += Math.abs((pa[i] as number) - (pb[i] as number));
    total += d / n;
  }
  return total / projections;
}

/**
 * Fraction of `samples` that lie within `radius` of some point of `reference`: how many generated
 * points land where real data actually is.
 */
export function fractionNearData(
  samples: Float64Array,
  reference: Float64Array,
  radius = 0.25,
): number {
  const n = samples.length / 2;
  const m = reference.length / 2;
  if (n === 0 || m === 0) return 0;
  const r2 = radius * radius;
  let near = 0;
  for (let i = 0; i < n; i++) {
    const x = samples[2 * i] as number;
    const y = samples[2 * i + 1] as number;
    for (let j = 0; j < m; j++) {
      const dx = x - (reference[2 * j] as number);
      const dy = y - (reference[2 * j + 1] as number);
      if (dx * dx + dy * dy <= r2) {
        near += 1;
        break;
      }
    }
  }
  return near / n;
}

/**
 * For the eight-Gaussians dataset: how many of the eight clusters receive at least `share` of
 * the samples (within four standard deviations), and which ones.
 */
export function modesCovered(
  samples: Float64Array,
  share = 0.02,
): { count: number; covered: boolean[] } {
  const n = samples.length / 2;
  const hits = new Array<number>(MODE_CENTERS.length).fill(0);
  const reach2 = (4 * MODE_STD) ** 2;
  for (let i = 0; i < n; i++) {
    const x = samples[2 * i] as number;
    const y = samples[2 * i + 1] as number;
    MODE_CENTERS.forEach((c, k) => {
      if ((x - c[0]) ** 2 + (y - c[1]) ** 2 <= reach2) hits[k] = (hits[k] as number) + 1;
    });
  }
  const covered = hits.map((h) => n > 0 && h / n >= share);
  return { count: covered.filter(Boolean).length, covered };
}
