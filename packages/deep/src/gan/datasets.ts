import { type Rng } from '@sim/core';

/**
 * Two-dimensional distributions a GAN can learn. Each is an infinite source: `sampleReal` draws
 * as many fresh points as asked, so training never sees the same batch twice.
 */
export const GAN_DATASETS = ['gaussian', 'gaussians8', 'ring', 'moons', 'spiral'] as const;
export type GanDataset = (typeof GAN_DATASETS)[number];

export const ganDatasetInfo: Record<GanDataset, { label: string; note: string }> = {
  gaussian: {
    label: 'One blob',
    note: 'The easiest target: a single Gaussian cloud.',
  },
  gaussians8: {
    label: 'Eight blobs',
    note: 'Eight separate clusters on a circle. The classic test for mode collapse.',
  },
  ring: {
    label: 'Ring',
    note: 'A thin circle: the generator must learn a curved, empty-inside shape.',
  },
  moons: {
    label: 'Two moons',
    note: 'Two interleaved crescents with a gap between them.',
  },
  spiral: {
    label: 'Spiral',
    note: 'A long curve that winds around the origin.',
  },
};

/** Centres of the eight Gaussians (radius 2, every 45°). */
export const MODE_CENTERS: readonly (readonly [number, number])[] = Array.from(
  { length: 8 },
  (_, k) => [2 * Math.cos((k * Math.PI) / 4), 2 * Math.sin((k * Math.PI) / 4)] as const,
);
export const MODE_STD = 0.12;

/** `n` points of a dataset, `n × 2` row-major. */
export function sampleReal(dataset: GanDataset, n: number, rng: Rng): Float64Array {
  const out = new Float64Array(n * 2);
  for (let i = 0; i < n; i++) {
    let x = 0;
    let y = 0;
    switch (dataset) {
      case 'gaussian':
        x = rng.normal() * 0.8;
        y = rng.normal() * 0.8;
        break;
      case 'gaussians8': {
        const c = MODE_CENTERS[Math.floor(rng.next() * 8)] as readonly [number, number];
        x = c[0] + rng.normal() * MODE_STD;
        y = c[1] + rng.normal() * MODE_STD;
        break;
      }
      case 'ring': {
        const angle = rng.next() * 2 * Math.PI;
        const r = 2 + rng.normal() * 0.08;
        x = r * Math.cos(angle);
        y = r * Math.sin(angle);
        break;
      }
      case 'moons': {
        const upper = rng.next() < 0.5;
        const angle = rng.next() * Math.PI;
        const r = 1.5;
        x = upper ? r * Math.cos(angle) - 0.75 : r * (1 - Math.cos(angle)) - 0.75;
        y = upper ? r * Math.sin(angle) - 0.35 : -r * Math.sin(angle) + 0.35;
        x += rng.normal() * 0.08;
        y += rng.normal() * 0.08;
        break;
      }
      case 'spiral': {
        const t = rng.next();
        const angle = t * 4 * Math.PI;
        const r = 0.3 + 2 * t;
        x = r * Math.cos(angle) + rng.normal() * 0.05;
        y = r * Math.sin(angle) + rng.normal() * 0.05;
        break;
      }
    }
    out[i * 2] = x;
    out[i * 2 + 1] = y;
  }
  return out;
}

/** Whether a dataset has well-separated modes (so "modes covered" is meaningful). */
export const hasModes = (dataset: GanDataset): boolean => dataset === 'gaussians8';
