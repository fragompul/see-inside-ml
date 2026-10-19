import { type Rng } from '@sim/core';

/**
 * Procedural handwritten-looking digits. Every digit is a skeleton of strokes in the unit square
 * that is jittered, tilted, scaled and thickened at random, then rasterised with anti-aliasing.
 * No dataset to download, no licence, and an endless supply of seeded variety.
 */

export const IMAGE_SIZE = 16;
export const DIGIT_CLASSES = 10;

type Point = readonly [number, number];
type Stroke = readonly Point[];

const TAU = Math.PI * 2;

/** Points along an elliptical arc (angles in radians, y pointing down). */
function arc(
  cx: number,
  cy: number,
  rx: number,
  ry: number,
  from: number,
  to: number,
  n = 18,
): Stroke {
  return Array.from({ length: n + 1 }, (_, i) => {
    const a = from + ((to - from) * i) / n;
    return [cx + rx * Math.cos(a), cy + ry * Math.sin(a)] as const;
  });
}

/** Skeleton strokes of each digit in the unit square. */
export const SKELETONS: readonly (readonly Stroke[])[] = [
  /* 0 */ [arc(0.5, 0.5, 0.28, 0.42, 0, TAU, 26)],
  /* 1 */ [
    [
      [0.32, 0.26],
      [0.52, 0.08],
      [0.52, 0.92],
    ],
  ],
  /* 2 */ [[...arc(0.5, 0.3, 0.28, 0.22, Math.PI, TAU + 0.45, 14), [0.2, 0.92], [0.8, 0.92]]],
  /* 3 */ [
    arc(0.48, 0.29, 0.28, 0.2, -0.85 * Math.PI, 0.5 * Math.PI, 14),
    arc(0.5, 0.7, 0.3, 0.22, -0.5 * Math.PI, 0.85 * Math.PI, 14),
  ],
  /* 4 */ [
    [
      [0.62, 0.92],
      [0.62, 0.08],
      [0.16, 0.66],
      [0.86, 0.66],
    ],
  ],
  /* 5 */ [
    [
      [0.78, 0.1],
      [0.28, 0.1],
      [0.25, 0.48],
    ],
    arc(0.48, 0.68, 0.3, 0.24, -0.75 * Math.PI, 0.8 * Math.PI, 14),
  ],
  /* 6 */ [
    [
      [0.68, 0.08],
      [0.4, 0.25],
      [0.24, 0.58],
    ],
    arc(0.5, 0.67, 0.27, 0.25, Math.PI, 3 * Math.PI, 20),
  ],
  /* 7 */ [
    [
      [0.2, 0.1],
      [0.82, 0.1],
      [0.42, 0.92],
    ],
  ],
  /* 8 */ [arc(0.5, 0.29, 0.24, 0.2, 0, TAU, 18), arc(0.5, 0.71, 0.29, 0.22, 0, TAU, 18)],
  /* 9 */ [
    arc(0.5, 0.32, 0.27, 0.24, 0, TAU, 18),
    [
      [0.77, 0.34],
      [0.71, 0.65],
      [0.4, 0.92],
    ],
  ],
];

export interface DigitStyle {
  /** Vertex wobble, in unit-square units (default 0.025). */
  jitter: number;
  /** Rotation range ± radians. */
  tilt: number;
  /** Pixel noise standard deviation. */
  noise: number;
}

export const DEFAULT_STYLE: DigitStyle = { jitter: 0.025, tilt: 0.2, noise: 0.04 };

const MARGIN = 2;

/** Squared distance from point p to segment ab. */
function segmentDistance(
  px: number,
  py: number,
  ax: number,
  ay: number,
  bx: number,
  by: number,
): number {
  const dx = bx - ax;
  const dy = by - ay;
  const len2 = dx * dx + dy * dy;
  const t = len2 === 0 ? 0 : Math.max(0, Math.min(1, ((px - ax) * dx + (py - ay) * dy) / len2));
  const cx = ax + t * dx;
  const cy = ay + t * dy;
  return Math.hypot(px - cx, py - cy);
}

/** Renders one handwritten-looking digit as `IMAGE_SIZE²` floats in [0, 1] (row-major, ink = 1). */
export function renderDigit(
  digit: number,
  rng: Rng,
  style: DigitStyle = DEFAULT_STYLE,
): Float64Array {
  const skeleton = SKELETONS[digit];
  if (!skeleton) throw new RangeError(`Digit must be 0..9, received ${digit}`);

  const angle = rng.uniform(-style.tilt, style.tilt);
  const scaleX = rng.uniform(0.82, 1.08);
  const scaleY = rng.uniform(0.82, 1.08);
  const shear = rng.uniform(-0.15, 0.15);
  const shiftX = rng.uniform(-0.07, 0.07);
  const shiftY = rng.uniform(-0.07, 0.07);
  const width = rng.uniform(1.3, 2.3);
  const cos = Math.cos(angle);
  const sin = Math.sin(angle);
  const span = IMAGE_SIZE - 2 * MARGIN;

  // Skeleton → pixel coordinates, with per-vertex wobble and a random affine transform.
  const strokes = skeleton.map((stroke) =>
    stroke.map(([ux, uy]) => {
      let x = ux - 0.5 + rng.normal() * style.jitter;
      const y = uy - 0.5 + rng.normal() * style.jitter;
      x += shear * y;
      const rx = (x * cos - y * sin) * scaleX + shiftX;
      const ry = (x * sin + y * cos) * scaleY + shiftY;
      return [MARGIN + (rx + 0.5) * span, MARGIN + (ry + 0.5) * span] as const;
    }),
  );

  const image = new Float64Array(IMAGE_SIZE * IMAGE_SIZE);
  for (let row = 0; row < IMAGE_SIZE; row++) {
    for (let col = 0; col < IMAGE_SIZE; col++) {
      const px = col + 0.5;
      const py = row + 0.5;
      let nearest = Infinity;
      for (const stroke of strokes) {
        for (let i = 0; i < stroke.length - 1; i++) {
          const a = stroke[i] as Point;
          const b = stroke[i + 1] as Point;
          nearest = Math.min(nearest, segmentDistance(px, py, a[0], a[1], b[0], b[1]));
        }
      }
      // anti-aliased disc profile: full ink inside the stroke, a one-pixel soft edge outside it
      const ink = Math.max(0, Math.min(1, width / 2 + 0.5 - nearest));
      const noisy = ink + (style.noise > 0 ? rng.normal() * style.noise : 0);
      image[row * IMAGE_SIZE + col] = Math.max(0, Math.min(1, noisy));
    }
  }
  return image;
}

export interface DigitSet {
  /** `count × IMAGE_SIZE²`, image-major. */
  images: Float64Array;
  labels: Uint8Array;
  count: number;
}

/** A balanced, shuffled set: `perClass` images of each digit. */
export function generateDigits(
  perClass: number,
  rng: Rng,
  style: DigitStyle = DEFAULT_STYLE,
): DigitSet {
  const count = perClass * DIGIT_CLASSES;
  const pixels = IMAGE_SIZE * IMAGE_SIZE;
  const order = Array.from({ length: count }, (_, i) => i);
  for (let i = count - 1; i > 0; i--) {
    const j = Math.floor(rng.next() * (i + 1));
    [order[i], order[j]] = [order[j] as number, order[i] as number];
  }
  const images = new Float64Array(count * pixels);
  const labels = new Uint8Array(count);
  order.forEach((slot, n) => {
    const digit = slot % DIGIT_CLASSES;
    labels[n] = digit;
    images.set(renderDigit(digit, rng, style), n * pixels);
  });
  return { images, labels, count };
}
