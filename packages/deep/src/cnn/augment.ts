import type { Rng } from '@sim/core';
import { IMAGE_SIZE } from './digits';

/**
 * Image augmentations for the 16 x 16 digits: small random changes that keep the label true. Each takes and
 * returns a flat array of IMAGE_SIZE^2 pixels in [0, 1].
 */
const S = IMAGE_SIZE;

/** Moves the picture by whole pixels (positive dx moves it right, dy down); uncovered pixels become 0. */
export function shiftImage(image: ArrayLike<number>, dx: number, dy: number): Float64Array {
  const out = new Float64Array(S * S);
  for (let y = 0; y < S; y++)
    for (let x = 0; x < S; x++) {
      const sx = x - dx;
      const sy = y - dy;
      if (sx >= 0 && sx < S && sy >= 0 && sy < S) out[y * S + x] = image[sy * S + sx] as number;
    }
  return out;
}

/** Rotates about the centre by `degrees` (positive is clockwise on screen) with bilinear sampling. */
export function rotateImage(image: ArrayLike<number>, degrees: number): Float64Array {
  const out = new Float64Array(S * S);
  const a = (degrees * Math.PI) / 180;
  const c = Math.cos(a);
  const s = Math.sin(a);
  const mid = (S - 1) / 2;
  for (let y = 0; y < S; y++)
    for (let x = 0; x < S; x++) {
      // where in the original does this output pixel come from?
      const u = x - mid;
      const v = y - mid;
      const sx = c * u + s * v + mid;
      const sy = -s * u + c * v + mid;
      const x0 = Math.floor(sx);
      const y0 = Math.floor(sy);
      const fx = sx - x0;
      const fy = sy - y0;
      const at = (px: number, py: number) =>
        px >= 0 && px < S && py >= 0 && py < S ? (image[py * S + px] as number) : 0;
      out[y * S + x] =
        at(x0, y0) * (1 - fx) * (1 - fy) +
        at(x0 + 1, y0) * fx * (1 - fy) +
        at(x0, y0 + 1) * (1 - fx) * fy +
        at(x0 + 1, y0 + 1) * fx * fy;
    }
  return out;
}

/** Adds Gaussian noise of the given standard deviation and clips to [0, 1]. */
export function noiseImage(image: ArrayLike<number>, sd: number, rng: Rng): Float64Array {
  return Float64Array.from(image, (v) => Math.min(1, Math.max(0, v + sd * rng.normal())));
}

export interface Augmentation {
  /** Largest shift in pixels in each direction. */
  shift: number;
  /** Largest rotation in degrees in either direction. */
  rotate: number;
  /** Standard deviation of the noise. */
  noise: number;
}

export const NO_AUGMENTATION: Augmentation = { shift: 0, rotate: 0, noise: 0 };

/** One random augmented copy: rotation, then shift, then noise. */
export function augmentImage(image: ArrayLike<number>, a: Augmentation, rng: Rng): Float64Array {
  const uniform = (limit: number) => (rng.next() * 2 - 1) * limit;
  let out: Float64Array = Float64Array.from(image);
  if (a.rotate > 0) out = rotateImage(out, uniform(a.rotate));
  if (a.shift > 0)
    out = shiftImage(out, Math.round(uniform(a.shift + 0.49)), Math.round(uniform(a.shift + 0.49)));
  if (a.noise > 0) out = noiseImage(out, a.noise, rng);
  return out;
}

/** The set with `copies` random augmented versions of every image (the original is kept as the first). */
export function augmentSet(
  images: ArrayLike<number>,
  count: number,
  a: Augmentation,
  copies: number,
  rng: Rng,
): { images: Float64Array; source: number[] } {
  const out = new Float64Array(count * (copies + 1) * S * S);
  const source: number[] = [];
  let k = 0;
  for (let i = 0; i < count; i++) {
    const original = Array.from({ length: S * S }, (_, p) => images[i * S * S + p] as number);
    out.set(original, k * S * S);
    source.push(i);
    k++;
    for (let c = 0; c < copies; c++) {
      out.set(augmentImage(original, a, rng), k * S * S);
      source.push(i);
      k++;
    }
  }
  return { images: out, source };
}
