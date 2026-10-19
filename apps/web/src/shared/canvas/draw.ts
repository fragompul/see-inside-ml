import { IMAGE_SIZE } from '@sim/deep';

/** Pure helpers behind the drawing pad (no DOM): brush, continuous strokes and normalisation. */

export const BRUSH_RADIUS = 1.35;
const INK_THRESHOLD = 0.15;
/** The training digits live inside this central box (see MARGIN in the digit generator). */
const FIT = IMAGE_SIZE - 4;

/** Soft round brush at pixel-space position (x, y); keeps the brightest value per pixel. */
export function paint(
  image: Float64Array,
  x: number,
  y: number,
  radius = BRUSH_RADIUS,
  size = IMAGE_SIZE,
): void {
  const reach = Math.ceil(radius + 1);
  for (
    let row = Math.max(0, Math.floor(y) - reach);
    row <= Math.min(size - 1, Math.floor(y) + reach);
    row++
  ) {
    for (
      let col = Math.max(0, Math.floor(x) - reach);
      col <= Math.min(size - 1, Math.floor(x) + reach);
      col++
    ) {
      const d = Math.hypot(col + 0.5 - x, row + 0.5 - y);
      const v = Math.max(0, Math.min(1, (radius + 0.5 - d) / 1.2));
      if (v > (image[row * size + col] as number)) image[row * size + col] = v;
    }
  }
}

/** Paints a continuous stroke from `from` to `to` so fast mouse moves leave no gaps. */
export function stroke(
  image: Float64Array,
  from: readonly [number, number],
  to: readonly [number, number],
  radius = BRUSH_RADIUS,
  size = IMAGE_SIZE,
): void {
  const distance = Math.hypot(to[0] - from[0], to[1] - from[1]);
  const steps = Math.max(1, Math.ceil(distance / 0.3));
  for (let i = 0; i <= steps; i++) {
    const t = i / steps;
    paint(image, from[0] + (to[0] - from[0]) * t, from[1] + (to[1] - from[1]) * t, radius, size);
  }
}

export interface InkStats {
  /** Total ink. */
  mass: number;
  /** Centre of mass in pixels, or null for an empty image. */
  centroid: [number, number] | null;
  /** Bounding box of pixels above the threshold: [minX, minY, maxX, maxY] (inclusive), or null. */
  box: [number, number, number, number] | null;
}

export function inkStats(image: ArrayLike<number>, size = IMAGE_SIZE): InkStats {
  let mass = 0;
  let sx = 0;
  let sy = 0;
  let box: [number, number, number, number] | null = null;
  for (let row = 0; row < size; row++) {
    for (let col = 0; col < size; col++) {
      const v = image[row * size + col] as number;
      mass += v;
      sx += v * (col + 0.5);
      sy += v * (row + 0.5);
      if (v > INK_THRESHOLD) {
        box = box
          ? [
              Math.min(box[0], col),
              Math.min(box[1], row),
              Math.max(box[2], col),
              Math.max(box[3], row),
            ]
          : [col, row, col, row];
      }
    }
  }
  return { mass, centroid: mass > 0 ? [sx / mass, sy / mass] : null, box };
}

const sample = (image: ArrayLike<number>, x: number, y: number, size: number): number => {
  const px = x - 0.5;
  const py = y - 0.5;
  const x0 = Math.floor(px);
  const y0 = Math.floor(py);
  const fx = px - x0;
  const fy = py - y0;
  const at = (c: number, r: number) =>
    c < 0 || r < 0 || c >= size || r >= size ? 0 : (image[r * size + c] as number);
  return (
    at(x0, y0) * (1 - fx) * (1 - fy) +
    at(x0 + 1, y0) * fx * (1 - fy) +
    at(x0, y0 + 1) * (1 - fx) * fy +
    at(x0 + 1, y0 + 1) * fx * fy
  );
};

/**
 * Centres the drawing and scales it to fill the box the training digits occupy: the same trick
 * used when MNIST was prepared. A convolutional network is translation-friendly, not magic: it
 * still reads best when the digit looks like the ones it was trained on.
 */
export function normalizeDrawing(image: Float64Array, size = IMAGE_SIZE): Float64Array {
  const { box } = inkStats(image, size);
  if (!box) return Float64Array.from(image);
  const w = box[2] - box[0] + 1;
  const h = box[3] - box[1] + 1;
  // never blow a tiny dot up into a blob
  const scale = Math.min(FIT / Math.max(w, h), 3);
  const cx = (box[0] + box[2] + 1) / 2;
  const cy = (box[1] + box[3] + 1) / 2;
  const out = new Float64Array(size * size);
  for (let row = 0; row < size; row++) {
    for (let col = 0; col < size; col++) {
      const sx = (col + 0.5 - size / 2) / scale + cx;
      const sy = (row + 0.5 - size / 2) / scale + cy;
      out[row * size + col] = Math.max(0, Math.min(1, sample(image, sx, sy, size)));
    }
  }
  return out;
}
