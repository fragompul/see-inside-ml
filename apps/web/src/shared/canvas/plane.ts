/** Mapping between data coordinates and pixels on a square plot `[-extent, extent]²`. */

/** Latent value → pixel along an axis of `size` pixels (x grows right, y is flipped by the caller). */
export const toPixel = (value: number, extent: number, size: number): number =>
  ((value + extent) / (2 * extent)) * size;

/** Pixel → latent value, clamped to the window. */
export const fromPixel = (pixel: number, extent: number, size: number): number =>
  Math.max(-extent, Math.min(extent, (pixel / size) * 2 * extent - extent));
