/** Classic hand-made 3×3 kernels for the convolution explainer (row-major). */
export interface KernelPreset {
  id: string;
  label: string;
  note: string;
  values: readonly number[];
}

export const KERNEL_PRESETS: readonly KernelPreset[] = [
  {
    id: 'edge-v',
    label: 'Vertical edges',
    note: 'Fires where dark meets light from left to right (Sobel X).',
    values: [-1, 0, 1, -2, 0, 2, -1, 0, 1],
  },
  {
    id: 'edge-h',
    label: 'Horizontal edges',
    note: 'Fires where dark meets light from top to bottom (Sobel Y).',
    values: [-1, -2, -1, 0, 0, 0, 1, 2, 1],
  },
  {
    id: 'blur',
    label: 'Blur',
    note: 'Averages each pixel with its neighbours.',
    values: [1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9],
  },
  {
    id: 'sharpen',
    label: 'Sharpen',
    note: 'Boosts the centre against its surroundings.',
    values: [0, -1, 0, -1, 5, -1, 0, -1, 0],
  },
  {
    id: 'identity',
    label: 'Identity',
    note: 'Copies the image: a 1 in the middle and zeros elsewhere.',
    values: [0, 0, 0, 0, 1, 0, 0, 0, 0],
  },
];

/** Keeps a window of `k × k` fully inside a `side × side` image, centred on a pointer position. */
export function windowOrigin(pointer: number, k: number, side: number): number {
  return Math.max(0, Math.min(side - k, Math.round(pointer - k / 2)));
}

/** The `k × k` patch of `image` whose top-left corner is (x, y). */
export function patchAt(
  image: ArrayLike<number>,
  side: number,
  x: number,
  y: number,
  k: number,
): number[] {
  const out: number[] = [];
  for (let ky = 0; ky < k; ky++)
    for (let kx = 0; kx < k; kx++) out.push(image[(y + ky) * side + x + kx] as number);
  return out;
}
