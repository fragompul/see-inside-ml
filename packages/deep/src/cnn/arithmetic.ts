/**
 * Convolution arithmetic in one dimension (everything here carries over unchanged to each axis of an image):
 * how big the output is, how much of the input each output value can see, and what the filters cost.
 */
export interface Window {
  kind: 'conv' | 'pool';
  kernel: number;
  stride: number;
  padding: number;
  dilation: number;
}

export const conv = (kernel: number, stride = 1, padding = 0, dilation = 1): Window => ({
  kind: 'conv',
  kernel,
  stride,
  padding,
  dilation,
});
export const pool = (kernel = 2, stride = kernel): Window => ({
  kind: 'pool',
  kernel,
  stride,
  padding: 0,
  dilation: 1,
});

/** Distance from the first to the last tap of the window, plus one: a dilated window is wider than its kernel. */
export const windowSpan = (w: Window): number => w.dilation * (w.kernel - 1) + 1;

/** floor((n + 2p - d(k-1) - 1) / s) + 1 */
export const outputSize = (n: number, w: Window): number =>
  Math.max(0, Math.floor((n + 2 * w.padding - windowSpan(w)) / w.stride) + 1);

/** Size produced by the matching transposed convolution (the shape of an up-sampling layer). */
export const transposedSize = (n: number, w: Window): number =>
  (n - 1) * w.stride - 2 * w.padding + windowSpan(w);

/** Padding that keeps the size for stride 1 ("same"). */
export const samePadding = (kernel: number, dilation = 1): number => (dilation * (kernel - 1)) / 2;

export interface StackStats {
  sizes: number[];
  /** Receptive field after each layer: how many input positions the central value depends on. */
  field: number[];
  /** Distance in input positions between neighbouring outputs after each layer. */
  jump: number[];
}

export function analyseStack(input: number, layers: Window[]): StackStats {
  const sizes = [input];
  const field = [1];
  const jump = [1];
  for (const w of layers) {
    sizes.push(outputSize(sizes[sizes.length - 1] as number, w));
    field.push(
      (field[field.length - 1] as number) + (windowSpan(w) - 1) * (jump[jump.length - 1] as number),
    );
    jump.push((jump[jump.length - 1] as number) * w.stride);
  }
  return { sizes, field, jump };
}

/**
 * The same receptive field found the slow way: follow the central output value back through every layer,
 * collecting the positions of the input it reads. Returns the span from the first to the last of them.
 */
export function tracedField(input: number, layers: Window[]): number {
  const sizes = analyseStack(input, layers).sizes;
  let reads = new Set<number>([Math.floor(((sizes[sizes.length - 1] as number) - 1) / 2)]);
  for (let l = layers.length - 1; l >= 0; l--) {
    const w = layers[l] as Window;
    const next = new Set<number>();
    for (const o of reads)
      for (let t = 0; t < w.kernel; t++) {
        const i = o * w.stride - w.padding + t * w.dilation;
        if (i >= 0 && i < (sizes[l] as number)) next.add(i);
      }
    reads = next;
  }
  const all = [...reads];
  return Math.max(...all) - Math.min(...all) + 1;
}

/** Weights and biases of a convolution with the given channel counts; `dims` is 1 for sequences, 2 for images. */
export const convParameters = (
  kernel: number,
  inChannels: number,
  outChannels: number,
  dims = 1,
): number => kernel ** dims * inChannels * outChannels + outChannels;
