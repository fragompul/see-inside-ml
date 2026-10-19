import { type GridField } from '@sim/deep';
import { mix, type Palette, type Rgb } from '@/shared/canvas/colors';

/**
 * Painters that turn a `GridField` (row 0 = bottom of the plane) into an image whose top row is
 * the top of the plane. They write into a reusable low-resolution canvas; callers scale it up
 * with smoothing, which gives the soft, continuous look of a real decision surface.
 */

function ensureCanvas(canvas: HTMLCanvasElement | null, size: number): HTMLCanvasElement {
  const target = canvas ?? document.createElement('canvas');
  if (target.width !== size || target.height !== size) {
    target.width = size;
    target.height = size;
  }
  return target;
}

function put(
  canvas: HTMLCanvasElement | null,
  size: number,
  fn: (set: (row: number, col: number, c: Rgb, alpha?: number) => void) => void,
): HTMLCanvasElement {
  const target = ensureCanvas(canvas, size);
  const ctx = target.getContext('2d');
  if (!ctx) return target;
  const image = ctx.createImageData(size, size);
  const data = image.data;
  fn((row, col, c, alpha = 1) => {
    // flip vertically: grid row 0 is the bottom of the plane
    const i = ((size - 1 - row) * size + col) * 4;
    data[i] = c[0];
    data[i + 1] = c[1];
    data[i + 2] = c[2];
    data[i + 3] = Math.round(alpha * 255);
  });
  ctx.putImageData(image, 0, 0);
  return target;
}

/** Diverging colour: 0 → neutral, +max → ember, −max → frost. */
export function divergingColor(value: number, maxAbs: number, palette: Palette, neutral: Rgb): Rgb {
  const t = maxAbs > 0 ? Math.max(-1, Math.min(1, value / maxAbs)) : 0;
  return t >= 0 ? mix(neutral, palette.pos, t) : mix(neutral, palette.neg, -t);
}

/** Classification surface: class colours weighted by probability, stronger where the model is sure. */
export function paintClassification(
  canvas: HTMLCanvasElement | null,
  field: GridField,
  palette: Palette,
): HTMLCanvasElement {
  const { resolution: n, channels: k, values } = field;
  return put(canvas, n, (set) => {
    for (let row = 0; row < n; row++) {
      for (let col = 0; col < n; col++) {
        const base = (row * n + col) * k;
        let r = 0;
        let g = 0;
        let b = 0;
        let best = 0;
        for (let c = 0; c < k; c++) {
          const p = values[base + c] as number;
          const color = palette.classes[c % palette.classes.length] as Rgb;
          r += color[0] * p;
          g += color[1] * p;
          b += color[2] * p;
          if (p > best) best = p;
        }
        const confidence = k > 1 ? (best - 1 / k) / (1 - 1 / k) : 1;
        set(row, col, [r, g, b], 0.14 + 0.62 * Math.max(0, Math.min(1, confidence)));
      }
    }
  });
}

/** Regression surface: diverging colour of the predicted value. */
export function paintRegression(
  canvas: HTMLCanvasElement | null,
  field: GridField,
  palette: Palette,
  range: readonly [number, number],
): HTMLCanvasElement {
  const n = field.resolution;
  const maxAbs = Math.max(Math.abs(range[0]), Math.abs(range[1]), 1e-6);
  return put(canvas, n, (set) => {
    for (let row = 0; row < n; row++) {
      for (let col = 0; col < n; col++) {
        const v = field.values[row * n + col] as number;
        set(row, col, divergingColor(v, maxAbs, palette, palette.surface), 0.85);
      }
    }
  });
}

/** One neuron's response map (hidden neurons: diverging; class outputs: that class's colour). */
export function paintNeuron(
  canvas: HTMLCanvasElement | null,
  values: Float32Array,
  resolution: number,
  palette: Palette,
  tint?: Rgb,
): HTMLCanvasElement {
  let maxAbs = 0;
  for (let i = 0; i < values.length; i++) maxAbs = Math.max(maxAbs, Math.abs(values[i] as number));
  const scale = tint ? 1 : Math.max(maxAbs, 1e-6);
  return put(canvas, resolution, (set) => {
    for (let row = 0; row < resolution; row++) {
      for (let col = 0; col < resolution; col++) {
        const v = values[row * resolution + col] as number;
        const color = tint
          ? mix(palette.surface, tint, Math.max(0, Math.min(1, v)))
          : divergingColor(v, scale, palette, palette.surface);
        set(row, col, color);
      }
    }
  });
}

/** Samples a field at plane coordinates (nearest cell). Returns the channel values. */
export function sampleField(field: GridField, x: number, y: number, domain: number): number[] {
  const n = field.resolution;
  const col = Math.min(n - 1, Math.max(0, Math.floor(((x + domain) / (2 * domain)) * n)));
  const row = Math.min(n - 1, Math.max(0, Math.floor(((y + domain) / (2 * domain)) * n)));
  const base = (row * n + col) * field.channels;
  return Array.from(field.values.subarray(base, base + field.channels));
}
