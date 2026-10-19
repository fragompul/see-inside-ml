import { createRng } from '@sim/core';
import { IMAGE_SIZE } from './digits';
import type { CnnNetwork } from './network';

/**
 * Feature visualisation: find the picture a unit likes best by gradient ascent on the pixels. Left alone the
 * ascent finds high-frequency patterns that excite the unit but look like noise, so the image is kept in the
 * valid range and, optionally, blurred a little at every step so that only smooth, picture-like solutions survive.
 */
export type Target = { kind: 'class'; index: number } | { kind: 'channel'; index: number };

export interface VisualiseOptions {
  steps: number;
  stepSize: number;
  /** Strength of the blur applied after every step (0 = none, 1 = replace each pixel by the 3 x 3 average). */
  blur: number;
  /** Fraction of the image removed towards zero after every step. */
  decay: number;
  seed: number;
}

export const DEFAULT_VISUALISE: VisualiseOptions = {
  steps: 120,
  stepSize: 0.04,
  blur: 0,
  decay: 0,
  seed: 1,
};

const S = IMAGE_SIZE;

export function blurImage(x: Float64Array, amount: number): Float64Array {
  if (amount <= 0) return x;
  const out = new Float64Array(x.length);
  for (let r = 0; r < S; r++)
    for (let c = 0; c < S; c++) {
      let sum = 0;
      let n = 0;
      for (let dr = -1; dr <= 1; dr++)
        for (let dc = -1; dc <= 1; dc++) {
          const rr = r + dr;
          const cc = c + dc;
          if (rr >= 0 && rr < S && cc >= 0 && cc < S) {
            sum += x[rr * S + cc] as number;
            n++;
          }
        }
      out[r * S + c] = (1 - amount) * (x[r * S + c] as number) + amount * (sum / n);
    }
  return out;
}

/** Total variation: the sum of absolute differences between neighbouring pixels (large for noisy pictures). */
export function totalVariation(x: ArrayLike<number>): number {
  let tv = 0;
  for (let r = 0; r < S; r++)
    for (let c = 0; c < S; c++) {
      if (c + 1 < S) tv += Math.abs((x[r * S + c] as number) - (x[r * S + c + 1] as number));
      if (r + 1 < S) tv += Math.abs((x[r * S + c] as number) - (x[(r + 1) * S + c] as number));
    }
  return tv;
}

const score = (net: CnnNetwork, x: Float64Array, target: Target) =>
  target.kind === 'class'
    ? net.logitGradient(x, target.index)
    : (() => {
        const { grad, activation } = net.channelGradient(x, target.index);
        return { grad, logit: activation };
      })();

export interface Visualisation {
  image: Float64Array;
  /** The unit's score (logit or mean activation) at the start and at the end. */
  startScore: number;
  endScore: number;
  /** Score after every step. */
  trace: number[];
}

export function visualise(
  net: CnnNetwork,
  target: Target,
  options: VisualiseOptions,
): Visualisation {
  const rng = createRng(options.seed);
  let x: Float64Array = Float64Array.from({ length: S * S }, () =>
    Math.min(1, Math.max(0, 0.5 + 0.05 * rng.normal())),
  );
  const first = score(net, x, target);
  const trace = [first.logit];
  for (let step = 0; step < options.steps; step++) {
    const { grad } = score(net, x, target);
    let norm = 0;
    for (const g of grad) norm += g * g;
    norm = Math.sqrt(norm) + 1e-12;
    x = x.map((v, i) =>
      Math.min(
        1,
        Math.max(0, v + ((options.stepSize * (grad[i] as number)) / norm) * Math.sqrt(S * S)),
      ),
    );
    if (options.blur > 0) x = blurImage(x, options.blur);
    if (options.decay > 0) x = x.map((v) => v * (1 - options.decay));
    trace.push(score(net, x, target).logit);
  }
  return { image: x, startScore: first.logit, endScore: trace[trace.length - 1] as number, trace };
}
