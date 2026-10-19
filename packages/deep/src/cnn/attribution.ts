import { createRng } from '@sim/core';
import { argmax, type CnnNetwork } from './network';

/**
 * Pixel attribution for the digit CNN: which pixels is the score of a class owed to?
 *  - saliency: the size of the gradient of the class score with respect to each pixel
 *  - gradient x input: the gradient times the pixel, a first-order account of the score
 *  - SmoothGrad: the gradient averaged over noisy copies of the image, to remove its jitter
 *  - integrated gradients: the gradient averaged along the straight path from a baseline (a blank image) to the
 *    image, times (image - baseline). Its attributions add up exactly to score(image) - score(baseline).
 */
export type AttributionMethod = 'saliency' | 'gradient-input' | 'smoothgrad' | 'integrated';

export const ATTRIBUTION_METHODS: AttributionMethod[] = [
  'saliency',
  'gradient-input',
  'smoothgrad',
  'integrated',
];

const logit = (net: CnnNetwork, image: Float64Array, target: number): number =>
  net.logitGradient(image, target).logit;

export function saliency(net: CnnNetwork, image: Float64Array, target: number): Float64Array {
  return net.logitGradient(image, target).grad.map(Math.abs);
}

export function gradientTimesInput(
  net: CnnNetwork,
  image: Float64Array,
  target: number,
): Float64Array {
  const { grad } = net.logitGradient(image, target);
  return grad.map((g, i) => g * (image[i] as number));
}

export function smoothGrad(
  net: CnnNetwork,
  image: Float64Array,
  target: number,
  samples = 25,
  noise = 0.15,
  seed = 1,
): Float64Array {
  const rng = createRng(seed);
  const sum = new Float64Array(image.length);
  for (let s = 0; s < samples; s++) {
    const noisy = image.map((v) => v + noise * rng.normal());
    const { grad } = net.logitGradient(noisy, target);
    for (let i = 0; i < sum.length; i++)
      sum[i] = (sum[i] as number) + Math.abs(grad[i] as number) / samples;
  }
  return sum;
}

/** Integrated gradients by the midpoint rule with `steps` steps from `baseline` (default: all zeros). */
export function integratedGradients(
  net: CnnNetwork,
  image: Float64Array,
  target: number,
  steps = 32,
  baseline: Float64Array = new Float64Array(image.length),
): Float64Array {
  const sum = new Float64Array(image.length);
  for (let k = 0; k < steps; k++) {
    const alpha = (k + 0.5) / steps;
    const point = image.map(
      (v, i) => (baseline[i] as number) + alpha * (v - (baseline[i] as number)),
    );
    const { grad } = net.logitGradient(point, target);
    for (let i = 0; i < sum.length; i++) sum[i] = (sum[i] as number) + (grad[i] as number) / steps;
  }
  return sum.map((g, i) => g * ((image[i] as number) - (baseline[i] as number)));
}

/** `score(image) - score(baseline)` for the completeness check of integrated gradients. */
export function scoreGap(
  net: CnnNetwork,
  image: Float64Array,
  target: number,
  baseline: Float64Array = new Float64Array(image.length),
): number {
  return logit(net, image, target) - logit(net, baseline, target);
}

export function attribute(
  method: AttributionMethod,
  net: CnnNetwork,
  image: Float64Array,
  target = -1,
): { map: Float64Array; target: number } {
  const cls = target >= 0 ? target : argmax(net.predict(image));
  const map =
    method === 'saliency'
      ? saliency(net, image, cls)
      : method === 'gradient-input'
        ? gradientTimesInput(net, image, cls)
        : method === 'smoothgrad'
          ? smoothGrad(net, image, cls)
          : integratedGradients(net, image, cls);
  return { map, target: cls };
}

/**
 * Faithfulness by deletion: remove the pixels one chunk at a time, most important first (blanking them), and
 * record the class score after each chunk. A good attribution makes the score fall quickly. Returns `points + 1`
 * scores, from "nothing removed" to "everything removed", and their average (the area under the curve).
 */
export function deletionCurve(
  net: CnnNetwork,
  image: Float64Array,
  target: number,
  order: ArrayLike<number>,
  points = 20,
): { scores: number[]; area: number } {
  const x = Float64Array.from(image);
  const total = x.length;
  const scores = [logit(net, x, target)];
  let removed = 0;
  for (let p = 1; p <= points; p++) {
    const upTo = Math.round((p / points) * total);
    for (; removed < upTo; removed++) x[order[removed] as number] = 0;
    scores.push(logit(net, x, target));
  }
  return { scores, area: scores.reduce((a, b) => a + b, 0) / scores.length };
}

/** Pixel indices from most to least important by the given map (larger value = more important). */
export const rankPixels = (map: ArrayLike<number>): number[] =>
  Array.from({ length: map.length }, (_, i) => i).sort(
    (a, b) => (map[b] as number) - (map[a] as number),
  );
