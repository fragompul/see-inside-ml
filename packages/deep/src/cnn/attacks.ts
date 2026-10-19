import { argmax, type CnnNetwork } from './network';

/**
 * Adversarial examples and Grad-CAM for the digit CNN.
 *
 * An adversarial example is an image changed by an amount too small to matter to a person that
 * nevertheless flips the network's answer. FGSM takes one step of size ε in the direction that
 * raises the loss fastest (the sign of the input gradient); PGD repeats smaller steps and projects
 * back into the ε-ball around the original, which is a much stronger attack.
 */
export type AttackMethod = 'fgsm' | 'pgd';

export interface AttackOptions {
  method: AttackMethod;
  /** Largest change allowed per pixel (L∞ budget), in pixel units where the image is in [0, 1]. */
  epsilon: number;
  /** Number of PGD iterations (ignored by FGSM). */
  steps?: number;
  /** PGD step size; default 2.5·ε / steps. */
  stepSize?: number;
  /** Aim at this class instead of just leaving the true one (descends its loss). */
  targetClass?: number;
}

export interface AttackResult {
  adversarial: Float64Array;
  /** `adversarial − original`. */
  perturbation: Float64Array;
  before: { label: number; probs: number[] };
  after: { label: number; probs: number[] };
  /** Did the predicted class change (or reach `targetClass`)? */
  fooled: boolean;
}

const clip01 = (v: number) => Math.min(1, Math.max(0, v));

export function attack(
  net: CnnNetwork,
  image: Float64Array,
  trueLabel: number,
  options: AttackOptions,
): AttackResult {
  const { method, epsilon, targetClass } = options;
  const steps = method === 'pgd' ? Math.max(1, options.steps ?? 10) : 1;
  const stepSize = method === 'pgd' ? (options.stepSize ?? (2.5 * epsilon) / steps) : epsilon;
  // untargeted: climb the loss of the true class; targeted: descend the loss of the target class
  const aim = targetClass ?? trueLabel;
  const direction = targetClass === undefined ? 1 : -1;
  let x = Float64Array.from(image);
  for (let s = 0; s < steps; s++) {
    const { grad } = net.inputGradient(x, aim);
    x = x.map((v, i) => {
      const moved = v + direction * stepSize * Math.sign(grad[i] as number);
      // stay inside the ε-ball around the original and inside the valid pixel range
      const lo = (image[i] as number) - epsilon;
      const hi = (image[i] as number) + epsilon;
      return clip01(Math.min(hi, Math.max(lo, moved)));
    });
  }
  const beforeProbs = net.predict(image);
  const afterProbs = net.predict(x);
  const after = argmax(afterProbs);
  return {
    adversarial: x,
    perturbation: x.map((v, i) => v - (image[i] as number)),
    before: { label: argmax(beforeProbs), probs: beforeProbs },
    after: { label: after, probs: afterProbs },
    fooled: targetClass === undefined ? after !== trueLabel : after === targetClass,
  };
}

/** Bilinear enlargement of a square map to `side × side`, aligned on the cell centres. */
export function upsample(map: Float64Array, from: number, side: number): Float64Array {
  const out = new Float64Array(side * side);
  const scale = from / side;
  const at = (r: number, c: number) => map[r * from + c] as number;
  for (let y = 0; y < side; y++)
    for (let x = 0; x < side; x++) {
      const fy = Math.min(from - 1, Math.max(0, (y + 0.5) * scale - 0.5));
      const fx = Math.min(from - 1, Math.max(0, (x + 0.5) * scale - 0.5));
      const y0 = Math.floor(fy);
      const x0 = Math.floor(fx);
      const y1 = Math.min(from - 1, y0 + 1);
      const x1 = Math.min(from - 1, x0 + 1);
      const ty = fy - y0;
      const tx = fx - x0;
      out[y * side + x] =
        at(y0, x0) * (1 - ty) * (1 - tx) +
        at(y0, x1) * (1 - ty) * tx +
        at(y1, x0) * ty * (1 - tx) +
        at(y1, x1) * ty * tx;
    }
  return out;
}

export interface GradCam {
  /** The coarse class-activation map on the last convolution's grid, ReLU'd and scaled to [0, 1]. */
  coarse: Float64Array;
  /** The same map enlarged to the image size, for overlaying. */
  map: Float64Array;
  /** How much each feature map of the last convolution votes for the class (the "α" weights). */
  channelWeights: Float64Array;
  /** Class whose evidence is shown. */
  target: number;
}

/**
 * Grad-CAM: which parts of the image made the network say `target` (default: its own answer).
 * Each feature map of the last convolution is weighted by the average gradient of the class score
 * with respect to it, the maps are added up, and negative evidence is dropped. For this network
 * the score is linear in those feature maps (a dense layer), so the gradient is simply the dense
 * layer's weights.
 */
export function gradCam(net: CnnNetwork, image: Float64Array, target = -1): GradCam {
  const trace = net.trace(image);
  const cls = target >= 0 ? target : argmax(trace.probs);
  const [channels, side] = trace.shapes.conv2;
  const cells = side * side;
  const weights = new Float64Array(channels);
  for (let k = 0; k < channels; k++) {
    let sum = 0;
    for (let p = 0; p < cells; p++) sum += net.dense.weights.get(k * cells + p, cls);
    weights[k] = sum / cells;
  }
  const coarse = new Float64Array(cells);
  for (let k = 0; k < channels; k++)
    for (let p = 0; p < cells; p++)
      coarse[p] =
        (coarse[p] as number) + (weights[k] as number) * (trace.conv2[k * cells + p] as number);
  let max = 0;
  for (let p = 0; p < cells; p++) {
    coarse[p] = Math.max(0, coarse[p] as number);
    max = Math.max(max, coarse[p] as number);
  }
  if (max > 0) for (let p = 0; p < cells; p++) coarse[p] = (coarse[p] as number) / max;
  return { coarse, map: upsample(coarse, side, net.size), channelWeights: weights, target: cls };
}

export interface EvidenceMap {
  /** Signed evidence at each cell of the last convolution's grid: positive supports the class. */
  signed: Float64Array;
  /** Positive evidence only, scaled to [0, 1] and enlarged to the image size. */
  map: Float64Array;
  /** The class score (logit) minus its bias: exactly `Σ signed`. */
  score: number;
  target: number;
}

/**
 * The exact, position-by-position version of Grad-CAM for this network. The class score is a dense
 * layer over the last feature maps, so the contribution of cell `p` of map `k` is
 * `weight[k, p, class] × activation[k, p]` (gradient × activation). Summing over the maps gives the
 * evidence each location adds to the score, and the contributions add up to the logit exactly.
 * Grad-CAM replaces the position-specific weights by their average over the map; here that
 * average throws away real information, which is why the two can disagree.
 */
export function evidenceMap(net: CnnNetwork, image: Float64Array, target = -1): EvidenceMap {
  const trace = net.trace(image);
  const cls = target >= 0 ? target : argmax(trace.probs);
  const [channels, side] = trace.shapes.conv2;
  const cells = side * side;
  const signed = new Float64Array(cells);
  for (let k = 0; k < channels; k++)
    for (let p = 0; p < cells; p++)
      signed[p] =
        (signed[p] as number) +
        net.dense.weights.get(k * cells + p, cls) * (trace.conv2[k * cells + p] as number);
  const positive = signed.map((v) => Math.max(0, v));
  const max = Math.max(...positive);
  const scaled = max > 0 ? positive.map((v) => v / max) : positive;
  return {
    signed,
    map: upsample(scaled, side, net.size),
    score: signed.reduce((a, b) => a + b, 0),
    target: cls,
  };
}

/** Accuracy of the network on `images` after attacking each one with the same budget. */
export function robustAccuracy(
  net: CnnNetwork,
  images: Float64Array,
  labels: ArrayLike<number>,
  options: AttackOptions,
): number {
  const px = net.size * net.size;
  const count = labels.length;
  let correct = 0;
  for (let i = 0; i < count; i++) {
    const img = images.slice(i * px, (i + 1) * px);
    const result = attack(net, img, labels[i] as number, options);
    if (result.after.label === labels[i]) correct++;
  }
  return correct / count;
}
