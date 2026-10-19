import { DenseLayer } from '../dense';
import { lossValue, outputGradient, Matrix, type Parameter, type Rng } from '@sim/core';
import { DIGIT_CLASSES, IMAGE_SIZE } from './digits';
import { Conv2d, maxPool2, maxPool2Backward, relu, reluBackward } from './layers';

export interface CnnConfig {
  /** Feature maps of the first convolution. */
  filters1: number;
  /** Feature maps of the second convolution. */
  filters2: number;
  /** Kernel side (3 or 5). */
  kernel: 3 | 5;
}

export const DEFAULT_CNN: CnnConfig = { filters1: 6, filters2: 12, kernel: 3 };

export const CNN_LIMITS = { filters1: [2, 12], filters2: [2, 16], kernels: [3, 5] } as const;

export function validateCnn(c: CnnConfig): void {
  const inRange = (v: number, [lo, hi]: readonly [number, number]) =>
    Number.isInteger(v) && v >= lo && v <= hi;
  if (!inRange(c.filters1, CNN_LIMITS.filters1)) throw new RangeError('filters1 must be 2..12');
  if (!inRange(c.filters2, CNN_LIMITS.filters2)) throw new RangeError('filters2 must be 2..16');
  if (c.kernel !== 3 && c.kernel !== 5) throw new RangeError('kernel must be 3 or 5');
}

/** Everything one image produced on its way through the network (for visualisation). */
export interface CnnTrace {
  /** `size²` input pixels. */
  input: Float64Array;
  /** Shapes of each tensor, as [channels, side]. */
  shapes: { conv1: [number, number]; pool1: [number, number]; conv2: [number, number] };
  /** Before ReLU / after ReLU feature maps of the first convolution, `(f1, h1, h1)`. */
  conv1Pre: Float64Array;
  conv1: Float64Array;
  /** After 2×2 max pooling, `(f1, p, p)`. */
  pool1: Float64Array;
  conv2Pre: Float64Array;
  conv2: Float64Array;
  logits: number[];
  probs: number[];
}

/**
 * conv(1→f1) → ReLU → maxpool 2 → conv(f1→f2) → ReLU → flatten → dense(10) → softmax,
 * with a hand-written backward pass.
 */
export class CnnNetwork {
  readonly size = IMAGE_SIZE;
  readonly classes = DIGIT_CLASSES;
  readonly conv1: Conv2d;
  readonly conv2: Conv2d;
  readonly dense: DenseLayer;
  readonly h1: number;
  readonly p1: number;
  readonly h2: number;

  private cache: {
    batch: number;
    x: Float64Array;
    pre1: Float64Array;
    act1: Float64Array;
    pooled: Float64Array;
    argmax: Int32Array;
    pre2: Float64Array;
  } | null = null;

  constructor(
    readonly config: CnnConfig,
    rng: Rng,
  ) {
    validateCnn(config);
    const { filters1, filters2, kernel } = config;
    this.h1 = this.size - kernel + 1;
    this.p1 = Math.floor(this.h1 / 2);
    this.h2 = this.p1 - kernel + 1;
    if (this.h2 < 1) throw new RangeError('The image is too small for this kernel size');
    this.conv1 = new Conv2d(1, filters1, kernel, rng);
    this.conv2 = new Conv2d(filters1, filters2, kernel, rng);
    this.dense = new DenseLayer(filters2 * this.h2 * this.h2, this.classes, 'softmax', rng);
  }

  get parameters(): Parameter[] {
    return [...this.conv1.parameters, ...this.conv2.parameters, ...this.dense.parameters];
  }

  get parameterCount(): number {
    return this.parameters.reduce((sum, p) => sum + p.data.length, 0);
  }

  /** Forward pass over `batch` images (`batch × size²`); returns class probabilities. */
  forward(images: Float64Array, batch: number): Matrix {
    const { size, h1, p1, h2, config } = this;
    const pre1 = this.conv1.forward(images, batch, size, size);
    const act1 = relu(pre1);
    const { out: pooled, argmax } = maxPool2(act1, batch, config.filters1, h1, h1);
    const pre2 = this.conv2.forward(pooled, batch, p1, p1);
    const act2 = relu(pre2);
    this.cache = { batch, x: images, pre1, act1, pooled, argmax, pre2 };
    const flat = new Matrix(batch, config.filters2 * h2 * h2, act2);
    return this.dense.forward(flat);
  }

  /** Backprop for the batch of the last `forward` (targets one-hot, batch × classes). */
  backward(targets: Matrix, probs: Matrix): void {
    const c = this.cache;
    if (!c) throw new Error('backward() called before forward()');
    const { size, p1 } = this;
    const dZ = outputGradient('cce', probs, targets);
    const dFlat = this.dense.backwardFromPreActivation(dZ);
    const dPre2 = reluBackward(dFlat.data, c.pre2);
    const dPooled = this.conv2.backward(dPre2, c.pooled, c.batch, p1, p1) as Float64Array;
    const dAct1 = maxPool2Backward(dPooled, c.argmax, c.act1.length);
    const dPre1 = reluBackward(dAct1, c.pre1);
    this.conv1.backward(dPre1, c.x, c.batch, size, size, false);
  }

  /** Mean cross-entropy of `probs` against one-hot `targets`. */
  loss(probs: Matrix, targets: Matrix): number {
    return lossValue('cce', probs, targets);
  }

  /** Runs one image and keeps every intermediate tensor. Does not disturb a pending backward. */
  trace(image: Float64Array): CnnTrace {
    const { h1, p1, h2, config } = this;
    const saved = this.cache;
    const savedDense = { inputs: this.dense.inputs, z: this.dense.z, a: this.dense.a };
    const probsMatrix = this.forward(image, 1);
    const c = this.cache as NonNullable<typeof this.cache>;
    const logits = Array.from((this.dense.z as Matrix).data);
    const result: CnnTrace = {
      input: Float64Array.from(image),
      shapes: {
        conv1: [config.filters1, h1],
        pool1: [config.filters1, p1],
        conv2: [config.filters2, h2],
      },
      conv1Pre: c.pre1,
      conv1: c.act1,
      pool1: c.pooled,
      conv2Pre: c.pre2,
      conv2: relu(c.pre2),
      logits,
      probs: Array.from(probsMatrix.data),
    };
    // restore what a training step may still depend on
    this.cache = saved;
    this.dense.inputs = savedDense.inputs;
    this.dense.z = savedDense.z;
    this.dense.a = savedDense.a;
    return result;
  }

  /**
   * Gradient of the cross-entropy loss for class `target` with respect to the *pixels* of one
   * image: which pixels, nudged which way, change the loss the most. Parameters, gradients and
   * caches are left exactly as they were, so it is safe to call between training steps.
   */
  inputGradient(
    image: Float64Array,
    target: number,
  ): { grad: Float64Array; probs: number[]; loss: number } {
    const { size, p1 } = this;
    const savedCache = this.cache;
    const savedDense = {
      inputs: this.dense.inputs,
      z: this.dense.z,
      a: this.dense.a,
      delta: this.dense.delta,
    };
    const savedGrads = this.parameters.map((p) => p.grad.slice());
    const probs = this.forward(image, 1);
    const targets = new Matrix(1, this.classes);
    targets.set(0, target, 1);
    const c = this.cache as NonNullable<typeof this.cache>;
    const dZ = outputGradient('cce', probs, targets);
    const dFlat = this.dense.backwardFromPreActivation(dZ);
    const dPre2 = reluBackward(dFlat.data, c.pre2);
    const dPooled = this.conv2.backward(dPre2, c.pooled, 1, p1, p1) as Float64Array;
    const dAct1 = maxPool2Backward(dPooled, c.argmax, c.act1.length);
    const dPre1 = reluBackward(dAct1, c.pre1);
    const grad = this.conv1.backward(dPre1, c.x, 1, size, size, true) as Float64Array;
    const result = { grad, probs: Array.from(probs.data), loss: this.loss(probs, targets) };
    this.cache = savedCache;
    this.dense.inputs = savedDense.inputs;
    this.dense.z = savedDense.z;
    this.dense.a = savedDense.a;
    this.dense.delta = savedDense.delta;
    this.parameters.forEach((p, i) => p.grad.set(savedGrads[i] as Float64Array));
    return result;
  }

  /**
   * Gradient of the class *score* (the logit before the softmax) with respect to the pixels of one image. Unlike the
   * loss gradient this does not vanish when the network is already sure, which is what attribution methods need.
   * Parameters, gradients and caches are left as they were.
   */
  logitGradient(image: Float64Array, target: number): { grad: Float64Array; logit: number } {
    const { size, p1 } = this;
    const savedCache = this.cache;
    const savedDense = {
      inputs: this.dense.inputs,
      z: this.dense.z,
      a: this.dense.a,
      delta: this.dense.delta,
    };
    const savedGrads = this.parameters.map((p) => p.grad.slice());
    this.forward(image, 1);
    const c = this.cache as NonNullable<typeof this.cache>;
    const logit = (this.dense.z as Matrix).get(0, target);
    const dZ = new Matrix(1, this.classes);
    dZ.set(0, target, 1);
    const dFlat = this.dense.backwardFromPreActivation(dZ);
    const dPre2 = reluBackward(dFlat.data, c.pre2);
    const dPooled = this.conv2.backward(dPre2, c.pooled, 1, p1, p1) as Float64Array;
    const dAct1 = maxPool2Backward(dPooled, c.argmax, c.act1.length);
    const dPre1 = reluBackward(dAct1, c.pre1);
    const grad = this.conv1.backward(dPre1, c.x, 1, size, size, true) as Float64Array;
    this.cache = savedCache;
    this.dense.inputs = savedDense.inputs;
    this.dense.z = savedDense.z;
    this.dense.a = savedDense.a;
    this.dense.delta = savedDense.delta;
    this.parameters.forEach((p, i) => p.grad.set(savedGrads[i] as Float64Array));
    return { grad, logit };
  }

  /**
   * Gradient, with respect to the pixels, of the mean activation of one feature map of the last convolution, and that
   * mean activation. Used to ask what a single unit of the network is looking for.
   */
  channelGradient(
    image: Float64Array,
    channel: number,
  ): { grad: Float64Array; activation: number } {
    const { size, p1, h2 } = this;
    const savedCache = this.cache;
    const savedDense = {
      inputs: this.dense.inputs,
      z: this.dense.z,
      a: this.dense.a,
      delta: this.dense.delta,
    };
    const savedGrads = this.parameters.map((p) => p.grad.slice());
    this.forward(image, 1);
    const c = this.cache as NonNullable<typeof this.cache>;
    const cells = h2 * h2;
    let activation = 0;
    const dAct = new Float64Array(this.config.filters2 * cells);
    for (let p = 0; p < cells; p++) {
      const pre = c.pre2[channel * cells + p] as number;
      activation += Math.max(0, pre) / cells;
      dAct[channel * cells + p] = 1 / cells;
    }
    const dPre2 = reluBackward(dAct, c.pre2);
    const dPooled = this.conv2.backward(dPre2, c.pooled, 1, p1, p1) as Float64Array;
    const dAct1 = maxPool2Backward(dPooled, c.argmax, c.act1.length);
    const dPre1 = reluBackward(dAct1, c.pre1);
    const grad = this.conv1.backward(dPre1, c.x, 1, size, size, true) as Float64Array;
    this.cache = savedCache;
    this.dense.inputs = savedDense.inputs;
    this.dense.z = savedDense.z;
    this.dense.a = savedDense.a;
    this.dense.delta = savedDense.delta;
    this.parameters.forEach((p, i) => p.grad.set(savedGrads[i] as Float64Array));
    return { grad, activation };
  }

  /** Class probabilities of one image (cheap: no trace; leaves any pending backward intact). */
  predict(image: Float64Array): number[] {
    const saved = this.cache;
    const savedDense = { inputs: this.dense.inputs, z: this.dense.z, a: this.dense.a };
    const probs = Array.from(this.forward(image, 1).data);
    this.cache = saved;
    this.dense.inputs = savedDense.inputs;
    this.dense.z = savedDense.z;
    this.dense.a = savedDense.a;
    return probs;
  }
}

/** Arg-max of a probability vector. */
export const argmax = (values: ArrayLike<number>): number => {
  let best = 0;
  for (let i = 1; i < values.length; i++)
    if ((values[i] as number) > (values[best] as number)) best = i;
  return best;
};

/**
 * Occlusion sensitivity: slide a blank patch over the image and record how much the network's
 * confidence in `target` drops. Bright cells are pixels the decision depends on.
 */
export function occlusionMap(
  net: CnnNetwork,
  image: Float64Array,
  target: number,
  patch = 3,
): Float64Array {
  const size = net.size;
  const base = net.predict(image)[target] as number;
  const out = new Float64Array(size * size);
  const counts = new Float64Array(size * size);
  const half = Math.floor(patch / 2);
  const work = new Float64Array(image.length);
  for (let cy = 0; cy < size; cy++) {
    for (let cx = 0; cx < size; cx++) {
      work.set(image);
      for (let y = cy - half; y <= cy + half; y++) {
        for (let x = cx - half; x <= cx + half; x++) {
          if (y >= 0 && y < size && x >= 0 && x < size) work[y * size + x] = 0;
        }
      }
      const drop = base - (net.predict(work)[target] as number);
      for (let y = cy - half; y <= cy + half; y++) {
        for (let x = cx - half; x <= cx + half; x++) {
          if (y >= 0 && y < size && x >= 0 && x < size) {
            out[y * size + x] = (out[y * size + x] as number) + drop;
            counts[y * size + x] = (counts[y * size + x] as number) + 1;
          }
        }
      }
    }
  }
  for (let i = 0; i < out.length; i++)
    out[i] = (out[i] as number) / Math.max(1, counts[i] as number);
  return out;
}
