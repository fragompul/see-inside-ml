import { Parameter, type Rng } from '@sim/core';

/**
 * Convolution and pooling over batches of single-sample tensors stored flat as
 * `(batch, channels, height, width)` in a Float64Array. Written for clarity first: the loops
 * mirror the textbook definitions so the code doubles as a reference.
 */

const at = (a: Float64Array, i: number) => a[i] as number;

/** "Valid" 2-D cross-correlation (what deep-learning libraries call convolution), stride 1. */
export class Conv2d {
  /** Layout `(outC, inC, k, k)`. */
  readonly weights: Float64Array;
  readonly biases: Float64Array;
  readonly weightParam: Parameter;
  readonly biasParam: Parameter;

  constructor(
    readonly inC: number,
    readonly outC: number,
    readonly k: number,
    rng: Rng,
  ) {
    if (![inC, outC, k].every((n) => Number.isInteger(n) && n >= 1)) {
      throw new RangeError('Conv2d sizes must be positive integers');
    }
    this.weights = new Float64Array(outC * inC * k * k);
    this.biases = new Float64Array(outC);
    const std = Math.sqrt(2 / (inC * k * k)); // He initialisation (ReLU follows)
    for (let i = 0; i < this.weights.length; i++) this.weights[i] = rng.normal() * std;
    this.weightParam = new Parameter(this.weights, true);
    this.biasParam = new Parameter(this.biases, false);
  }

  get parameters(): Parameter[] {
    return [this.weightParam, this.biasParam];
  }

  /** Output side length for an input side length. */
  outSize(side: number): number {
    return side - this.k + 1;
  }

  /** Input `(batch, inC, h, w)` → output `(batch, outC, h - k + 1, w - k + 1)`. */
  forward(input: Float64Array, batch: number, h: number, w: number): Float64Array {
    const { inC, outC, k } = this;
    const oh = h - k + 1;
    const ow = w - k + 1;
    if (oh < 1 || ow < 1) throw new RangeError(`Kernel ${k}×${k} does not fit a ${h}×${w} input`);
    if (input.length !== batch * inC * h * w)
      throw new RangeError('Conv2d input has the wrong size');
    const out = new Float64Array(batch * outC * oh * ow);
    for (let b = 0; b < batch; b++) {
      for (let oc = 0; oc < outC; oc++) {
        const outBase = (b * outC + oc) * oh * ow;
        out.fill(at(this.biases, oc), outBase, outBase + oh * ow);
        for (let ic = 0; ic < inC; ic++) {
          const inBase = (b * inC + ic) * h * w;
          for (let ky = 0; ky < k; ky++) {
            for (let kx = 0; kx < k; kx++) {
              const wv = at(this.weights, ((oc * inC + ic) * k + ky) * k + kx);
              for (let oy = 0; oy < oh; oy++) {
                const inRow = inBase + (oy + ky) * w + kx;
                const outRow = outBase + oy * ow;
                for (let ox = 0; ox < ow; ox++) {
                  out[outRow + ox] = at(out, outRow + ox) + wv * at(input, inRow + ox);
                }
              }
            }
          }
        }
      }
    }
    return out;
  }

  /**
   * Backprop. Fills the parameter gradients (averaged over the batch, matching the dense layers)
   * and returns dL/dInput, or null when the caller does not need it (the first layer).
   */
  backward(
    dOut: Float64Array,
    input: Float64Array,
    batch: number,
    h: number,
    w: number,
    needInput = true,
  ): Float64Array | null {
    const { inC, outC, k } = this;
    const oh = h - k + 1;
    const ow = w - k + 1;
    const dW = this.weightParam.grad;
    const dB = this.biasParam.grad;
    dW.fill(0);
    dB.fill(0);
    const dIn = needInput ? new Float64Array(input.length) : null;

    for (let b = 0; b < batch; b++) {
      for (let oc = 0; oc < outC; oc++) {
        const outBase = (b * outC + oc) * oh * ow;
        for (let i = 0; i < oh * ow; i++) dB[oc] = at(dB, oc) + at(dOut, outBase + i);
        for (let ic = 0; ic < inC; ic++) {
          const inBase = (b * inC + ic) * h * w;
          for (let ky = 0; ky < k; ky++) {
            for (let kx = 0; kx < k; kx++) {
              const wi = ((oc * inC + ic) * k + ky) * k + kx;
              const wv = at(this.weights, wi);
              let gradW = 0;
              for (let oy = 0; oy < oh; oy++) {
                const inRow = inBase + (oy + ky) * w + kx;
                const outRow = outBase + oy * ow;
                for (let ox = 0; ox < ow; ox++) {
                  const g = at(dOut, outRow + ox);
                  gradW += g * at(input, inRow + ox);
                  if (dIn) dIn[inRow + ox] = at(dIn, inRow + ox) + g * wv;
                }
              }
              dW[wi] = at(dW, wi) + gradW;
            }
          }
        }
      }
    }
    for (let i = 0; i < dW.length; i++) dW[i] = at(dW, i) / batch;
    for (let i = 0; i < dB.length; i++) dB[i] = at(dB, i) / batch;
    return dIn;
  }
}

export interface PoolResult {
  out: Float64Array;
  /** Flat index (into the input) of the winner of every output cell, for backprop. */
  argmax: Int32Array;
}

/** 2×2 max pooling with stride 2 (odd trailing rows / columns are dropped). */
export function maxPool2(
  input: Float64Array,
  batch: number,
  channels: number,
  h: number,
  w: number,
): PoolResult {
  const oh = Math.floor(h / 2);
  const ow = Math.floor(w / 2);
  const out = new Float64Array(batch * channels * oh * ow);
  const argmax = new Int32Array(out.length);
  for (let bc = 0; bc < batch * channels; bc++) {
    const inBase = bc * h * w;
    const outBase = bc * oh * ow;
    for (let oy = 0; oy < oh; oy++) {
      for (let ox = 0; ox < ow; ox++) {
        let best = -Infinity;
        let bestIndex = 0;
        for (let dy = 0; dy < 2; dy++) {
          for (let dx = 0; dx < 2; dx++) {
            const index = inBase + (oy * 2 + dy) * w + ox * 2 + dx;
            const v = at(input, index);
            if (v > best) {
              best = v;
              bestIndex = index;
            }
          }
        }
        out[outBase + oy * ow + ox] = best;
        argmax[outBase + oy * ow + ox] = bestIndex;
      }
    }
  }
  return { out, argmax };
}

/** Routes each output gradient back to the input cell that won the max. */
export function maxPool2Backward(
  dOut: Float64Array,
  argmax: Int32Array,
  inputLength: number,
): Float64Array {
  const dIn = new Float64Array(inputLength);
  for (let i = 0; i < dOut.length; i++) {
    const target = argmax[i] as number;
    dIn[target] = at(dIn, target) + at(dOut, i);
  }
  return dIn;
}

export const relu = (x: Float64Array): Float64Array => x.map((v) => (v > 0 ? v : 0));

/** dL/dz for ReLU: pass the gradient where the pre-activation was positive. */
export function reluBackward(dOut: Float64Array, pre: Float64Array): Float64Array {
  const out = new Float64Array(dOut.length);
  for (let i = 0; i < out.length; i++) out[i] = at(pre, i) > 0 ? at(dOut, i) : 0;
  return out;
}

/** Applies one `k × k` kernel to a single-channel image (valid, stride 1): used by the explainer. */
export function convolveSingle(
  image: ArrayLike<number>,
  size: number,
  kernel: ArrayLike<number>,
  k: number,
): Float64Array {
  const o = size - k + 1;
  const out = new Float64Array(o * o);
  for (let y = 0; y < o; y++) {
    for (let x = 0; x < o; x++) {
      let sum = 0;
      for (let ky = 0; ky < k; ky++) {
        for (let kx = 0; kx < k; kx++)
          sum += (image[(y + ky) * size + x + kx] as number) * (kernel[ky * k + kx] as number);
      }
      out[y * o + x] = sum;
    }
  }
  return out;
}
