import { describe, expect, it } from 'vitest';
import { createRng } from '@sim/core';
import { Conv2d, convolveSingle, maxPool2, maxPool2Backward, relu, reluBackward } from './layers';

const random = (n: number, seed: number) => {
  const rng = createRng(seed);
  return Float64Array.from({ length: n }, () => rng.normal());
};

/** Textbook definition, deliberately unoptimised: the reference the real layer must match. */
function naiveConv(
  input: Float64Array,
  w: Float64Array,
  b: Float64Array,
  B: number,
  inC: number,
  outC: number,
  k: number,
  h: number,
  wd: number,
) {
  const oh = h - k + 1;
  const ow = wd - k + 1;
  const out = new Float64Array(B * outC * oh * ow);
  for (let n = 0; n < B; n++)
    for (let oc = 0; oc < outC; oc++)
      for (let y = 0; y < oh; y++)
        for (let x = 0; x < ow; x++) {
          let s = b[oc] as number;
          for (let ic = 0; ic < inC; ic++)
            for (let ky = 0; ky < k; ky++)
              for (let kx = 0; kx < k; kx++)
                s +=
                  (input[((n * inC + ic) * h + y + ky) * wd + x + kx] as number) *
                  (w[((oc * inC + ic) * k + ky) * k + kx] as number);
          out[((n * outC + oc) * oh + y) * ow + x] = s;
        }
  return out;
}

describe('Conv2d', () => {
  it.each([
    [1, 3, 3, 8],
    [2, 4, 3, 9],
    [3, 2, 5, 10],
  ])('matches the textbook definition (inC=%i outC=%i k=%i size=%i)', (inC, outC, k, size) => {
    const conv = new Conv2d(inC, outC, k, createRng(1));
    conv.biases.set(random(outC, 5));
    const B = 3;
    const input = random(B * inC * size * size, 2);
    const fast = conv.forward(input, B, size, size);
    const slow = naiveConv(input, conv.weights, conv.biases, B, inC, outC, k, size, size);
    expect(fast).toHaveLength(slow.length);
    slow.forEach((v, i) => expect(fast[i]).toBeCloseTo(v, 12));
  });

  it('detects an edge: a vertical-edge kernel lights up exactly at the boundary', () => {
    const size = 6;
    const image = Float64Array.from({ length: 36 }, (_, i) => (i % 6 < 3 ? 0 : 1));
    const sobelX = [-1, 0, 1, -2, 0, 2, -1, 0, 1];
    const out = convolveSingle(image, size, sobelX, 3);
    const side = 4;
    // windows 0..3 cover columns 0-2, 1-3, 2-4, 3-5: only the two that straddle the edge respond
    for (let y = 0; y < side; y++) {
      expect(Array.from(out.slice(y * side, (y + 1) * side)).map((v) => Math.round(v))).toEqual([
        0, 4, 4, 0,
      ]);
    }
  });

  it('is translation-equivariant: shifting the input shifts the output', () => {
    const conv = new Conv2d(1, 2, 3, createRng(3));
    const size = 10;
    const a = new Float64Array(size * size);
    a[3 * size + 3] = 1;
    const b = new Float64Array(size * size);
    b[4 * size + 5] = 1; // same dot, moved by (+1, +2)
    const outA = conv.forward(a, 1, size, size);
    const outB = conv.forward(b, 1, size, size);
    const o = size - 2;
    for (let oc = 0; oc < 2; oc++)
      for (let y = 0; y < o - 1; y++)
        for (let x = 0; x < o - 2; x++)
          expect(outB[(oc * o + y + 1) * o + x + 2]).toBeCloseTo(
            outA[(oc * o + y) * o + x] as number,
            12,
          );
  });

  it('backward matches finite differences for weights, biases and inputs', () => {
    const conv = new Conv2d(2, 3, 3, createRng(4));
    conv.biases.set(random(3, 8));
    const B = 2;
    const size = 6;
    const input = random(B * 2 * size * size, 6);
    const outLen = B * 3 * 4 * 4;
    const upstream = random(outLen, 7);
    // L = Σ upstream · conv(input) / B   (the layer averages parameter grads over the batch)
    const loss = () => {
      const out = conv.forward(input, B, size, size);
      let s = 0;
      for (let i = 0; i < out.length; i++) s += (out[i] as number) * (upstream[i] as number);
      return s / B;
    };
    const dIn = conv.backward(upstream, input, B, size, size) as Float64Array;
    const eps = 1e-6;
    const check = (data: Float64Array, analytic: Float64Array, scale = 1) => {
      for (const i of [0, 1, Math.floor(data.length / 2), data.length - 1]) {
        const original = data[i] as number;
        data[i] = original + eps;
        const plus = loss();
        data[i] = original - eps;
        const minus = loss();
        data[i] = original;
        expect(((plus - minus) / (2 * eps)) * scale).toBeCloseTo(analytic[i] as number, 6);
      }
    };
    check(conv.weights, conv.weightParam.grad);
    check(conv.biases, conv.biasParam.grad);
    check(input, dIn, B); // dIn is not batch-averaged, the loss is
  });

  it('skips dL/dInput when asked and validates sizes', () => {
    const conv = new Conv2d(1, 1, 3, createRng(1));
    expect(conv.backward(new Float64Array(16), new Float64Array(36), 1, 6, 6, false)).toBeNull();
    expect(() => conv.forward(new Float64Array(4), 1, 2, 2)).toThrow(/does not fit/);
    expect(() => conv.forward(new Float64Array(5), 1, 6, 6)).toThrow(/wrong size/);
    expect(() => new Conv2d(0, 1, 3, createRng(1))).toThrow(/positive/);
  });
});

describe('max pooling', () => {
  const input = Float64Array.from([
    1,
    2,
    5,
    6, //
    3,
    4,
    7,
    8, //
    9,
    1,
    0,
    0, //
    2,
    3,
    0,
    1,
  ]);

  it('keeps the maximum of every 2×2 block and remembers where it was', () => {
    const { out, argmax } = maxPool2(input, 1, 1, 4, 4);
    expect(Array.from(out)).toEqual([4, 8, 9, 1]);
    expect(Array.from(argmax)).toEqual([5, 7, 8, 15]);
  });

  it('routes gradients only to the winners', () => {
    const { argmax } = maxPool2(input, 1, 1, 4, 4);
    const dIn = maxPool2Backward(Float64Array.from([10, 20, 30, 40]), argmax, 16);
    expect(dIn[5]).toBe(10);
    expect(dIn[7]).toBe(20);
    expect(dIn[8]).toBe(30);
    expect(dIn[15]).toBe(40);
    expect(dIn.reduce((a, b) => a + b, 0)).toBe(100);
  });

  it('drops an odd trailing row and column, per channel and sample', () => {
    const odd = random(2 * 3 * 5 * 5, 9);
    const { out } = maxPool2(odd, 2, 3, 5, 5);
    expect(out).toHaveLength(2 * 3 * 2 * 2);
  });
});

describe('relu', () => {
  it('clamps negatives and blocks their gradient', () => {
    const pre = Float64Array.from([-1, 0, 2]);
    expect(Array.from(relu(pre))).toEqual([0, 0, 2]);
    expect(Array.from(reluBackward(Float64Array.from([5, 5, 5]), pre))).toEqual([0, 0, 5]);
  });
});
