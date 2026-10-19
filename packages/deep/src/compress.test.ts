import { createRng } from '@sim/core';
import { describe, expect, it } from 'vitest';
import {
  cloneModel,
  pruneModel,
  quantizeModel,
  quantizeValues,
  softTargets,
  weightBytes,
} from './compress';
import { DEFAULT_TOOLBOX, ToolboxMLP } from './toolbox';
import { Matrix } from '@sim/core';

describe('quantisation', () => {
  const rng = createRng(1);
  const w = Float64Array.from({ length: 2000 }, () => rng.normal());

  it('the error never exceeds half a step and the number of levels is bounded by the bits', () => {
    for (const bits of [2, 3, 4, 8]) {
      const q = quantizeValues(w, bits);
      expect(q.maxError).toBeLessThanOrEqual(q.step / 2 + 1e-12);
      expect(q.levels).toBeLessThanOrEqual(2 ** bits - 1);
    }
  });

  it('the mean squared error falls by about 4x per extra bit (step^2 / 12)', () => {
    const e4 = quantizeValues(w, 4).meanSquaredError;
    const e5 = quantizeValues(w, 5).meanSquaredError;
    expect(e4 / e5).toBeGreaterThan(3);
    expect(e4 / e5).toBeLessThan(5.5);
    const q = quantizeValues(w, 6);
    expect(q.meanSquaredError).toBeCloseTo((q.step * q.step) / 12, 3);
  });

  it('quantises a copy and leaves the original alone; storage scales with the bits', () => {
    const net = new ToolboxMLP({ ...DEFAULT_TOOLBOX, layers: [4, 8, 2] });
    const before = Array.from(net.weights[0]!.data);
    const q = quantizeModel(net, 3);
    expect(Array.from(net.weights[0]!.data)).toEqual(before);
    expect(new Set(Array.from(q.weights[0]!.data)).size).toBeLessThanOrEqual(7);
    expect(weightBytes(net, 8)).toBe(weightBytes(net, 4) * 2);
  });
});

describe('pruning and distillation helpers', () => {
  it('global magnitude pruning removes the requested share and keeps the largest weights', () => {
    const net = new ToolboxMLP({ ...DEFAULT_TOOLBOX, layers: [6, 10, 2] });
    const p = pruneModel(net, 0.5);
    expect(p.sparsity).toBeGreaterThan(0.48);
    expect(p.sparsity).toBeLessThan(0.52);
    const kept = p.model.weights.flatMap((w) => Array.from(w.data).filter((v) => v !== 0));
    const dropped = net.weights.flatMap((w, l) =>
      Array.from(w.data).filter((_, i) => !p.masks[l]![i]),
    );
    expect(Math.min(...kept.map(Math.abs))).toBeGreaterThanOrEqual(
      Math.max(...dropped.map(Math.abs)) - 1e-12,
    );
  });

  it('clones are independent', () => {
    const net = new ToolboxMLP({ ...DEFAULT_TOOLBOX, layers: [3, 4, 2] });
    const c = cloneModel(net);
    c.weights[0]!.data[0] = 99;
    expect(net.weights[0]!.data[0]).not.toBe(99);
  });

  it('a higher temperature flattens the teacher distribution and keeps the order', () => {
    const net = new ToolboxMLP({ ...DEFAULT_TOOLBOX, layers: [3, 6, 4] });
    const x = new Matrix(1, 3, Float64Array.from([1, -2, 0.5]));
    const sharp = softTargets(net, x, 1);
    const soft = softTargets(net, x, 4);
    const top = (m: Matrix) => Array.from(m.data).indexOf(Math.max(...m.data));
    expect(top(soft)).toBe(top(sharp));
    expect(Math.max(...soft.data)).toBeLessThanOrEqual(Math.max(...sharp.data) + 1e-12);
    expect(Array.from(soft.data).reduce((a, b) => a + b, 0)).toBeCloseTo(1, 12);
  });
});
