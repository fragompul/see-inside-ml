import { describe, expect, it } from 'vitest';
import { createRng } from '@sim/core';
import { CELL_NAMES, RecurrentNet, type CellName } from './recurrent';

const make = (cell: CellName, hidden = 4, seed = 3) => {
  const net = new RecurrentNet({ cell, hidden, vocab: 4, classes: 3 }, createRng(seed));
  // non-zero biases everywhere so no gate sits at a symmetric point
  const rng = createRng(seed + 100);
  for (let i = 0; i < net.b.data.length; i++)
    net.b.data[i] = (net.b.data[i] as number) + rng.uniform(-0.4, 0.4);
  for (let i = 0; i < net.bo.data.length; i++) net.bo.data[i] = rng.uniform(-0.3, 0.3);
  return net;
};

const lossOf = (net: RecurrentNet, tokens: number[], target: number) =>
  -Math.log(net.run(tokens).probs[target] as number);

describe.each(CELL_NAMES)('%s: backpropagation through time', (cell) => {
  it('matches finite differences for every parameter', () => {
    const net = make(cell);
    const sequences = [
      { tokens: [0, 2, 1, 3, 3, 1], target: 2 },
      { tokens: [3, 3], target: 0 },
      { tokens: [1], target: 1 },
    ];
    for (const { tokens, target } of sequences) {
      net.zeroGrad();
      net.backward(net.run(tokens), target);
      const eps = 1e-6;
      let worst = 0;
      for (const p of net.parameters) {
        for (let i = 0; i < p.data.length; i++) {
          const original = p.data[i] as number;
          p.data[i] = original + eps;
          const up = lossOf(net, tokens, target);
          p.data[i] = original - eps;
          const down = lossOf(net, tokens, target);
          p.data[i] = original;
          const numeric = (up - down) / (2 * eps);
          const analytic = p.grad[i] as number;
          const error =
            Math.abs(numeric - analytic) / Math.max(1e-6, Math.abs(numeric) + Math.abs(analytic));
          worst = Math.max(worst, error);
        }
      }
      expect(worst).toBeLessThan(1e-4);
    }
  });

  it('accumulates and scales gradients linearly', () => {
    const net = make(cell);
    const tokens = [1, 0, 3, 2];
    net.zeroGrad();
    net.backward(net.run(tokens), 1, 1);
    const full = net.parameters.map((p) => Float64Array.from(p.grad));
    net.zeroGrad();
    net.backward(net.run(tokens), 1, 0.25);
    net.backward(net.run(tokens), 1, 0.75);
    net.parameters.forEach((p, k) =>
      p.grad.forEach((g, i) => expect(g).toBeCloseTo((full[k] as Float64Array)[i] as number, 12)),
    );
  });

  it('reports gradient norms that do not depend on the scale', () => {
    const net = make(cell);
    const tokens = [1, 0, 3, 2, 2];
    net.zeroGrad();
    const a = net.backward(net.run(tokens), 0, 1).gradientNorms;
    const b = net.backward(net.run(tokens), 0, 0.1).gradientNorms;
    expect(a).toHaveLength(5);
    a.forEach((v, i) => expect(v).toBeCloseTo(b[i] as number, 12));
    expect(a.every((v) => v > 0 && Number.isFinite(v))).toBe(true);
  });

  it('produces probabilities and exposes states for every step', () => {
    const net = make(cell, 5);
    const trace = net.run([0, 1, 2, 3, 0, 1]);
    expect(trace.probs.reduce((s, v) => s + v, 0)).toBeCloseTo(1, 12);
    expect(trace.h).toHaveLength(7 * 5);
    expect(trace.h.slice(0, 5).every((v) => v === 0)).toBe(true);
    expect(trace.h.every((v) => Math.abs(v) <= 1)).toBe(true);
    expect(Object.keys(trace.gates)).toHaveLength({ rnn: 0, lstm: 4, gru: 4 }[cell]); // gru: r, z, n and h·Whn
  });

  it('is deterministic for a seed and reads the sequence in order', () => {
    const a = make(cell, 4, 9).run([0, 1, 2]).probs;
    const b = make(cell, 4, 9).run([0, 1, 2]).probs;
    expect(a).toEqual(b);
    const swapped = make(cell, 4, 9).run([2, 1, 0]).probs;
    expect(swapped).not.toEqual(a);
  });

  it('rejects tokens outside the vocabulary', () => {
    expect(() => make(cell).run([0, 4])).toThrow(RangeError);
    expect(() => make(cell).run([-1])).toThrow(RangeError);
  });
});

describe('cell specifics', () => {
  it('LSTM starts with open forget gates and a bounded cell', () => {
    const net = make('lstm', 6);
    const H = 6;
    // bias init was 1 (plus the small jitter added by `make`)
    for (let k = 0; k < H; k++) expect(net.b.data[H + k]).toBeGreaterThan(0.5);
    const trace = net.run([0, 1, 2, 3, 0, 1, 2, 3]);
    const forget = trace.gates.forget as Float64Array;
    expect(forget.reduce((s, v) => s + v, 0) / forget.length).toBeGreaterThan(0.6);
    expect(trace.c).not.toBeNull();
  });

  it('GRU update gate 1 would copy the old state: h = (1 − z)·n + z·h_prev', () => {
    const net = make('gru', 4);
    const t = net.run([2, 0, 1]);
    const H = 4;
    for (let step = 0; step < 3; step++) {
      for (let k = 0; k < H; k++) {
        const z = (t.gates.update as Float64Array)[step * H + k] as number;
        const n = (t.gates.candidate as Float64Array)[step * H + k] as number;
        const expected = (1 - z) * n + z * (t.h[step * H + k] as number);
        expect(t.h[(step + 1) * H + k]).toBeCloseTo(expected, 12);
      }
    }
  });

  it('rejects invalid configurations', () => {
    const rng = createRng(1);
    expect(
      () => new RecurrentNet({ cell: 'lstm', hidden: 1, vocab: 4, classes: 2 }, rng),
    ).toThrow();
    expect(
      () => new RecurrentNet({ cell: 'lstm', hidden: 99, vocab: 4, classes: 2 }, rng),
    ).toThrow();
    expect(
      () => new RecurrentNet({ cell: 'x' as CellName, hidden: 4, vocab: 4, classes: 2 }, rng),
    ).toThrow();
    expect(() => new RecurrentNet({ cell: 'rnn', hidden: 4, vocab: 1, classes: 2 }, rng)).toThrow();
  });
});
