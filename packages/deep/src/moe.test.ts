import { createRng } from '@sim/core';
import { describe, expect, it } from 'vitest';
import { DEFAULT_MOE_TRAINING, Moe, trainMoe } from './moe';

const data = (n: number, seed: number) => {
  const rng = createRng(seed);
  const xs = Float64Array.from({ length: 2 * n }, () => rng.next() * 2 - 1);
  const ts = Float64Array.from(
    { length: n },
    (_, i) => Math.sin(3 * (xs[2 * i] as number)) + (xs[2 * i + 1] as number),
  );
  return { xs, ts };
};

describe('mixture of experts', () => {
  it.each([1, 2, 4])(
    'with %i experts running, backpropagation matches finite differences',
    (top) => {
      const model = new Moe({ experts: 4, hidden: 3, top }, createRng(2));
      const { xs, ts } = data(12, 5);
      const balance = 0.3;
      model.loss(xs, ts, balance, true);
      const h = 1e-6;
      for (const p of model.parameters) {
        const grad = Float64Array.from(p.grad);
        for (const i of [0, Math.floor(p.data.length / 2), p.data.length - 1]) {
          const saved = p.data[i]!;
          p.data[i] = saved + h;
          const up = model.loss(xs, ts, balance, false).total;
          p.data[i] = saved - h;
          const down = model.loss(xs, ts, balance, false).total;
          p.data[i] = saved;
          expect(grad[i]).toBeCloseTo((up - down) / (2 * h), 5);
        }
      }
    },
  );

  it('gate probabilities sum to one and only the best experts are chosen', () => {
    const model = new Moe({ experts: 5, hidden: 2, top: 2 }, createRng(3));
    const p = model.gate(0.3, -0.4);
    expect(p.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 12);
    const chosen = model.route(0.3, -0.4);
    expect(chosen).toHaveLength(2);
    const sorted = Array.from(p).sort((a, b) => b - a);
    expect(p[chosen[0]!]).toBe(sorted[0]);
    expect(p[chosen[1]!]).toBe(sorted[1]);
  });

  it('the balance term is 1 when the load is even and E when one expert takes everything', () => {
    const model = new Moe({ experts: 4, hidden: 2, top: 1 }, createRng(4));
    model.wg.fill(0);
    model.bg.fill(0);
    const { xs, ts } = data(40, 6);
    // equal gate probabilities, but the routing then breaks ties by index: everything goes to expert 0
    const collapsed = model.loss(xs, ts, 0, false);
    expect(collapsed.share[0]).toBeCloseTo(1, 12);
    expect(collapsed.balance).toBeCloseTo(1, 12); // E * f_0 * P_0 = 4 * 1 * 0.25
    model.bg[0] = 10;
    expect(model.loss(xs, ts, 0, false).balance).toBeGreaterThan(3.9);
  });

  it('active parameters are fewer than all parameters when only some experts run', () => {
    const sparse = new Moe({ experts: 8, hidden: 4, top: 1 }, createRng(1));
    expect(sparse.activeParameterCount).toBeLessThan(sparse.parameterCount / 3);
    const dense = new Moe({ experts: 8, hidden: 4, top: 8 }, createRng(1));
    expect(dense.activeParameterCount).toBe(dense.parameterCount);
  });

  it('training lowers the error on a smooth task', () => {
    const test = data(200, 9);
    const run = trainMoe(
      { experts: 4, hidden: 4, top: 2 },
      { ...DEFAULT_MOE_TRAINING, steps: 400 },
      (rng) => {
        const x1 = rng.next() * 2 - 1;
        const x2 = rng.next() * 2 - 1;
        return { x: [x1, x2], t: Math.sin(3 * x1) + x2 };
      },
      test,
    );
    expect(run.history[run.history.length - 1]!.mse).toBeLessThan(run.history[0]!.mse);
  });
});
