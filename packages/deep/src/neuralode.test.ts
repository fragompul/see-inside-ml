import { createRng } from '@sim/core';
import { describe, expect, it } from 'vitest';
import { NeuralOde, trainNeuralOde, type Solver } from './neuralode';

const data = (n: number, seed: number) => {
  const rng = createRng(seed);
  const xs = Float64Array.from({ length: 2 * n }, () => rng.normal());
  const ys = Float64Array.from({ length: n }, (_, i) => ((xs[2 * i] as number) > 0 ? 1 : 0));
  return { xs, ys };
};

const copyInto = (to: NeuralOde, from: NeuralOde) =>
  to.parameters.forEach((p, i) => p.data.set((from.parameters[i] as typeof p).data));

describe('neural ODE', () => {
  it.each<Solver>(['euler', 'rk4'])(
    '%s: backpropagation through the solver matches finite differences',
    (solver) => {
      const model = new NeuralOde({ dim: 3, hidden: 5, solver, steps: 4 }, createRng(2));
      const { xs, ys } = data(6, 3);
      model.loss(xs, ys, true);
      const h = 1e-6;
      for (const p of model.parameters) {
        const grad = Float64Array.from(p.grad);
        for (const i of [0, Math.floor(p.data.length / 2), p.data.length - 1]) {
          const saved = p.data[i]!;
          p.data[i] = saved + h;
          const up = model.loss(xs, ys, false);
          p.data[i] = saved - h;
          const down = model.loss(xs, ys, false);
          p.data[i] = saved;
          expect(grad[i]).toBeCloseTo((up - down) / (2 * h), 5);
        }
      }
    },
  );

  it('Euler converges at first order and Runge-Kutta at fourth order as the steps get finer', () => {
    const reference = new NeuralOde({ dim: 2, hidden: 6, solver: 'rk4', steps: 512 }, createRng(5));
    // a stronger field makes the error easy to see
    reference.parameters.forEach((p) =>
      p.data.forEach((_, i) => (p.data[i] = (p.data[i] as number) * 2.5)),
    );
    const truth = reference.trajectory(0.7, -0.4).pop() as Float64Array;
    const error = (solver: Solver, steps: number) => {
      const m = new NeuralOde({ dim: 2, hidden: 6, solver, steps }, createRng(5));
      copyInto(m, reference);
      const z = m.trajectory(0.7, -0.4).pop() as Float64Array;
      return Math.hypot(z[0]! - truth[0]!, z[1]! - truth[1]!);
    };
    const euler = [error('euler', 8) / error('euler', 16), error('euler', 16) / error('euler', 32)];
    const rk = [error('rk4', 2) / error('rk4', 4), error('rk4', 4) / error('rk4', 8)];
    for (const r of euler) {
      expect(r).toBeGreaterThan(1.7);
      expect(r).toBeLessThan(2.3);
    }
    for (const r of rk) expect(r).toBeGreaterThan(10);
  });

  it('the path has one state more than the steps, and the probability is between 0 and 1', () => {
    const model = new NeuralOde({ dim: 4, hidden: 4, solver: 'rk4', steps: 6 }, createRng(1));
    expect(model.trajectory(0.3, 0.2)).toHaveLength(7);
    const p = model.probability(0.3, 0.2);
    expect(p).toBeGreaterThan(0);
    expect(p).toBeLessThan(1);
  });

  it('learns a problem that a straight line can solve', () => {
    const { xs, ys } = data(80, 7);
    const run = trainNeuralOde(
      { dim: 2, hidden: 8, solver: 'rk4', steps: 4 },
      { xs, ys },
      { epochs: 100, learningRate: 0.05, seed: 1 },
    );
    expect(run.history[run.history.length - 1]!.accuracy).toBeGreaterThan(0.9);
  });
});
