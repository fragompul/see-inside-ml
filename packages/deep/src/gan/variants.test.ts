import { Matrix, createRng } from '@sim/core';
import { describe, expect, it } from 'vitest';
import { Critic, DEFAULT_GAN_VARIANT, nearestMode, trainGanVariant } from './variants';

const points = (n: number, seed: number) => {
  const rng = createRng(seed);
  return new Matrix(
    n,
    2,
    Float64Array.from({ length: n * 2 }, () => rng.normal()),
  );
};

describe('critic gradients', () => {
  it('the input gradient equals the numerical derivative of the score', () => {
    const critic = new Critic(2, 6, 'linear', createRng(3));
    const x = points(5, 1);
    critic.forward(x, true);
    const { grads } = critic.inputGradient(2);
    const h = 1e-6;
    for (let r = 0; r < 5; r++)
      for (let i = 0; i < 2; i++) {
        const up = new Matrix(5, 2, Float64Array.from(x.data));
        const down = new Matrix(5, 2, Float64Array.from(x.data));
        up.data[r * 2 + i] = up.data[r * 2 + i]! + h;
        down.data[r * 2 + i] = down.data[r * 2 + i]! - h;
        const numeric =
          (critic.forward(up, false).data[r]! - critic.forward(down, false).data[r]!) / (2 * h);
        expect(grads[r * 2 + i]).toBeCloseTo(numeric, 6);
      }
  });

  it('the gradient of the penalty matches finite differences of the penalty', () => {
    const critic = new Critic(2, 5, 'linear', createRng(5));
    const x = points(6, 2);
    const lambda = 3;
    const penalty = () => {
      critic.forward(x, true);
      const { grads } = critic.inputGradient(2);
      let p = 0;
      for (let r = 0; r < 6; r++) p += (Math.hypot(grads[2 * r]!, grads[2 * r + 1]!) - 1) ** 2 / 6;
      return lambda * p;
    };
    critic.forward(x, true);
    for (const p of critic.parameters) p.grad.fill(0);
    const value = critic.addGradientPenalty(lambda, 2);
    expect(value).toBeCloseTo(penalty(), 10);
    const h = 1e-6;
    // the biases do not enter the gradient of a piecewise-linear network, so their penalty gradient is zero
    for (const p of critic.parameters) {
      const grad = Float64Array.from(p.grad);
      for (const i of [0, Math.floor(p.data.length / 2), p.data.length - 1]) {
        const saved = p.data[i]!;
        p.data[i] = saved + h;
        const up = penalty();
        p.data[i] = saved - h;
        const down = penalty();
        p.data[i] = saved;
        expect(grad[i]).toBeCloseTo((up - down) / (2 * h), 5);
      }
    }
  });
});

describe('GAN variants', () => {
  const quick = { ...DEFAULT_GAN_VARIANT, steps: 300 };

  it('every kind trains without producing non-finite numbers and records its history', () => {
    for (const kind of ['gan', 'wgan-gp', 'cgan'] as const) {
      const run = trainGanVariant({ ...quick, kind, criticSteps: kind === 'wgan-gp' ? 3 : 1 }, 100);
      expect(run.history.map((h) => h.step)).toEqual([100, 200, 300]);
      expect(Array.from(run.samples).every(Number.isFinite)).toBe(true);
      expect(Number.isFinite(run.distance)).toBe(true);
    }
  });

  it('is deterministic for a seed', () => {
    const a = trainGanVariant({ ...quick, steps: 60 }, 60);
    const b = trainGanVariant({ ...quick, steps: 60 }, 60);
    expect(Array.from(a.samples)).toEqual(Array.from(b.samples));
  });

  it('the nearest mode of a point is the cluster whose centre it is closest to', () => {
    expect(nearestMode(2, 0)).toBe(0);
    expect(nearestMode(0, 2)).toBe(2);
    expect(nearestMode(-1.9, 0.1)).toBe(4);
  });

  it('the conditional GAN learns to put points in the cluster that was asked for', () => {
    const run = trainGanVariant({ ...DEFAULT_GAN_VARIANT, kind: 'cgan', steps: 1200 });
    expect(run.classAccuracy).toBeGreaterThan(0.5);
  });
});
