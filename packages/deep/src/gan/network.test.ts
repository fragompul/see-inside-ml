import { describe, expect, it } from 'vitest';
import { Matrix, createRng } from '@sim/core';
import { GENERATOR_LOSSES, GanNet, type GeneratorLoss } from './network';

const ARCH = { noiseDim: 3, hidden: 8 };

const points = (rows: number, seed: number) => {
  const rng = createRng(seed);
  return new Matrix(
    rows,
    2,
    Float64Array.from({ length: rows * 2 }, () => rng.normal()),
  );
};
const noise = (rows: number, seed: number) => {
  const rng = createRng(seed);
  return new Matrix(
    rows,
    ARCH.noiseDim,
    Float64Array.from({ length: rows * ARCH.noiseDim }, () => rng.normal()),
  );
};

const make = (seed = 3) => {
  const net = new GanNet(ARCH, createRng(seed));
  // non-zero biases so no ReLU sits exactly on its kink
  const rng = createRng(seed + 40);
  for (const p of [...net.generator.parameters, ...net.discriminator.parameters])
    if (!p.decay) for (let i = 0; i < p.data.length; i++) p.data[i] = rng.uniform(-0.3, 0.3);
  return net;
};

/** Largest relative error between analytic gradients and central finite differences. */
function worstError(
  params: { data: Float64Array; grad: Float64Array }[],
  loss: () => number,
): number {
  const h = 1e-6;
  let worst = 0;
  for (const p of params) {
    for (let i = 0; i < p.data.length; i++) {
      const original = p.data[i] as number;
      p.data[i] = original + h;
      const up = loss();
      p.data[i] = original - h;
      const down = loss();
      p.data[i] = original;
      const numeric = (up - down) / (2 * h);
      const analytic = p.grad[i] as number;
      worst = Math.max(
        worst,
        Math.abs(numeric - analytic) / Math.max(1e-6, Math.abs(numeric) + Math.abs(analytic)),
      );
    }
  }
  return worst;
}

describe('discriminator gradients', () => {
  it('match finite differences', () => {
    const net = make();
    const real = points(6, 1);
    const z = noise(6, 2);
    net.discriminatorGradients(real, z);
    const loss = () => {
      const fake = net.generator.forward(z, false);
      let total = 0;
      const pr = net.discriminator.forward(real, false);
      const pf = net.discriminator.forward(fake, false);
      for (const p of pr.data) total -= Math.log(p);
      for (const p of pf.data) total -= Math.log(1 - p);
      return total / (real.rows + fake.rows);
    };
    expect(worstError(net.discriminator.parameters, loss)).toBeLessThan(1e-4);
  });

  it('reports sensible numbers and leaves the generator untouched', () => {
    const net = make();
    const before = net.generator.parameters.map((p) => Float64Array.from(p.data));
    const step = net.discriminatorGradients(points(8, 5), noise(8, 6));
    expect(step.loss).toBeGreaterThan(0);
    expect(step.dReal).toBeGreaterThan(0);
    expect(step.dReal).toBeLessThan(1);
    expect(step.dFake).toBeGreaterThan(0);
    net.generator.parameters.forEach((p, i) => expect(p.data).toEqual(before[i]));
  });
});

describe.each(GENERATOR_LOSSES)('generator gradients (%s)', (lossName: GeneratorLoss) => {
  it('match finite differences through the discriminator', () => {
    const net = make(7);
    const z = noise(5, 8);
    net.generatorGradients(z, lossName);
    const loss = () => {
      const p = net.discriminator.forward(net.generator.forward(z, false), false);
      let total = 0;
      for (const v of p.data)
        total += lossName === 'nonsaturating' ? -Math.log(v) : Math.log(1 - v);
      return total / z.rows;
    };
    expect(worstError(net.generator.parameters, loss)).toBeLessThan(1e-4);
  });
});

describe('the two generator losses', () => {
  it('give the generator a tiny gradient with minimax when the discriminator is sure points are fake', () => {
    const net = make(9);
    // make D certain that everything is fake: large negative output bias
    const last = net.discriminator.layers[2]!;
    last.biases.data.fill(-9);
    const z = noise(16, 10);
    const ns = net.generatorGradients(z, 'nonsaturating');
    const mm = net.generatorGradients(z, 'minimax');
    expect(ns.dFake).toBeLessThan(0.01);
    expect(mm.signal).toBeLessThan(0.01 * ns.signal);
  });

  it('agree in direction near D = 1/2 (their gradients differ by a factor ~ 1)', () => {
    const net = make(11);
    const last = net.discriminator.layers[2]!;
    last.weights.data.fill(0);
    last.biases.data.fill(0); // D(x) = 1/2 everywhere
    const z = noise(8, 12);
    const ns = net.generatorGradients(z, 'nonsaturating');
    const mm = net.generatorGradients(z, 'minimax');
    expect(ns.dFake).toBeCloseTo(0.5, 12);
    // dZ is (p − 1) = −0.5 for one and −p = −0.5 for the other
    expect(mm.signal).toBeCloseTo(ns.signal, 12);
  });
});

describe('inference helpers', () => {
  it('discriminate and generate do not disturb pending gradients', () => {
    const a = make(13);
    const b = make(13);
    const real = points(6, 14);
    const z = noise(6, 15);
    a.discriminatorGradients(real, z);
    b.discriminatorGradients(real, z);
    a.discriminate(Float64Array.from([0, 0, 1, 1]));
    a.generate(noise(4, 16).data);
    a.discriminator.parameters.forEach((p, i) =>
      expect(p.grad).toEqual((b.discriminator.parameters[i] as typeof p).grad),
    );
  });

  it('outputs have the right shapes and valid probabilities', () => {
    const net = make();
    expect(net.generate(noise(5, 1).data)).toHaveLength(10);
    const probs = net.discriminate(Float64Array.from({ length: 20 }, (_, i) => i / 3));
    expect(probs).toHaveLength(10);
    expect(probs.every((p) => p >= 0 && p <= 1)).toBe(true);
  });

  it('rejects invalid architectures and is deterministic for a seed', () => {
    expect(() => new GanNet({ noiseDim: 1, hidden: 8 }, createRng(1))).toThrow(RangeError);
    expect(() => new GanNet({ noiseDim: 2, hidden: 500 }, createRng(1))).toThrow(RangeError);
    const z = noise(3, 1).data;
    expect(make(2).generate(z)).toEqual(make(2).generate(z));
  });
});
