import { Matrix, createRng } from '@sim/core';
import { describe, expect, it } from 'vitest';
import { ConditionalVae, VqVae, oneHot } from './variants';
import { trainConditionalVae, trainVqVae } from './variantsTrain';

const batch = (rows: number, cols: number, seed: number) => {
  const rng = createRng(seed);
  return new Matrix(
    rows,
    cols,
    Float64Array.from({ length: rows * cols }, () => rng.next()),
  );
};

describe('conditional VAE', () => {
  it('backpropagation matches finite differences (with the sampling noise held fixed)', () => {
    const model = new ConditionalVae(12, 3, 2, 5, createRng(2));
    const x = batch(4, 12, 7);
    const labels = [0, 2, 1, 2];
    const eps = batch(4, 2, 8);
    const beta = 0.7;
    const run = () => model.loss(x, model.forward(x, labels, eps, false), beta).total;
    const out = model.forward(x, labels, eps);
    model.backward(x, out, beta);
    const h = 1e-6;
    for (const p of model.parameters) {
      const grad = Float64Array.from(p.grad);
      for (const i of [0, Math.floor(p.data.length / 2), p.data.length - 1]) {
        const saved = p.data[i]!;
        p.data[i] = saved + h;
        const up = run();
        p.data[i] = saved - h;
        const down = run();
        p.data[i] = saved;
        expect(grad[i]).toBeCloseTo((up - down) / (2 * h), 5);
      }
    }
  });

  it('the class changes what is decoded from the same code', () => {
    const model = new ConditionalVae(12, 3, 2, 8, createRng(4));
    const a = model.decode([0.3, -0.2], 0);
    const b = model.decode([0.3, -0.2], 2);
    expect(Array.from(a)).not.toEqual(Array.from(b));
    expect(oneHot([2, 0], 3).data).toEqual(Float64Array.of(0, 0, 1, 1, 0, 0));
  });
});

describe('vector-quantised autoencoder', () => {
  it('decoder and codebook gradients match finite differences of the terms that produce them', () => {
    const model = new VqVae(12, 4, 3, 6, createRng(3));
    const x = batch(5, 12, 9);
    const commitment = 0.25;
    const out = model.forward(x);
    model.backward(x, out, commitment);
    const h = 1e-6;
    const total = () => model.loss(x, model.forward(x, false), commitment);
    // decoder weights: the reconstruction depends on them exactly (the assignment does not change for a tiny step)
    for (const p of [...model.dec1.parameters, ...model.dec2.parameters]) {
      const grad = Float64Array.from(p.grad);
      for (const i of [0, p.data.length - 1]) {
        const saved = p.data[i]!;
        p.data[i] = saved + h;
        const up = total().recon;
        p.data[i] = saved - h;
        const down = total().recon;
        p.data[i] = saved;
        expect(grad[i]).toBeCloseTo((up - down) / (2 * h), 5);
      }
    }
    // codebook: only the codebook term moves it
    const grad = Float64Array.from(model.codebookParam.grad);
    for (let i = 0; i < grad.length; i++) {
      const saved = model.codebook.data[i]!;
      model.codebook.data[i] = saved + h;
      const up = total().codebook;
      model.codebook.data[i] = saved - h;
      const down = total().codebook;
      model.codebook.data[i] = saved;
      expect(grad[i]).toBeCloseTo((up - down) / (2 * h), 5);
    }
  });

  it('snaps every code to its nearest entry', () => {
    const model = new VqVae(12, 5, 2, 4, createRng(1));
    const z = batch(20, 2, 3);
    const index = model.nearest(z);
    for (let r = 0; r < 20; r++) {
      const dist = (k: number) =>
        (z.data[r * 2]! - model.codebook.data[k * 2]!) ** 2 +
        (z.data[r * 2 + 1]! - model.codebook.data[k * 2 + 1]!) ** 2;
      for (let k = 0; k < 5; k++) expect(dist(index[r]!)).toBeLessThanOrEqual(dist(k) + 1e-12);
    }
  });
});

describe('training', () => {
  it('the conditional VAE reconstructs better with training and the class steers the output', () => {
    const run = trainConditionalVae({
      epochs: 8,
      perClass: 40,
      batchSize: 32,
      learningRate: 0.003,
      seed: 2,
      latent: 2,
      hidden: 48,
      beta: 1,
      conditioned: true,
    });
    expect(run.testRecon[run.testRecon.length - 1]!).toBeLessThan(run.testRecon[0]!);
    const sevens = run.model.decode([0, 0], 7);
    const ones = run.model.decode([0, 0], 1);
    const sum = (v: Float64Array) => v.reduce((a, b) => a + b, 0);
    expect(sum(sevens)).toBeGreaterThan(sum(ones)); // a 7 has more ink than a 1
  });

  it('the VQ-VAE reconstructs better with training and uses its codebook', () => {
    const run = trainVqVae({
      epochs: 10,
      perClass: 40,
      batchSize: 32,
      learningRate: 0.003,
      seed: 2,
      entries: 10,
      dim: 4,
      hidden: 48,
      commitment: 0.25,
      revive: true,
    });
    expect(run.testRecon[run.testRecon.length - 1]!).toBeLessThan(run.testRecon[0]!);
    expect(run.usage.reduce((a, b) => a + b, 0)).toBe(run.train.count);
    expect(run.usage.filter((u) => u > 0).length).toBeGreaterThanOrEqual(5);
  });
});
