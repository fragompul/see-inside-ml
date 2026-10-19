import { describe, expect, it } from 'vitest';
import { Matrix, createRng } from '@sim/core';
import { AutoencoderNet, type AutoencoderMode } from './network';

const PIXELS = 12;

const images = (rows: number, seed: number) => {
  const rng = createRng(seed);
  return new Matrix(
    rows,
    PIXELS,
    Float64Array.from({ length: rows * PIXELS }, () => rng.next()),
  );
};

const noise = (rows: number, cols: number, seed: number) => {
  const rng = createRng(seed);
  return new Matrix(
    rows,
    cols,
    Float64Array.from({ length: rows * cols }, () => rng.normal()),
  );
};

const make = (mode: AutoencoderMode, seed = 3) => {
  const net = new AutoencoderNet({ mode, latent: 3, hidden: 7 }, PIXELS, createRng(seed));
  const rng = createRng(seed + 50);
  for (const p of net.parameters)
    if (!p.decay)
      for (let i = 0; i < p.data.length; i++)
        p.data[i] = (p.data[i] as number) + rng.uniform(-0.3, 0.3);
  return net;
};

describe.each(['ae', 'vae'] as const)('%s: gradients', (mode) => {
  it('match finite differences for every parameter', () => {
    const net = make(mode);
    const x = images(5, 1);
    const eps = mode === 'vae' ? noise(5, 3, 2) : null;
    const beta = 0.7;
    const total = () => net.loss(x, net.forward(x, eps, false), beta).total;

    const out = net.forward(x, eps);
    net.backward(x, out, beta);
    const analytic = net.parameters.map((p) => Float64Array.from(p.grad));

    const h = 1e-6;
    let worst = 0;
    net.parameters.forEach((p, k) => {
      for (let i = 0; i < p.data.length; i++) {
        const original = p.data[i] as number;
        p.data[i] = original + h;
        const up = total();
        p.data[i] = original - h;
        const down = total();
        p.data[i] = original;
        const numeric = (up - down) / (2 * h);
        const a = (analytic[k] as Float64Array)[i] as number;
        worst = Math.max(
          worst,
          Math.abs(numeric - a) / Math.max(1e-6, Math.abs(numeric) + Math.abs(a)),
        );
      }
    });
    expect(worst).toBeLessThan(1e-4);
  });
});

describe('losses', () => {
  it('the plain autoencoder has no KL term and no log-variance head', () => {
    const net = make('ae');
    const x = images(3, 4);
    const out = net.forward(x);
    expect(out.logvar).toBeNull();
    const loss = net.loss(x, out, 5);
    expect(loss.kl).toBe(0);
    expect(loss.total).toBe(loss.recon);
    expect(net.encLogVar).toBeNull();
  });

  it('the KL term is zero for a code that already equals the prior and positive otherwise', () => {
    const net = make('vae');
    const x = images(2, 5);
    const prior = {
      mu: new Matrix(2, 3),
      logvar: new Matrix(2, 3),
      z: new Matrix(2, 3),
      xhat: net.forward(x).xhat,
    };
    expect(net.loss(x, prior, 1).kl).toBeCloseTo(0, 12);
    const shifted = { ...prior, mu: new Matrix(2, 3, new Float64Array(6).fill(1)) };
    // −½ Σ (1 + 0 − 1 − 1)... per dimension: ½ μ² = 0.5, three dimensions
    expect(net.loss(x, shifted, 1).kl).toBeCloseTo(1.5, 12);
  });

  it('beta scales only the KL part of the total', () => {
    const net = make('vae');
    const x = images(4, 6);
    const out = net.forward(x, noise(4, 3, 7));
    const a = net.loss(x, out, 0);
    const b = net.loss(x, out, 2);
    expect(a.total).toBeCloseTo(a.recon, 12);
    expect(b.total).toBeCloseTo(b.recon + 2 * b.kl, 12);
    expect(b.recon).toBeCloseTo(a.recon, 12);
  });

  it('reconstruction loss is the pixel-wise binary cross-entropy', () => {
    const net = make('ae');
    const x = new Matrix(1, PIXELS, new Float64Array(PIXELS).fill(1));
    const xhat = new Matrix(1, PIXELS, new Float64Array(PIXELS).fill(0.5));
    const loss = net.loss(x, { mu: new Matrix(1, 3), logvar: null, z: new Matrix(1, 3), xhat }, 1);
    expect(loss.recon).toBeCloseTo(PIXELS * Math.LN2, 12);
  });
});

describe('forward behaviour', () => {
  it('without noise the decoder sees the mean; with noise it sees a sample around it', () => {
    const net = make('vae');
    const x = images(3, 8);
    const plain = net.forward(x, null, false);
    expect(plain.z.data).toEqual(plain.mu.data);
    const sampled = net.forward(x, noise(3, 3, 9), false);
    expect(sampled.z.data).not.toEqual(sampled.mu.data);
    expect(sampled.mu.data).toEqual(plain.mu.data);
  });

  it('inference helpers do not disturb a pending backward pass', () => {
    const a = make('vae');
    const b = make('vae');
    const x = images(4, 10);
    const eps = noise(4, 3, 11);
    const outA = a.forward(x, eps);
    const outB = b.forward(x, eps);
    a.encode(x.data.slice(0, PIXELS));
    a.decode([0.2, -0.4, 1]);
    a.reconstruct(x.data.slice(0, PIXELS));
    a.backward(x, outA, 1);
    b.backward(x, outB, 1);
    a.parameters.forEach((p, k) => expect(p.grad).toEqual((b.parameters[k] as typeof p).grad));
  });

  it('decodes any code into pixels in [0, 1] and rejects a wrong size', () => {
    const net = make('ae');
    const image = net.decode([5, -5, 100]);
    expect(image).toHaveLength(PIXELS);
    expect(image.every((v) => v >= 0 && v <= 1)).toBe(true);
    expect(() => net.decode([0, 0])).toThrow(RangeError);
  });

  it('is deterministic for a seed and validates its configuration', () => {
    expect(make('ae', 4).reconstruct(new Float64Array(PIXELS).fill(0.3))).toEqual(
      make('ae', 4).reconstruct(new Float64Array(PIXELS).fill(0.3)),
    );
    const rng = createRng(1);
    expect(() => new AutoencoderNet({ mode: 'ae', latent: 1, hidden: 16 }, PIXELS, rng)).toThrow();
    expect(() => new AutoencoderNet({ mode: 'ae', latent: 2, hidden: 500 }, PIXELS, rng)).toThrow();
    expect(
      () => new AutoencoderNet({ mode: 'gan' as 'ae', latent: 2, hidden: 16 }, PIXELS, rng),
    ).toThrow();
  });
});
