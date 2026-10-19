import { describe, expect, it } from 'vitest';
import { createRng } from '@sim/core';
import {
  AutoencoderSession,
  DEFAULT_AE_LAB,
  corruptImage,
  validateAutoencoderLab,
} from './session';

const make = (patch = {}) =>
  new AutoencoderSession({ ...DEFAULT_AE_LAB, perClass: 10, batchSize: 16, ...patch });

describe('AutoencoderSession', () => {
  it.each(['ae', 'vae'] as const)('%s: learns something and is deterministic', (mode) => {
    const a = make({ mode });
    const b = make({ mode });
    const first = a.evaluate().trainRecon;
    a.step(6);
    b.step(6);
    expect(a.history).toEqual(b.history);
    expect(a.history[5]?.trainRecon as number).toBeLessThan(first);
    expect(a.history[5]?.testRecon).toBeGreaterThan(0);
    if (mode === 'ae') expect(a.history[5]?.trainKl).toBe(0);
    else expect(a.history[5]?.trainKl as number).toBeGreaterThan(0);
  });

  it('batch-wise training equals epoch-wise training', () => {
    const batches = make({ mode: 'vae', perClass: 10, batchSize: 20 }); // 100 images, batches of 20 → 5 per epoch
    const epochs = make({ mode: 'vae', perClass: 10, batchSize: 20 });
    expect(batches.trainBatches(10)).toHaveLength(2);
    epochs.step(2);
    expect(batches.history).toEqual(epochs.history);
  });

  it('totals combine reconstruction and beta times KL', () => {
    const s = make({ mode: 'vae', beta: 3 });
    s.step(2);
    const h = s.history[1]!;
    expect(h.trainTotal).toBeCloseTo(h.trainRecon + 3 * h.trainKl, 10);
    expect(h.testTotal).toBeCloseTo(h.testRecon + 3 * h.testKl, 10);
  });

  it('applies learning settings live and rebuilds on structure changes', () => {
    const s = make();
    s.step(1);
    expect(
      s.update({ lr: 0.01, beta: 2, corruption: 0.2, optimizer: 'rmsprop', batchSize: 8 }),
    ).toBe('live');
    expect(s.history).toHaveLength(1);
    expect(s.update({ latent: 3 })).toBe('rebuilt');
    expect(s.history).toHaveLength(0);
    expect(s.update({ mode: 'vae' })).toBe('rebuilt');
    expect(s.network.encLogVar).not.toBeNull();
    expect(s.update({ perClass: 20 })).toBe('rebuilt');
    expect(s.trainCount).toBe(200);
  });

  it('rejects invalid settings and keeps the previous ones', () => {
    const s = make();
    expect(() => s.update({ lr: 0 })).toThrow();
    expect(() => s.update({ latent: 1 })).toThrow();
    expect(() => s.update({ beta: -1 })).toThrow();
    expect(() => s.update({ corruption: 2 })).toThrow();
    expect(() => s.update({ perClass: 3 })).toThrow();
    expect(() => validateAutoencoderLab({ ...DEFAULT_AE_LAB, batchSize: 1.5 })).toThrow();
    expect(s.config.lr).toBe(DEFAULT_AE_LAB.lr);
  });

  it('exposes codes, reconstructions, samples and generation', () => {
    const s = make({ latent: 3 });
    const { mu, labels, count } = s.latents();
    expect(count).toBe(s.testCount);
    expect(mu).toHaveLength(count * 3);
    expect(labels).toHaveLength(count);
    const { image } = s.sample(0);
    expect(s.encode(image).mu).toHaveLength(3);
    expect(s.reconstruct(image)).toHaveLength(256);
    expect(s.decode([0, 0, 0])).toHaveLength(256);
    const generated = s.generate(5, createRng(1));
    expect(generated).toHaveLength(5);
    expect(generated[0]).not.toEqual(generated[1]);
    expect(s.testIndicesOf(7)).toHaveLength(30);
    expect(s.sample(s.testIndicesOf(7)[3] as number).label).toBe(7);
    const codes = Float64Array.from([0, 0, 0, 1, -1, 2]);
    const many = s.decodeMany(codes, 2);
    expect(many).toHaveLength(2 * 256);
    expect(many.slice(256)).toEqual(s.decode([1, -1, 2]));
    expect(() => s.decodeMany(codes, 3)).toThrow(RangeError);
    expect(() => s.sample(99999)).toThrow();
  });

  it('inspecting the model does not disturb training', () => {
    const a = make({ mode: 'vae' });
    const b = make({ mode: 'vae' });
    a.trainBatches(1);
    b.trainBatches(1);
    a.latents();
    a.reconstruct(a.sample(0).image);
    a.evaluate();
    a.trainBatches(1);
    b.trainBatches(1);
    expect(a.network.enc1.weights.data).toEqual(b.network.enc1.weights.data);
  });
});

describe('corruptImage', () => {
  it('keeps values in [0, 1], changes the image and does nothing at level 0', () => {
    const image = new Float64Array(256).fill(0.5);
    const noisy = corruptImage(image, 0.4, createRng(1));
    expect(noisy.every((v) => v >= 0 && v <= 1)).toBe(true);
    expect(noisy).not.toEqual(image);
    expect(corruptImage(image, 0, createRng(1))).toEqual(image);
  });
});
