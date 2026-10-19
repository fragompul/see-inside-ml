import { describe, expect, it } from 'vitest';
import { DEFAULT_GAN_LAB, EVAL_POINTS, GanSession, RECORD_EVERY, validateGanLab } from './session';

const make = (patch = {}) =>
  new GanSession({ ...DEFAULT_GAN_LAB, hidden: 12, batchSize: 32, ...patch });

describe('GanSession', () => {
  it('trains deterministically for a seed', () => {
    const a = make();
    const b = make();
    a.train(30);
    b.train(30);
    expect(a.history).toEqual(b.history);
    expect(a.fixedFake()).toEqual(b.fixedFake());
  });

  it('records statistics every RECORD_EVERY iterations, and train() is chunk-independent', () => {
    const once = make();
    const chunks = make();
    expect(once.train(25)).toHaveLength(2);
    chunks.train(7);
    chunks.train(18);
    expect(once.iteration).toBe(25);
    expect(once.history.map((h) => h.iteration)).toEqual([RECORD_EVERY, 2 * RECORD_EVERY]);
    expect(chunks.history).toEqual(once.history);
  });

  it('records meaningful numbers', () => {
    const s = make();
    s.train(RECORD_EVERY);
    const h = s.history[0]!;
    for (const v of [h.lossD, h.lossG, h.signal, h.distance]) expect(Number.isFinite(v)).toBe(true);
    expect(h.dReal).toBeGreaterThan(0);
    expect(h.dFake).toBeLessThan(1);
    expect(h.near).toBeGreaterThanOrEqual(0);
    expect(h.near).toBeLessThanOrEqual(1);
    expect(h.modes).not.toBeNull(); // eight Gaussians
    expect(make({ dataset: 'ring' }).snapshot().modes).toBeNull();
  });

  it('applies learning settings live and rebuilds on structure or data changes', () => {
    const s = make();
    s.train(RECORD_EVERY);
    expect(s.update({ lrG: 0.01, lrD: 0.005, dSteps: 3, lossG: 'minimax', batchSize: 64 })).toBe(
      'live',
    );
    expect(s.history).toHaveLength(1);
    expect(s.update({ hidden: 16 })).toBe('rebuilt');
    expect(s.history).toHaveLength(0);
    expect(s.update({ noiseDim: 4 })).toBe('rebuilt');
    expect(s.update({ dataset: 'moons' })).toBe('rebuilt');
    expect(s.iteration).toBe(0);
  });

  it('rejects invalid settings and keeps the previous ones', () => {
    const s = make();
    expect(() => s.update({ lrG: 0 })).toThrow();
    expect(() => s.update({ dSteps: 0 })).toThrow();
    expect(() => s.update({ batchSize: 5 })).toThrow();
    expect(() => s.update({ noiseDim: 1 })).toThrow();
    expect(() => s.update({ dataset: 'x' as 'ring' })).toThrow();
    expect(() => s.update({ lossG: 'x' as 'minimax' })).toThrow();
    expect(() => validateGanLab({ ...DEFAULT_GAN_LAB, dSteps: 2.5 })).toThrow();
    expect(s.config.lrG).toBe(DEFAULT_GAN_LAB.lrG);
  });

  it('reset restarts from the same weights, and the fixed noise keeps the cloud comparable', () => {
    const s = make();
    const before = s.fixedFake();
    s.train(40);
    expect(s.fixedFake()).not.toEqual(before);
    s.reset();
    expect(s.iteration).toBe(0);
    expect(s.history).toHaveLength(0);
    expect(s.fixedFake()).toEqual(before);
  });

  it('exposes the plots the UI needs', () => {
    const s = make({ noiseDim: 3 });
    expect(s.reference).toHaveLength(EVAL_POINTS * 2);
    expect(s.fixedFake()).toHaveLength(EVAL_POINTS * 2);
    expect(s.generate(17)).toHaveLength(34);
    expect(s.real(9)).toHaveLength(18);
    const grid = s.discriminatorGrid(3, 5);
    expect(grid).toHaveLength(25);
    expect(grid.every((p) => p > 0 && p < 1)).toBe(true);
    expect(s.warpedGrid(4)).toHaveLength(4 * 4 * 2);
  });

  it('inspecting the model does not disturb training', () => {
    const a = make();
    const b = make();
    a.train(5);
    b.train(5);
    a.fixedFake();
    a.discriminatorGrid(3, 6);
    a.warpedGrid(5);
    a.generate(10);
    a.snapshot();
    a.train(5);
    b.train(5);
    // snapshot() resets the running averages, so compare the networks, not the history
    expect(a.net.generator.parameters.map((p) => p.data)).toEqual(
      b.net.generator.parameters.map((p) => p.data),
    );
  });
});
