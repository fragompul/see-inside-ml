import { describe, expect, it } from 'vitest';
import {
  DEFAULT_DIFFUSION_LAB,
  DIFFUSION_EVAL_POINTS,
  DIFFUSION_EVALUATE_EVERY,
  DIFFUSION_RECORD_EVERY,
  DIFFUSION_TRACKED,
  DiffusionSession,
  validateDiffusionLab,
} from './session';

const make = (patch = {}) =>
  new DiffusionSession({
    ...DEFAULT_DIFFUSION_LAB,
    hidden: 16,
    batchSize: 32,
    steps: 20,
    ...patch,
  });

describe('DiffusionSession', () => {
  it('trains deterministically for a seed and records the loss every few iterations', () => {
    const a = make();
    const b = make();
    expect(a.train(25)).toHaveLength(2);
    b.train(10);
    b.train(15);
    expect(a.history).toEqual(b.history);
    expect(a.history.map((h) => h.iteration)).toEqual([
      DIFFUSION_RECORD_EVERY,
      2 * DIFFUSION_RECORD_EVERY,
    ]);
    expect(a.history.every((h) => Number.isFinite(h.loss) && h.loss > 0)).toBe(true);
  });

  it('measures the sample distance only every EVALUATE_EVERY iterations', () => {
    const s = make();
    s.train(DIFFUSION_EVALUATE_EVERY);
    const measured = s.history.filter((h) => h.distance !== null);
    expect(measured).toHaveLength(1);
    expect(measured[0]?.iteration).toBe(DIFFUSION_EVALUATE_EVERY);
    expect(measured[0]?.distance as number).toBeGreaterThan(0);
  });

  it('the loss drops as the denoiser learns', () => {
    const s = make({ hidden: 32, batchSize: 64 });
    s.train(300);
    const first = s.history[0]?.loss as number;
    const last = s.history[s.history.length - 1]?.loss as number;
    expect(last).toBeLessThan(first * 0.8);
  });

  it('applies learning settings live and rebuilds on data, schedule or architecture changes', () => {
    const s = make();
    s.train(10);
    expect(
      s.update({ lr: 0.01, batchSize: 64, sampler: 'ddpm', sampleSteps: 7, optimizer: 'sgd' }),
    ).toBe('live');
    expect(s.history).toHaveLength(1);
    expect(s.update({ steps: 30 })).toBe('rebuilt');
    expect(s.schedule.steps).toBe(30);
    expect(s.update({ schedule: 'cosine' })).toBe('rebuilt');
    expect(s.update({ hidden: 32 })).toBe('rebuilt');
    expect(s.update({ dataset: 'ring' })).toBe('rebuilt');
    expect(s.iteration).toBe(0);
  });

  it('rejects invalid settings and keeps the previous ones', () => {
    const s = make();
    expect(() => s.update({ steps: 3 })).toThrow();
    expect(() => s.update({ lr: 0 })).toThrow();
    expect(() => s.update({ batchSize: 3 })).toThrow();
    expect(() => s.update({ sampleSteps: 0 })).toThrow();
    expect(() => s.update({ sampler: 'x' as 'ddim' })).toThrow();
    expect(() => s.update({ schedule: 'x' as 'linear' })).toThrow();
    expect(() => validateDiffusionLab({ ...DEFAULT_DIFFUSION_LAB, hidden: 5 })).toThrow();
    expect(s.config.lr).toBe(DEFAULT_DIFFUSION_LAB.lr);
  });

  it('exposes forward noising, sampling, trajectories and the score field', () => {
    const s = make();
    expect(s.forward(0)).toEqual(s.reference);
    expect(s.forward(20)).not.toEqual(s.reference);
    const fixed = s.fixedSample({ method: 'ddim', steps: 4 });
    expect(fixed.samples).toHaveLength(DIFFUSION_EVAL_POINTS * 2);
    expect(fixed.trajectory).toHaveLength(5 * DIFFUSION_TRACKED * 2);
    expect(fixed.times).toEqual([20, 15, 10, 5, 0]);
    expect(s.sample(7).samples).toHaveLength(14);
    expect(s.sample(5, { seed: 1 }).samples).toEqual(s.sample(5, { seed: 1 }).samples);
    const { points, score } = s.scoreField(10, 6, 3);
    expect(points).toHaveLength(72);
    expect(score).toHaveLength(72);
    expect(score.every(Number.isFinite)).toBe(true);
  });

  it('reset restarts from the same weights; inspecting does not disturb training', () => {
    const a = make();
    const b = make();
    const before = a.fixedSample().samples;
    a.train(10);
    a.reset();
    expect(a.iteration).toBe(0);
    expect(a.fixedSample().samples).toEqual(before);
    a.train(5);
    b.train(5);
    a.fixedSample();
    a.scoreField(5, 4, 3);
    a.distance();
    a.train(5);
    b.train(5);
    expect(a.net.parameters.map((p) => p.data)).toEqual(b.net.parameters.map((p) => p.data));
  });
});
