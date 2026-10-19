import { describe, expect, it } from 'vitest';
import { createRng } from '@sim/core';
import { Denoiser } from './network';
import { sampleDiffusion, samplingTimes, type NoisePredictor } from './sampler';
import { SCHEDULE_KINDS, TIME_FEATURES, makeSchedule, noiseTo, timeFeatures } from './schedule';

describe.each(SCHEDULE_KINDS)('%s schedule', (kind) => {
  it('starts clean, ends almost pure noise, and ᾱ decreases monotonically', () => {
    for (const steps of [10, 50, 200]) {
      const s = makeSchedule(kind, steps);
      expect(s.alphaBar[0]).toBe(1);
      expect(s.beta[0]).toBe(0);
      expect(s.alphaBar[steps] as number).toBeLessThan(0.02);
      for (let t = 1; t <= steps; t++) {
        expect(s.alphaBar[t] as number).toBeLessThan(s.alphaBar[t - 1] as number);
        expect(s.beta[t] as number).toBeGreaterThan(0);
        expect(s.beta[t] as number).toBeLessThanOrEqual(0.999);
      }
    }
  });

  it('keeps ᾱ equal to the running product of 1 − β and the roots consistent', () => {
    const s = makeSchedule(kind, 40);
    let product = 1;
    for (let t = 1; t <= 40; t++) {
      product *= 1 - (s.beta[t] as number);
      expect(s.alphaBar[t]).toBeCloseTo(product, 12);
      expect(
        (s.sqrtAlphaBar[t] as number) ** 2 + (s.sqrtOneMinusAlphaBar[t] as number) ** 2,
      ).toBeCloseTo(1, 12);
    }
  });
});

describe('forward process', () => {
  it('noiseTo keeps the variance of unit-variance data at 1 and interpolates data and noise', () => {
    const s = makeSchedule('linear', 50);
    const rng = createRng(1);
    const n = 20_000;
    const x0 = Float64Array.from({ length: n * 2 }, () => rng.normal());
    const eps = Float64Array.from({ length: n * 2 }, () => rng.normal());
    for (const t of [0, 10, 25, 50]) {
      const xt = noiseTo(s, x0, eps, t);
      const variance = xt.reduce((a, v) => a + v * v, 0) / xt.length;
      expect(variance).toBeGreaterThan(0.95);
      expect(variance).toBeLessThan(1.05);
    }
    expect(noiseTo(s, x0, eps, 0)).toEqual(x0);
    const end = noiseTo(s, x0, eps, 50);
    let corr = 0;
    for (let i = 0; i < end.length; i++) corr += (end[i] as number) * (eps[i] as number);
    expect(corr / end.length).toBeGreaterThan(0.95); // almost all noise
  });

  it('rejects times outside 0..T', () => {
    const s = makeSchedule('linear', 20);
    expect(() => noiseTo(s, [0, 0], [0, 0], 21)).toThrow(RangeError);
    expect(() => noiseTo(s, [0, 0], [0, 0], -1)).toThrow(RangeError);
    expect(() => noiseTo(s, [0, 0], [0, 0], 1.5)).toThrow(RangeError);
    expect(() => makeSchedule('linear', 5)).toThrow(RangeError);
    expect(() => makeSchedule('x' as 'linear', 50)).toThrow(RangeError);
  });

  it('time features have a fixed size and distinguish different times', () => {
    expect(timeFeatures(7, 50)).toHaveLength(TIME_FEATURES);
    expect(timeFeatures(7, 50)).not.toEqual(timeFeatures(8, 50));
    expect(timeFeatures(0, 50)[0]).toBe(0);
    expect(timeFeatures(50, 50)[0]).toBe(1);
  });
});

describe('sampling times', () => {
  it('DDPM visits every step; DDIM an evenly spaced decreasing subset including both ends', () => {
    expect(samplingTimes('ddpm', 5, 99)).toEqual([5, 4, 3, 2, 1, 0]);
    expect(samplingTimes('ddim', 50, 5)).toEqual([50, 40, 30, 20, 10, 0]);
    const t = samplingTimes('ddim', 50, 1);
    expect(t).toEqual([50, 0]);
    expect(samplingTimes('ddim', 10, 500)).toHaveLength(11); // never more than T steps
    expect(samplingTimes('ddim', 50, 7)[0]).toBe(50);
    expect(samplingTimes('ddim', 50, 7).at(-1)).toBe(0);
  });
});

/**
 * For data x₀ ~ N(0, σ²I) the best possible noise prediction is known exactly:
 * ε*(x, t) = √(1 − ᾱ_t) · x / (ᾱ_t σ² + 1 − ᾱ_t). Sampling with it must reproduce N(0, σ²I).
 */
describe.each(['ddpm', 'ddim'] as const)('%s sampler with the exact denoiser', (method) => {
  it('turns standard normal noise into the target Gaussian', () => {
    const sigma = 0.8;
    const schedule = makeSchedule('linear', 50);
    const exact: NoisePredictor = (x, t) => {
      const abar = schedule.alphaBar[t] as number;
      const k = Math.sqrt(1 - abar) / (abar * sigma * sigma + 1 - abar);
      return x.map((v) => v * k);
    };
    const rng = createRng(3);
    const n = 6000;
    const init = Float64Array.from({ length: n * 2 }, () => rng.normal());
    const { samples } = sampleDiffusion(exact, schedule, init, {
      method,
      steps: 25,
      rng: createRng(4),
    });
    const mean = samples.reduce((a, v) => a + v, 0) / samples.length;
    const std = Math.sqrt(samples.reduce((a, v) => a + (v - mean) ** 2, 0) / samples.length);
    expect(Math.abs(mean)).toBeLessThan(0.05);
    // Discrete steps are not exact (the late steps of this schedule are big), so a few percent of
    // under-dispersion is expected; a wrong sign or coefficient would blow up or collapse instead.
    expect(Math.abs(std - sigma)).toBeLessThan(0.12);
    expect(std).toBeLessThan(1);
  });
});

describe('sampler output', () => {
  const schedule = makeSchedule('linear', 20);
  const zero: NoisePredictor = (x) => new Float64Array(x.length);

  it('records the path of the tracked points, starting at the initial noise', () => {
    const rng = createRng(1);
    const init = Float64Array.from({ length: 10 * 2 }, () => rng.normal());
    const result = sampleDiffusion(zero, schedule, init, {
      method: 'ddpm',
      steps: 0,
      rng: createRng(2),
      track: 3,
    });
    expect(result.times).toHaveLength(21);
    expect(result.trajectory).toHaveLength(21 * 3 * 2);
    expect(Array.from(result.trajectory!.slice(0, 6))).toEqual(Array.from(init.slice(0, 6)));
    expect(Array.from(result.trajectory!.slice(-6))).toEqual(
      Array.from(result.samples.slice(0, 6)),
    );
    const none = sampleDiffusion(zero, schedule, init, {
      method: 'ddim',
      steps: 4,
      rng: createRng(2),
    });
    expect(none.trajectory).toBeNull();
    expect(none.samples).toHaveLength(20);
  });

  it('DDIM is deterministic, DDPM depends on its random source, and neither mutates the input', () => {
    const rng = createRng(1);
    const init = Float64Array.from({ length: 8 }, () => rng.normal());
    const copy = Float64Array.from(init);
    const predict: NoisePredictor = (x) => x.map((v) => v * 0.3);
    const a = sampleDiffusion(predict, schedule, init, {
      method: 'ddim',
      steps: 5,
      rng: createRng(1),
    });
    const b = sampleDiffusion(predict, schedule, init, {
      method: 'ddim',
      steps: 5,
      rng: createRng(99),
    });
    expect(a.samples).toEqual(b.samples);
    const c = sampleDiffusion(predict, schedule, init, {
      method: 'ddpm',
      steps: 0,
      rng: createRng(1),
    });
    const d = sampleDiffusion(predict, schedule, init, {
      method: 'ddpm',
      steps: 0,
      rng: createRng(2),
    });
    expect(c.samples).not.toEqual(d.samples);
    expect(init).toEqual(copy);
  });
});

describe('denoiser gradients', () => {
  it('match finite differences', () => {
    const schedule = makeSchedule('linear', 20);
    const net = new Denoiser({ hidden: 16 }, createRng(5));
    const rng = createRng(6);
    for (const p of net.parameters)
      if (!p.decay) for (let i = 0; i < p.data.length; i++) p.data[i] = rng.uniform(-0.3, 0.3);
    const x0 = Float64Array.from({ length: 12 }, () => rng.normal());
    const eps = Float64Array.from({ length: 12 }, () => rng.normal());
    const t = [3, 17, 8, 1, 20, 12];
    net.gradients(schedule, x0, t, eps);
    const loss = () => {
      // same computation without touching gradients
      const noisy = new Float64Array(12);
      for (let i = 0; i < 6; i++)
        noisy.set(
          noiseTo(
            schedule,
            x0.subarray(2 * i, 2 * i + 2),
            eps.subarray(2 * i, 2 * i + 2),
            t[i] as number,
          ),
          2 * i,
        );
      let total = 0;
      for (let i = 0; i < 6; i++) {
        const pred = net.predict(noisy.slice(2 * i, 2 * i + 2), t[i] as number, 20);
        total +=
          ((pred[0] as number) - (eps[2 * i] as number)) ** 2 +
          ((pred[1] as number) - (eps[2 * i + 1] as number)) ** 2;
      }
      return total / 12;
    };
    const h = 1e-6;
    let worst = 0;
    for (const p of net.parameters) {
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
    expect(worst).toBeLessThan(1e-4);
  });

  it('predict does not disturb a pending backward pass and validates the size', () => {
    const schedule = makeSchedule('linear', 20);
    const a = new Denoiser({ hidden: 16 }, createRng(1));
    const b = new Denoiser({ hidden: 16 }, createRng(1));
    const x0 = Float64Array.from([0.5, -0.2, 1, 1]);
    const eps = Float64Array.from([0.1, 0.2, -0.3, 0.4]);
    a.gradients(schedule, x0, [4, 9], eps);
    b.gradients(schedule, x0, [4, 9], eps);
    a.predict(Float64Array.from([1, 2, 3, 4]), 5, 20);
    a.parameters.forEach((p, i) => expect(p.grad).toEqual((b.parameters[i] as typeof p).grad));
    expect(() => new Denoiser({ hidden: 3 }, createRng(1))).toThrow(RangeError);
  });
});
