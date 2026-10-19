import { type Rng } from '@sim/core';
import type { NoiseSchedule } from './schedule';

/** Predicts the noise ε that was added to produce the points `x` (`n × 2`) at time `t`. */
export type NoisePredictor = (x: Float64Array, t: number) => Float64Array;

export const SAMPLERS = ['ddpm', 'ddim'] as const;
export type SamplerName = (typeof SAMPLERS)[number];

export const samplerInfo: Record<SamplerName, { label: string; note: string }> = {
  ddpm: {
    label: 'DDPM',
    note: 'Walks back through every one of the T steps and adds a little fresh noise at each.',
  },
  ddim: {
    label: 'DDIM',
    note: 'Deterministic: jumps along a subset of the steps, so a handful is enough.',
  },
};

export interface SampleOptions {
  method: SamplerName;
  /** DDIM only: how many denoising steps to take (DDPM always takes all of them). */
  steps: number;
  /** Random source for DDPM's fresh noise. */
  rng: Rng;
  /** Record the full path of the first `track` points. */
  track?: number;
}

export interface SampleResult {
  /** Final points, `n × 2`. */
  samples: Float64Array;
  /** Times visited, from T down to 0. */
  times: number[];
  /** `times.length × track × 2` positions of the tracked points (null if `track` is 0). */
  trajectory: Float64Array | null;
}

/** The decreasing times a sampler visits: all of T … 0 for DDPM, evenly spaced for DDIM. */
export function samplingTimes(method: SamplerName, total: number, steps: number): number[] {
  if (method === 'ddpm') return Array.from({ length: total + 1 }, (_, i) => total - i);
  const n = Math.max(1, Math.min(total, Math.round(steps)));
  const times = Array.from({ length: n + 1 }, (_, i) => Math.round(total - (i * total) / n));
  return [...new Set(times)];
}

const CLIP = 6;

/**
 * Turns noise into data: starting from `init` (`n × 2`, standard normal) the points are denoised
 * step by step with the predictor. Pure in the predictor, so it is testable against exact ones.
 */
export function sampleDiffusion(
  predict: NoisePredictor,
  schedule: NoiseSchedule,
  init: Float64Array,
  options: SampleOptions,
): SampleResult {
  const n = init.length / 2;
  const times = samplingTimes(options.method, schedule.steps, options.steps);
  const track = Math.min(options.track ?? 0, n);
  const trajectory = track > 0 ? new Float64Array(times.length * track * 2) : null;
  const x = Float64Array.from(init);
  const record = (k: number) => trajectory?.set(x.subarray(0, track * 2), k * track * 2);
  record(0);

  for (let k = 0; k < times.length - 1; k++) {
    const t = times[k] as number;
    const prev = times[k + 1] as number;
    const eps = predict(x, t);
    const abar = schedule.alphaBar[t] as number;
    const abarPrev = schedule.alphaBar[prev] as number;
    if (options.method === 'ddpm') {
      // t → t − 1 with the posterior variance β̃_t
      const alpha = schedule.alpha[t] as number;
      const beta = schedule.beta[t] as number;
      const coef = beta / Math.sqrt(1 - abar);
      const variance = (beta * (1 - abarPrev)) / (1 - abar);
      const sigma = prev === 0 ? 0 : Math.sqrt(variance);
      for (let i = 0; i < x.length; i++) {
        const mean = ((x[i] as number) - coef * (eps[i] as number)) / Math.sqrt(alpha);
        x[i] = mean + (sigma > 0 ? sigma * options.rng.normal() : 0);
      }
    } else {
      // DDIM (η = 0): estimate x₀, then re-noise it to the earlier time with the same ε
      const sa = Math.sqrt(abar);
      const sb = Math.sqrt(1 - abar);
      const pa = Math.sqrt(abarPrev);
      const pb = Math.sqrt(1 - abarPrev);
      for (let i = 0; i < x.length; i++) {
        const e = eps[i] as number;
        const x0 = Math.max(-CLIP, Math.min(CLIP, ((x[i] as number) - sb * e) / sa));
        x[i] = pa * x0 + pb * e;
      }
    }
    record(k + 1);
  }
  return { samples: x, times, trajectory };
}
