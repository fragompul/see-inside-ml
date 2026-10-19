/**
 * Noise schedules for denoising diffusion. Time runs t = 0 (clean data) … T (almost pure noise);
 * all arrays have length T + 1 and index 0 is the clean state (β₀ = 0, ᾱ₀ = 1).
 *
 *   q(x_t | x_0) = N(√ᾱ_t · x_0, (1 − ᾱ_t) · I),   ᾱ_t = ∏_{s ≤ t} (1 − β_s)
 */
export const SCHEDULE_KINDS = ['linear', 'cosine'] as const;
export type ScheduleKind = (typeof SCHEDULE_KINDS)[number];

export const noiseScheduleInfo: Record<ScheduleKind, { label: string; note: string }> = {
  linear: {
    label: 'Linear',
    note: 'Noise per step grows linearly; the data is destroyed fairly quickly.',
  },
  cosine: {
    label: 'Cosine',
    note: 'ᾱ follows a cosine: gentler at the start and end, so more steps are spent on mid-level noise.',
  },
};

export const DIFFUSION_LIMITS = { steps: [10, 200] } as const;

export interface NoiseSchedule {
  kind: ScheduleKind;
  steps: number;
  beta: Float64Array;
  alpha: Float64Array;
  alphaBar: Float64Array;
  sqrtAlphaBar: Float64Array;
  sqrtOneMinusAlphaBar: Float64Array;
}

export function makeSchedule(kind: ScheduleKind, steps: number): NoiseSchedule {
  const [lo, hi] = DIFFUSION_LIMITS.steps;
  if (!Number.isInteger(steps) || steps < lo || steps > hi)
    throw new RangeError(`Diffusion steps must be an integer in [${lo}, ${hi}]`);
  if (!(SCHEDULE_KINDS as readonly string[]).includes(kind))
    throw new RangeError('Unknown schedule');
  const beta = new Float64Array(steps + 1);
  if (kind === 'linear') {
    // the usual 1e-4 … 0.02 over 1000 steps, rescaled so any step count reaches similar noise
    const scale = 1000 / steps;
    const start = 1e-4 * scale;
    const end = 0.02 * scale;
    for (let t = 1; t <= steps; t++)
      beta[t] = Math.min(0.999, start + ((end - start) * (t - 1)) / Math.max(1, steps - 1));
  } else {
    const s = 0.008;
    const f = (t: number) => Math.cos(((t / steps + s) / (1 + s)) * (Math.PI / 2)) ** 2;
    for (let t = 1; t <= steps; t++) beta[t] = Math.min(0.999, 1 - f(t) / f(t - 1));
  }
  const alpha = new Float64Array(steps + 1);
  const alphaBar = new Float64Array(steps + 1);
  alpha[0] = 1;
  alphaBar[0] = 1;
  for (let t = 1; t <= steps; t++) {
    alpha[t] = 1 - (beta[t] as number);
    alphaBar[t] = (alphaBar[t - 1] as number) * (alpha[t] as number);
  }
  return {
    kind,
    steps,
    beta,
    alpha,
    alphaBar,
    sqrtAlphaBar: alphaBar.map(Math.sqrt),
    sqrtOneMinusAlphaBar: alphaBar.map((a) => Math.sqrt(1 - a)),
  };
}

/** Jumps straight to time `t`: `x_t = √ᾱ_t · x_0 + √(1 − ᾱ_t) · ε` for every point (`n × 2`). */
export function noiseTo(
  schedule: NoiseSchedule,
  x0: ArrayLike<number>,
  eps: ArrayLike<number>,
  t: number,
): Float64Array {
  if (!Number.isInteger(t) || t < 0 || t > schedule.steps)
    throw new RangeError('Time out of range');
  const a = schedule.sqrtAlphaBar[t] as number;
  const b = schedule.sqrtOneMinusAlphaBar[t] as number;
  const out = new Float64Array(x0.length);
  for (let i = 0; i < out.length; i++) out[i] = a * (x0[i] as number) + b * (eps[i] as number);
  return out;
}

/** Number of features `timeFeatures` returns. */
export const TIME_FEATURES = 9;

/** What the denoiser is told about the time: `t / T` and a few sines and cosines of it. */
export function timeFeatures(t: number, steps: number): number[] {
  const u = t / steps;
  const out = [u];
  for (const k of [1, 2, 4, 8]) out.push(Math.sin(Math.PI * k * u), Math.cos(Math.PI * k * u));
  return out;
}
