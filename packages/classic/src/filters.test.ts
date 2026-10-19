import { Matrix, createRng } from '@sim/core';
import { describe, expect, it } from 'vitest';
import {
  constantVelocity2D,
  effectiveSampleSize,
  kalmanFilter,
  particleFilter,
  rtsSmoother,
  steadyStateGain,
  systematicResample,
  type LinearGaussianModel,
} from './filters';

const scalar = (q: number, r: number): LinearGaussianModel => ({
  F: Matrix.from([[1]]),
  H: Matrix.from([[1]]),
  Q: Matrix.from([[q]]),
  R: Matrix.from([[r]]),
});

describe('Kalman filter against closed forms', () => {
  it('the random-walk gain converges to the Riccati solution', () => {
    const q = 0.3;
    const r = 1.7;
    // steady-state prior variance P- solves P- = q + P-r/(P-+r), i.e. P-^2 - qP- - qr = 0
    const prior = (q + Math.sqrt(q * q + 4 * q * r)) / 2;
    const gain = prior / (prior + r);
    expect(steadyStateGain(scalar(q, r)).get(0, 0)).toBeCloseTo(gain, 10);
  });

  it('with no process noise and a vague prior it is the running mean with variance r/n', () => {
    const rng = createRng(2);
    const zs = Array.from({ length: 40 }, () => [3 + rng.normal() * 0.5]);
    const steps = kalmanFilter(scalar(0, 0.25), zs, [0], Matrix.from([[1e9]]));
    const mean = zs.reduce((s, z) => s + (z[0] as number), 0) / zs.length;
    const last = steps[steps.length - 1]!;
    expect(last.mean[0]).toBeCloseTo(mean, 6);
    expect(last.cov.get(0, 0)).toBeCloseTo(0.25 / zs.length, 6);
  });

  it('a missing measurement only grows the uncertainty', () => {
    const steps = kalmanFilter(scalar(0.1, 1), [[0.5], null, null, [0.4]], [0], Matrix.from([[1]]));
    expect(steps[1]!.gain).toBeNull();
    expect(steps[2]!.cov.get(0, 0)).toBeGreaterThan(steps[1]!.cov.get(0, 0));
    expect(steps[3]!.cov.get(0, 0)).toBeLessThan(steps[2]!.cov.get(0, 0));
  });

  it('the covariance stays symmetric positive definite for a 4-D tracker', () => {
    const model = constantVelocity2D(1, 0.1, 1);
    const rng = createRng(4);
    const zs = Array.from({ length: 60 }, (_, t) => [
      t * 0.5 + rng.normal(),
      t * 0.2 + rng.normal(),
    ]);
    const P0 = Matrix.from([
      [10, 0, 0, 0],
      [0, 10, 0, 0],
      [0, 0, 10, 0],
      [0, 0, 0, 10],
    ]);
    for (const s of kalmanFilter(model, zs, [0, 0, 0, 0], P0)) {
      for (let i = 0; i < 4; i++) {
        expect(s.cov.get(i, i)).toBeGreaterThan(0);
        for (let j = 0; j < 4; j++) expect(s.cov.get(i, j)).toBeCloseTo(s.cov.get(j, i), 9);
      }
    }
  });

  it('the smoother is never less certain than the filter and agrees at the last step', () => {
    const model = scalar(0.2, 1);
    const rng = createRng(5);
    let x = 0;
    const zs = Array.from({ length: 30 }, () => {
      x += rng.normal() * Math.sqrt(0.2);
      return [x + rng.normal()];
    });
    const filt = kalmanFilter(model, zs, [0], Matrix.from([[1]]));
    const smooth = rtsSmoother(model, filt);
    smooth.forEach((s, t) =>
      expect(s.cov.get(0, 0)).toBeLessThanOrEqual(filt[t]!.cov.get(0, 0) + 1e-12),
    );
    expect(smooth[29]!.mean[0]).toBeCloseTo(filt[29]!.mean[0]!, 12);
    // and it is strictly better somewhere in the middle
    expect(smooth[10]!.cov.get(0, 0)).toBeLessThan(filt[10]!.cov.get(0, 0));
  });

  it('the log-likelihood of the measurements is the sum of the one-step predictive densities', () => {
    const steps = kalmanFilter(scalar(0, 1), [[0]], [0], Matrix.from([[1]]));
    // z ~ N(0, 1 + 1): log density of 0
    expect(steps[0]!.logLikelihood).toBeCloseTo(-0.5 * Math.log(2 * Math.PI * 2), 10);
  });
});

describe('particle filter', () => {
  it('resampling keeps particles in proportion to their weight', () => {
    const idx = systematicResample([0.7, 0.2, 0.1, 0], 0.5);
    expect(idx).toHaveLength(4);
    expect(idx.filter((i) => i === 0).length).toBeGreaterThanOrEqual(2);
    expect(idx).not.toContain(3);
  });

  it('effective sample size is N for equal weights and 1 for a single survivor', () => {
    expect(effectiveSampleSize([0.25, 0.25, 0.25, 0.25])).toBeCloseTo(4, 12);
    expect(effectiveSampleSize([1, 0, 0, 0])).toBeCloseTo(1, 12);
  });

  it('approaches the Kalman filter on a linear Gaussian problem', () => {
    const q = 0.2;
    const r = 1;
    const model = scalar(q, r);
    const rng = createRng(7);
    let x = 0;
    const zs = Array.from({ length: 40 }, () => {
      x += rng.normal() * Math.sqrt(q);
      return x + rng.normal() * Math.sqrt(r);
    });
    const kf = kalmanFilter(
      model,
      zs.map((z) => [z]),
      [0],
      Matrix.from([[1]]),
    );
    const pf = particleFilter<number>(
      {
        initial: (g) => [g.normal()],
        propagate: (s, g) => [(s[0] as number) + g.normal() * Math.sqrt(q)],
        logLikelihood: (s, z) => (-0.5 * (z - (s[0] as number)) ** 2) / r,
      },
      zs,
      { particles: 4000, seed: 3 },
    );
    let worst = 0;
    pf.forEach(
      (p, t) =>
        (worst = Math.max(worst, Math.abs((p.mean[0] as number) - (kf[t]!.mean[0] as number)))),
    );
    expect(worst).toBeLessThan(0.15);
    for (const p of pf) {
      expect(p.ess).toBeGreaterThanOrEqual(1);
      expect(p.ess).toBeLessThanOrEqual(4000 + 1e-6);
      expect(p.weights.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 9);
    }
  });

  it('a heavy-tailed likelihood shrugs off an outlier that drags the Kalman filter', () => {
    const q = 0.01;
    const zs = Array.from({ length: 30 }, () => 0);
    zs[15] = 30; // one wild reading of a target that stays at 0
    const kf = kalmanFilter(
      scalar(q, 0.25),
      zs.map((z) => [z]),
      [0],
      Matrix.from([[1]]),
    );
    const pf = particleFilter<number>(
      {
        initial: (g) => [g.normal()],
        propagate: (s, g) => [(s[0] as number) + g.normal() * Math.sqrt(q)],
        // Student-t-like (Cauchy) likelihood
        logLikelihood: (s, z) => -Math.log(1 + ((z - (s[0] as number)) / 0.5) ** 2),
      },
      zs,
      { particles: 3000, seed: 4 },
    );
    expect(Math.abs(kf[15]!.mean[0] as number)).toBeGreaterThan(5);
    expect(Math.abs(pf[15]!.mean[0] as number)).toBeLessThan(1.5);
  });
});
