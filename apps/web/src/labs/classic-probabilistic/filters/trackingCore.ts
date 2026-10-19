import {
  constantVelocity2D,
  kalmanFilter,
  particleFilter,
  rtsSmoother,
  type KalmanStep,
  type ParticleStep,
} from '@sim/classic';
import { Matrix, createRng } from '@sim/core';

/**
 * Tracking a moving target from noisy position readings: the true path is hidden, the sensor
 * gives a jittery (sometimes missing, sometimes wild) reading of it, and the filters estimate where
 * the target really is.
 */
export interface TrackingScenario {
  /** Sensor noise (standard deviation, in the units of the plane). */
  sensorNoise: number;
  /** How much random acceleration the target really has. */
  truePush: number;
  /** The path: straight cruising or a steady turn (which a straight-line model gets wrong). */
  motion: 'cruise' | 'turn';
  /** Share of readings that are lost. */
  dropout: number;
  /** Share of readings that are wild (far from the truth). */
  outliers: number;
  /** How much random acceleration the filters EXPECT (their trust in the straight-line model). */
  assumedPush: number;
  /** Likelihood of the particle filter: tuned to Gaussian noise or tolerant of wild readings. */
  particleLikelihood: 'gaussian' | 'robust';
  particles: number;
  steps: number;
  seed: number;
}

export const DEFAULT_TRACKING: TrackingScenario = {
  sensorNoise: 3,
  truePush: 0.15,
  motion: 'cruise',
  dropout: 0,
  outliers: 0,
  assumedPush: 0.15,
  particleLikelihood: 'gaussian',
  particles: 1500,
  steps: 80,
  seed: 6,
};

export interface TrackingRun {
  scenario: TrackingScenario;
  /** True positions `[x, y]` per step. */
  truth: [number, number][];
  /** Readings, or null where lost. */
  readings: ([number, number] | null)[];
  kalman: KalmanStep[];
  smoothed: { mean: number[]; cov: Matrix }[];
  particles: ParticleStep[];
  /** Root-mean-square position error of each estimate (the raw readings only where they exist). */
  rmse: { raw: number; kalman: number; smoother: number; particle: number };
}

const rmse = (
  est: readonly (readonly number[] | null)[],
  truth: readonly [number, number][],
): number => {
  let s = 0;
  let n = 0;
  est.forEach((e, t) => {
    if (!e) return;
    s += ((e[0] as number) - truth[t]![0]) ** 2 + ((e[1] as number) - truth[t]![1]) ** 2;
    n++;
  });
  return Math.sqrt(s / Math.max(n, 1));
};

export function runTracking(s: TrackingScenario): TrackingRun {
  const rng = createRng(s.seed);
  // --- the world
  let x = 0;
  let y = 0;
  let vx = 1.2;
  let vy = 0.5;
  const omega = s.motion === 'turn' ? 0.09 : 0;
  const truth: [number, number][] = [];
  for (let t = 0; t < s.steps; t++) {
    truth.push([x, y]);
    const c = Math.cos(omega);
    const si = Math.sin(omega);
    [vx, vy] = [
      vx * c - vy * si + rng.normal() * s.truePush,
      vx * si + vy * c + rng.normal() * s.truePush,
    ];
    x += vx;
    y += vy;
  }
  const readings: ([number, number] | null)[] = truth.map(([tx, ty], t) => {
    if (t > 0 && rng.next() < s.dropout) return null;
    const wild = t > 0 && rng.next() < s.outliers;
    const sd = wild ? s.sensorNoise * 8 : s.sensorNoise;
    return [tx + rng.normal() * sd, ty + rng.normal() * sd];
  });

  // --- the filters (they know the straight-line model but not the world)
  const model = constantVelocity2D(1, Math.max(s.assumedPush, 1e-3), s.sensorNoise);
  const first = readings[0] as [number, number];
  const P0 = Matrix.from([
    [s.sensorNoise ** 2, 0, 0, 0],
    [0, s.sensorNoise ** 2, 0, 0],
    [0, 0, 4, 0],
    [0, 0, 0, 4],
  ]);
  const kalman = kalmanFilter(model, readings, [first[0], first[1], 0, 0], P0);
  const smoothed = rtsSmoother(model, kalman);

  const sd = s.sensorNoise;
  // a little extra jitter on the velocity ("roughening") keeps the cloud from collapsing onto one velocity
  const push = Math.max(s.assumedPush, 0.3);
  const particles = particleFilter<[number, number]>(
    {
      initial: (g) => [
        first[0] + g.normal() * sd,
        first[1] + g.normal() * sd,
        g.normal() * 2,
        g.normal() * 2,
      ],
      propagate: (st, g) => {
        const ax = g.normal() * push;
        const ay = g.normal() * push;
        return [
          (st[0] as number) + (st[2] as number) + 0.5 * ax,
          (st[1] as number) + (st[3] as number) + 0.5 * ay,
          (st[2] as number) + ax,
          (st[3] as number) + ay,
        ];
      },
      logLikelihood: (st, z) => {
        const d2 = ((z[0] - (st[0] as number)) ** 2 + (z[1] - (st[1] as number)) ** 2) / (sd * sd);
        // robust: a heavy-tailed bump, so one wild reading cannot veto every particle
        return s.particleLikelihood === 'gaussian' ? -0.5 * d2 : -1.5 * Math.log(1 + d2 / 3);
      },
    },
    readings,
    { particles: s.particles, seed: s.seed + 1 },
  );

  return {
    scenario: s,
    truth,
    readings,
    kalman,
    smoothed,
    particles,
    rmse: {
      raw: rmse(readings, truth),
      kalman: rmse(
        kalman.map((k) => k.mean),
        truth,
      ),
      smoother: rmse(
        smoothed.map((k) => k.mean),
        truth,
      ),
      particle: rmse(
        particles.map((p) => p.mean),
        truth,
      ),
    },
  };
}
