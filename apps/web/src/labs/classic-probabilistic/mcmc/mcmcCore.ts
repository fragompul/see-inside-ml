import {
  chainColumn,
  chainEss,
  chainMean,
  gibbsBivariateNormal,
  hamiltonian,
  metropolis,
  rHat,
  type Chain,
  type Gradient,
  type LogDensity,
} from '@sim/classic';

export type TargetId = 'gaussian' | 'banana' | 'modes';
export type SamplerId = 'metropolis' | 'gibbs' | 'hmc';

export interface Target {
  id: TargetId;
  label: string;
  note: string;
  logDensity: LogDensity;
  gradient: Gradient;
  /** Exact mean of each coordinate. */
  mean: [number, number];
  /** Where the chains start (dispersed on purpose, so R-hat has something to detect). */
  starts: [number, number][];
  bounds: { x0: number; x1: number; y0: number; y1: number };
}

const RHO = 0.95;

const lse = (a: number, b: number): number => {
  const m = Math.max(a, b);
  return m + Math.log(Math.exp(a - m) + Math.exp(b - m));
};

export const TARGETS: Record<TargetId, Target> = {
  gaussian: {
    id: 'gaussian',
    label: 'Correlated Gaussian',
    note: 'A narrow diagonal ridge (correlation 0.95): easy to describe, awkward to walk along with axis-aligned steps.',
    logDensity: (x) =>
      (-0.5 *
        ((x[0] as number) ** 2 -
          2 * RHO * (x[0] as number) * (x[1] as number) +
          (x[1] as number) ** 2)) /
      (1 - RHO * RHO),
    gradient: (x) => [
      (-(x[0] as number) + RHO * (x[1] as number)) / (1 - RHO * RHO),
      (-(x[1] as number) + RHO * (x[0] as number)) / (1 - RHO * RHO),
    ],
    mean: [0, 0],
    starts: [
      [-4, 4],
      [4, -4],
      [-4, -4],
      [4, 4],
    ],
    bounds: { x0: -4.5, x1: 4.5, y0: -4.5, y1: 4.5 },
  },
  banana: {
    id: 'banana',
    label: 'Banana',
    note: 'A curved ridge: the density is high only in a thin bent band, so the right direction to move keeps changing.',
    // x ~ N(0, 1); y | x ~ N(x^2 - 1, 0.5^2)
    logDensity: (x) =>
      -0.5 * (x[0] as number) ** 2 -
      0.5 * (((x[1] as number) - ((x[0] as number) ** 2 - 1)) / 0.5) ** 2,
    gradient: (x) => {
      const r = ((x[1] as number) - ((x[0] as number) ** 2 - 1)) / 0.25;
      return [-(x[0] as number) + r * 2 * (x[0] as number), -r];
    },
    mean: [0, 0],
    starts: [
      [-2.5, 5],
      [2.5, 5],
      [0, -1],
      [-1.5, 1],
    ],
    bounds: { x0: -3.5, x1: 3.5, y0: -2, y1: 7 },
  },
  modes: {
    id: 'modes',
    label: 'Two separated modes',
    note: 'Two blobs with a valley of near-zero density between them. A chain has to jump across the valley to see both.',
    logDensity: (x) => {
      const a = (-0.5 * (((x[0] as number) + 3) ** 2 + (x[1] as number) ** 2)) / 0.6;
      const b = (-0.5 * (((x[0] as number) - 3) ** 2 + (x[1] as number) ** 2)) / 0.6;
      return lse(a, b);
    },
    gradient: (x) => {
      const a = (-0.5 * (((x[0] as number) + 3) ** 2 + (x[1] as number) ** 2)) / 0.6;
      const b = (-0.5 * (((x[0] as number) - 3) ** 2 + (x[1] as number) ** 2)) / 0.6;
      const wa = 1 / (1 + Math.exp(b - a));
      return [
        (-wa * ((x[0] as number) + 3) - (1 - wa) * ((x[0] as number) - 3)) / 0.6,
        -(x[1] as number) / 0.6,
      ];
    },
    mean: [0, 0],
    starts: [
      [-3, 0],
      [3, 0],
      [-3, 1],
      [3, -1],
    ],
    bounds: { x0: -6, x1: 6, y0: -3.5, y1: 3.5 },
  },
};

export interface McmcScenario {
  target: TargetId;
  sampler: SamplerId;
  steps: number;
  burnIn: number;
  /** Random-walk jump size (Metropolis). */
  proposalSd: number;
  /** Leapfrog step size and count (HMC). */
  stepSize: number;
  leapfrog: number;
  seed: number;
}

export const DEFAULT_MCMC: McmcScenario = {
  target: 'gaussian',
  sampler: 'metropolis',
  steps: 1500,
  burnIn: 200,
  proposalSd: 0.3,
  stepSize: 0.1,
  leapfrog: 20,
  seed: 1,
};

export interface McmcRun {
  scenario: McmcScenario;
  target: Target;
  chains: Chain[];
  acceptance: number;
  /** Effective sample size of x, summed over the four chains (after burn-in). */
  ess: number;
  /** Total number of draws kept (after burn-in). */
  kept: number;
  rHat: number;
  /** Absolute error of the estimated mean of x and y. */
  meanError: number;
  /** For the two-mode target: share of kept draws in the right-hand mode (0.5 expected). */
  rightShare: number;
}

export function runMcmc(s: McmcScenario): McmcRun {
  const target = TARGETS[s.target];
  if (s.sampler === 'gibbs' && s.target !== 'gaussian')
    throw new Error(
      'Gibbs sampling needs the exact conditional distributions, which this lab only has for the correlated Gaussian. Pick another sampler or distribution.',
    );
  const chains = target.starts.map((start, i) => {
    const seed = s.seed * 100 + i;
    if (s.sampler === 'gibbs') return gibbsBivariateNormal(RHO, start, s.steps, seed);
    if (s.sampler === 'hmc')
      return hamiltonian(target.logDensity, target.gradient, start, {
        steps: s.steps,
        stepSize: s.stepSize,
        leapfrog: s.leapfrog,
        seed,
      });
    return metropolis(target.logDensity, start, { steps: s.steps, proposalSd: s.proposalSd, seed });
  });
  const kept = chains.map((c) => chainColumn({ ...c, samples: c.samples.slice(s.burnIn) }, 0));
  const keptY = chains.map((c) => chainColumn({ ...c, samples: c.samples.slice(s.burnIn) }, 1));
  const all = kept.flat();
  const allY = keptY.flat();
  const mx = chainMean(all);
  const my = chainMean(allY);
  return {
    scenario: s,
    target,
    chains,
    acceptance: chains.reduce((a, c) => a + c.acceptance, 0) / chains.length,
    ess: kept.reduce((a, k) => a + chainEss(k), 0),
    kept: all.length,
    rHat: rHat(kept),
    meanError: Math.hypot(mx - target.mean[0], my - target.mean[1]),
    rightShare: all.filter((v) => v > 0).length / all.length,
  };
}
