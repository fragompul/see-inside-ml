import { createRng } from '@sim/core';

/**
 * Markov chain Monte Carlo: draw samples from a distribution known only up to a constant, by
 * walking through it. Random-walk Metropolis proposes a nearby point and accepts it with probability
 * min(1, p(new)/p(old)); Gibbs resamples one coordinate at a time from its exact conditional;
 * Hamiltonian Monte Carlo gives the walker momentum and simulates physics on -log p, so it can take
 * long, informed steps. Diagnostics tell whether the chain explored enough.
 */
export type LogDensity = (x: readonly number[]) => number;
export type Gradient = (x: readonly number[]) => number[];

export interface Chain {
  /** The state after each step (the start is not included). */
  samples: number[][];
  /** Proposed point of every step (equal to the sample when the move was rejected). */
  proposals: number[][];
  accepted: boolean[];
  /** Share of proposals that were accepted. */
  acceptance: number;
}

export interface MetropolisOptions {
  steps: number;
  /** Standard deviation of the Gaussian jump, in every coordinate. */
  proposalSd: number;
  seed?: number;
}

export function metropolis(
  logTarget: LogDensity,
  start: readonly number[],
  options: MetropolisOptions,
): Chain {
  const rng = createRng(options.seed ?? 1);
  let x = [...start];
  let lp = logTarget(x);
  const samples: number[][] = [];
  const proposals: number[][] = [];
  const accepted: boolean[] = [];
  for (let i = 0; i < options.steps; i++) {
    const cand = x.map((v) => v + rng.normal() * options.proposalSd);
    const lc = logTarget(cand);
    const ok = Math.log(rng.next()) < lc - lp;
    if (ok) {
      x = cand;
      lp = lc;
    }
    proposals.push(cand);
    accepted.push(ok);
    samples.push([...x]);
  }
  return {
    samples,
    proposals,
    accepted,
    acceptance: accepted.filter(Boolean).length / Math.max(options.steps, 1),
  };
}

export interface HamiltonianOptions {
  steps: number;
  stepSize: number;
  /** Leapfrog steps per trajectory. */
  leapfrog: number;
  seed?: number;
}

/** Hamiltonian Monte Carlo with unit mass and the leapfrog integrator. */
export function hamiltonian(
  logTarget: LogDensity,
  gradient: Gradient,
  start: readonly number[],
  options: HamiltonianOptions,
): Chain {
  const rng = createRng(options.seed ?? 1);
  const { stepSize: eps, leapfrog: L } = options;
  let x = [...start];
  let lp = logTarget(x);
  const samples: number[][] = [];
  const proposals: number[][] = [];
  const accepted: boolean[] = [];
  for (let i = 0; i < options.steps; i++) {
    const p0 = x.map(() => rng.normal());
    let q = [...x];
    let p = [...p0];
    let g = gradient(q);
    p = p.map((v, d) => v + 0.5 * eps * (g[d] as number));
    for (let s = 0; s < L; s++) {
      q = q.map((v, d) => v + eps * (p[d] as number));
      g = gradient(q);
      const scale = s === L - 1 ? 0.5 : 1;
      p = p.map((v, d) => v + scale * eps * (g[d] as number));
    }
    const lq = logTarget(q);
    const k0 = 0.5 * p0.reduce((s, v) => s + v * v, 0);
    const k1 = 0.5 * p.reduce((s, v) => s + v * v, 0);
    const logAccept = lq - k1 - (lp - k0);
    const ok = Number.isFinite(logAccept) && Math.log(rng.next()) < logAccept;
    if (ok) {
      x = q;
      lp = lq;
    }
    proposals.push(q);
    accepted.push(ok);
    samples.push([...x]);
  }
  return {
    samples,
    proposals,
    accepted,
    acceptance: accepted.filter(Boolean).length / Math.max(options.steps, 1),
  };
}

/** Gibbs sampling of a bivariate normal: alternate exact draws from the two conditionals. */
export function gibbsBivariateNormal(
  correlation: number,
  start: readonly [number, number],
  steps: number,
  seed = 1,
): Chain {
  const rng = createRng(seed);
  const sd = Math.sqrt(1 - correlation * correlation);
  let y = start[1];
  const samples: number[][] = [];
  const proposals: number[][] = [];
  for (let i = 0; i < steps; i++) {
    const x = correlation * y + sd * rng.normal();
    proposals.push([x, y]);
    y = correlation * x + sd * rng.normal();
    samples.push([x, y]);
  }
  return { samples, proposals, accepted: samples.map(() => true), acceptance: 1 };
}

// ---- diagnostics -------------------------------------------------------------------------------

export function autocorrelation(x: ArrayLike<number>, maxLag: number): number[] {
  const n = x.length;
  let mean = 0;
  for (let i = 0; i < n; i++) mean += (x[i] as number) / n;
  let c0 = 0;
  for (let i = 0; i < n; i++) c0 += ((x[i] as number) - mean) ** 2;
  const out: number[] = [];
  for (let lag = 0; lag <= Math.min(maxLag, n - 1); lag++) {
    let s = 0;
    for (let i = 0; i + lag < n; i++)
      s += ((x[i] as number) - mean) * ((x[i + lag] as number) - mean);
    out.push(c0 > 0 ? s / c0 : 0);
  }
  return out;
}

/**
 * Effective sample size: how many independent draws the correlated chain is worth,
 * `N / (1 + 2 Σ ρ_k)`, summing the autocorrelations while consecutive pairs stay positive (Geyer).
 */
export function chainEss(x: ArrayLike<number>): number {
  const n = x.length;
  const rho = autocorrelation(x, Math.min(n - 1, 1000));
  let sum = 0;
  for (let k = 1; k + 1 < rho.length; k += 2) {
    const pair = (rho[k] as number) + (rho[k + 1] as number);
    if (pair < 0) break;
    sum += pair;
  }
  const tau = Math.max(1 + 2 * sum, 1e-9);
  return Math.min(n, n / tau);
}

/** Split R-hat (Gelman-Rubin): about 1 when chains agree, clearly above 1.05 when they do not. */
export function rHat(chains: readonly ArrayLike<number>[]): number {
  // split every chain in two so a single drifting chain is also detected
  const parts: ArrayLike<number>[] = [];
  for (const c of chains) {
    const half = Math.floor(c.length / 2);
    parts.push(
      Array.prototype.slice.call(c, 0, half),
      Array.prototype.slice.call(c, half, 2 * half),
    );
  }
  const m = parts.length;
  const n = (parts[0] as ArrayLike<number>).length;
  const means = parts.map((p) => {
    let s = 0;
    for (let i = 0; i < n; i++) s += (p[i] as number) / n;
    return s;
  });
  const vars = parts.map((p, j) => {
    let s = 0;
    for (let i = 0; i < n; i++) s += ((p[i] as number) - (means[j] as number)) ** 2;
    return s / (n - 1);
  });
  const grand = means.reduce((a, b) => a + b, 0) / m;
  const between = (n / (m - 1)) * means.reduce((s, v) => s + (v - grand) ** 2, 0);
  const within = vars.reduce((a, b) => a + b, 0) / m;
  const varPlus = ((n - 1) / n) * within + between / n;
  return Math.sqrt(varPlus / (within || 1e-300));
}

/** Kolmogorov-Smirnov distance between a sample and a cumulative distribution function. */
export function ksStatistic(samples: ArrayLike<number>, cdf: (x: number) => number): number {
  const s = Array.from(samples).sort((a, b) => a - b);
  const n = s.length;
  let d = 0;
  for (let i = 0; i < n; i++) {
    const f = cdf(s[i] as number);
    d = Math.max(d, Math.abs((i + 1) / n - f), Math.abs(f - i / n));
  }
  return d;
}

export const chainMean = (x: ArrayLike<number>): number => {
  let s = 0;
  for (let i = 0; i < x.length; i++) s += x[i] as number;
  return s / x.length;
};

export const chainColumn = (chain: Chain, d: number): number[] =>
  chain.samples.map((s) => s[d] as number);
