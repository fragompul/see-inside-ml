import { Matrix, createRng, normalCdf, type Rng } from '@sim/core';
import { GaussianProcess } from './gp';

/**
 * Hyperparameter search on a score you can only measure by running an experiment. A landscape maps a
 * configuration in the unit square (two hyperparameters rescaled to [0, 1]) to a score to maximise;
 * measuring it is noisy, and a cheap measurement (a short training run, a subsample) is noisier than a
 * full one. Grid search, random search, successive halving and Bayesian optimisation (a Gaussian
 * process plus expected improvement) all try to find the best configuration within a budget.
 */
export interface Landscape {
  id: string;
  label: string;
  note: string;
  /** Noise-free score of a configuration. */
  value(x: readonly number[]): number;
  /** Best achievable score. */
  best: number;
  /** A configuration that attains it. */
  argBest: [number, number];
}

const gauss = (d2: number, w: number) => Math.exp(-d2 / (2 * w * w));

export const LANDSCAPES: Record<string, Landscape> = {
  bowl: {
    id: 'bowl',
    label: 'Smooth bowl',
    note: 'One broad optimum: the easy case.',
    value: (x) => 0.95 - 1.6 * ((x[0]! - 0.62) ** 2 + (x[1]! - 0.35) ** 2),
    best: 0.95,
    argBest: [0.62, 0.35],
  },
  ridge: {
    id: 'ridge',
    label: 'Narrow ridge',
    note: 'A thin diagonal band of good settings, as when two hyperparameters trade off against each other.',
    value: (x) => 0.35 + 0.6 * gauss((x[0]! + x[1]! - 1.1) ** 2, 0.07) * (0.6 + 0.4 * x[0]!),
    best: 0, // set below, once the maximum has been found on a fine grid
    argBest: [0.5, 0.6],
  },
  needle: {
    id: 'needle',
    label: 'Needle in a haystack',
    note: 'A flat plateau with one small spike of good settings.',
    value: (x) => 0.3 + 0.65 * gauss((x[0]! - 0.74) ** 2 + (x[1]! - 0.22) ** 2, 0.06),
    best: 0.95,
    argBest: [0.74, 0.22],
  },
  oneThing: {
    id: 'oneThing',
    label: 'Only one setting matters',
    note: 'The score depends on the first hyperparameter alone; the second is irrelevant.',
    value: (x) => 0.9 - 2.4 * (x[0]! - 0.5) ** 2,
    best: 0.9,
    argBest: [0.5, 0.5],
  },
  decoys: {
    id: 'decoys',
    label: 'Decoy peaks',
    note: 'Several local optima; one of them, a narrow one, is the best.',
    value: (x) =>
      0.25 +
      0.45 * gauss((x[0]! - 0.2) ** 2 + (x[1]! - 0.75) ** 2, 0.14) +
      0.4 * gauss((x[0]! - 0.8) ** 2 + (x[1]! - 0.8) ** 2, 0.12) +
      0.55 * gauss((x[0]! - 0.45) ** 2 + (x[1]! - 0.2) ** 2, 0.06),
    best: 0.8,
    argBest: [0.45, 0.2],
  },
};

// the ridge's true maximum, found once on a fine grid (the formula is not trivially maximised by hand)
{
  const ridge = LANDSCAPES.ridge as Landscape;
  let best = -Infinity;
  let arg: [number, number] = [0.5, 0.6];
  for (let i = 0; i <= 400; i++)
    for (let j = 0; j <= 400; j++) {
      const v = ridge.value([i / 400, j / 400]);
      if (v > best) {
        best = v;
        arg = [i / 400, j / 400];
      }
    }
  ridge.best = best;
  ridge.argBest = arg;
}

export interface Evaluation {
  x: [number, number];
  /** What was measured (noisy). */
  observed: number;
  /** Fraction of a full experiment this measurement used. */
  fidelity: number;
}

export interface SearchResult {
  method: string;
  evaluations: Evaluation[];
  /** Total cost in units of one full experiment. */
  cost: number;
  /** After each evaluation: the true score of the configuration that looks best so far. */
  incumbent: number[];
  /** True score of the final recommendation. */
  finalValue: number;
  /** Final regret: how far that is from the best achievable score. */
  regret: number;
  recommended: [number, number];
}

/** One noisy measurement: noise falls with the square root of the fraction of the experiment used. */
export function measure(
  land: Landscape,
  x: readonly number[],
  noise: number,
  fidelity: number,
  rng: Rng,
): number {
  return land.value(x) + (rng.normal() * noise) / Math.sqrt(fidelity);
}

function summarise(
  method: string,
  land: Landscape,
  evaluations: Evaluation[],
  pickFinal?: (evals: Evaluation[]) => [number, number],
): SearchResult {
  const incumbent: number[] = [];
  let bestObs = -Infinity;
  let bestX: [number, number] = [0.5, 0.5];
  for (const e of evaluations) {
    // only full-fidelity measurements compete for the incumbent (cheap ones are too noisy to trust)
    if (e.fidelity >= 1 - 1e-9 && e.observed > bestObs) {
      bestObs = e.observed;
      bestX = e.x;
    } else if (!isFinite(bestObs) && e === evaluations[0]) bestX = e.x;
    incumbent.push(land.value(bestX));
  }
  const recommended = pickFinal ? pickFinal(evaluations) : bestX;
  const finalValue = land.value(recommended);
  return {
    method,
    evaluations,
    cost: evaluations.reduce((s, e) => s + e.fidelity, 0),
    incumbent,
    finalValue,
    regret: land.best - finalValue,
    recommended,
  };
}

export function gridSearch(land: Landscape, budget: number, noise: number, seed = 1): SearchResult {
  const rng = createRng(seed);
  const side = Math.max(1, Math.floor(Math.sqrt(budget)));
  const evals: Evaluation[] = [];
  for (let i = 0; i < side; i++)
    for (let j = 0; j < side; j++) {
      const x: [number, number] = [(i + 0.5) / side, (j + 0.5) / side];
      evals.push({ x, observed: measure(land, x, noise, 1, rng), fidelity: 1 });
    }
  return summarise('grid', land, evals);
}

export function randomSearch(
  land: Landscape,
  budget: number,
  noise: number,
  seed = 1,
): SearchResult {
  const rng = createRng(seed);
  const evals: Evaluation[] = [];
  for (let i = 0; i < budget; i++) {
    const x: [number, number] = [rng.next(), rng.next()];
    evals.push({ x, observed: measure(land, x, noise, 1, rng), fidelity: 1 });
  }
  return summarise('random', land, evals);
}

/** Expected improvement over the best value so far (maximisation), with exploration margin `xi`. */
export function expectedImprovement(mean: number, std: number, best: number, xi = 0): number {
  if (std <= 1e-12) return Math.max(mean - best - xi, 0);
  const z = (mean - best - xi) / std;
  const pdf = Math.exp(-0.5 * z * z) / Math.sqrt(2 * Math.PI);
  return (mean - best - xi) * normalCdf(z) + std * pdf;
}

export interface BayesOptions {
  initial?: number;
  /** Exploration margin of the expected improvement. */
  xi?: number;
  lengthScale?: number;
}

export function bayesianOptimization(
  land: Landscape,
  budget: number,
  noise: number,
  seed = 1,
  options: BayesOptions = {},
): SearchResult {
  const { initial = 5, xi = 0.01, lengthScale = 0.2 } = options;
  const rng = createRng(seed);
  const evals: Evaluation[] = [];
  const add = (x: [number, number]) =>
    evals.push({ x, observed: measure(land, x, noise, 1, rng), fidelity: 1 });
  for (let i = 0; i < Math.min(initial, budget); i++) add([rng.next(), rng.next()]);
  while (evals.length < budget) {
    const xs = new Matrix(evals.length, 2, Float64Array.from(evals.flatMap((e) => e.x)));
    const ys = evals.map((e) => e.observed);
    const spread = Math.max(
      1e-6,
      Math.sqrt(
        ys.reduce((a, v) => a + (v - ys.reduce((s, w) => s + w, 0) / ys.length) ** 2, 0) /
          ys.length,
      ),
    );
    const gp = new GaussianProcess({
      kernel: 'matern52',
      lengthScale,
      variance: spread * spread * 2,
      noise: Math.max(noise * noise, 1e-6),
      normalizeY: true,
    }).fit(xs, ys);
    const candidates = new Matrix(
      800,
      2,
      Float64Array.from({ length: 1600 }, () => rng.next()),
    );
    const post = gp.posterior(candidates);
    const best = Math.max(...Array.from(gp.posterior(xs).mean));
    let bestEi = -1;
    let pick: [number, number] = [0.5, 0.5];
    for (let c = 0; c < candidates.rows; c++) {
      const ei = expectedImprovement(post.mean[c] as number, post.std[c] as number, best, xi);
      if (ei > bestEi) {
        bestEi = ei;
        pick = [candidates.get(c, 0), candidates.get(c, 1)];
      }
    }
    add(pick);
  }
  // recommend the point the model believes is best among the points it has measured
  const xs = new Matrix(evals.length, 2, Float64Array.from(evals.flatMap((e) => e.x)));
  const gp = new GaussianProcess({
    kernel: 'matern52',
    lengthScale,
    variance: 0.1,
    noise: Math.max(noise * noise, 1e-6),
    normalizeY: true,
  }).fit(
    xs,
    evals.map((e) => e.observed),
  );
  const means = gp.posterior(xs).mean;
  let bi = 0;
  means.forEach((m, i) => {
    if (m > (means[bi] as number)) bi = i;
  });
  return summarise('bayesian', land, evals, () => (evals[bi] as Evaluation).x);
}

export interface HalvingOptions {
  /** Keep the best 1/eta of the configurations at every round. */
  eta?: number;
  /** Fidelity of the first round. */
  minFidelity?: number;
}

/**
 * Successive halving: start many configurations on a cheap, noisy measurement, keep the best third,
 * measure those more carefully, and so on until the survivors get a full measurement.
 */
export function successiveHalving(
  land: Landscape,
  configurations: number,
  noise: number,
  seed = 1,
  options: HalvingOptions = {},
): SearchResult {
  const { eta = 3, minFidelity = 1 / 9 } = options;
  const rng = createRng(seed);
  let alive: [number, number][] = Array.from({ length: configurations }, () => [
    rng.next(),
    rng.next(),
  ]);
  const evals: Evaluation[] = [];
  let fidelity = minFidelity;
  for (;;) {
    const scored = alive.map((x) => {
      const observed = measure(land, x, noise, Math.min(fidelity, 1), rng);
      evals.push({ x, observed, fidelity: Math.min(fidelity, 1) });
      return { x, observed };
    });
    if (fidelity >= 1 - 1e-9 || alive.length <= 1) break;
    scored.sort((a, b) => b.observed - a.observed);
    alive = scored.slice(0, Math.max(1, Math.floor(scored.length / eta))).map((s) => s.x);
    fidelity *= eta;
  }
  return summarise('halving', land, evals);
}

/** The number of configurations successive halving can afford with a given budget (in full experiments). */
export function halvingConfigurations(budget: number, options: HalvingOptions = {}): number {
  const { eta = 3, minFidelity = 1 / 9 } = options;
  let n = 1;
  for (;;) {
    let cost = 0;
    let alive = n;
    let f = minFidelity;
    for (;;) {
      cost += alive * Math.min(f, 1);
      if (f >= 1 - 1e-9 || alive <= 1) break;
      alive = Math.max(1, Math.floor(alive / eta));
      f *= eta;
    }
    if (cost > budget) return Math.max(1, n - 1);
    n++;
    if (n > 5000) return n;
  }
}

export type SearchMethod = 'grid' | 'random' | 'halving' | 'bayesian';

export function runSearch(
  method: SearchMethod,
  land: Landscape,
  budget: number,
  noise: number,
  seed: number,
  options: BayesOptions & HalvingOptions = {},
): SearchResult {
  if (method === 'grid') return gridSearch(land, budget, noise, seed);
  if (method === 'random') return randomSearch(land, budget, noise, seed);
  if (method === 'halving')
    return successiveHalving(land, halvingConfigurations(budget, options), noise, seed, options);
  return bayesianOptimization(land, budget, noise, seed, options);
}
