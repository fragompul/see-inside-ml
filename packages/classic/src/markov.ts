import { Matrix, createRng, solve } from '@sim/core';

/**
 * Markov chains and PageRank. A chain is a row-stochastic matrix `P` (row i = where the walker goes
 * next from state i). Its long-run behaviour is the stationary distribution, the left eigenvector
 * `pi P = pi`. PageRank is the stationary distribution of a "random surfer" who follows a random link
 * most of the time and teleports to a random page otherwise.
 */
export type TransitionMatrix = number[][];

export function isStochastic(p: TransitionMatrix, tolerance = 1e-9): boolean {
  return p.every(
    (row) =>
      row.length === p.length &&
      row.every((v) => v >= -tolerance) &&
      Math.abs(row.reduce((a, b) => a + b, 0) - 1) < tolerance,
  );
}

/** One step of the distribution: `pi' = pi P`. */
export function step(pi: readonly number[], p: TransitionMatrix): number[] {
  const n = pi.length;
  const out = new Array<number>(n).fill(0);
  for (let i = 0; i < n; i++)
    for (let j = 0; j < n; j++)
      out[j] = (out[j] as number) + (pi[i] as number) * (p[i]![j] as number);
  return out;
}

/** The distribution after each step of the iteration, starting from `start` (entry 0 is the start). */
export function evolve(start: readonly number[], p: TransitionMatrix, steps: number): number[][] {
  const out = [Array.from(start)];
  for (let t = 0; t < steps; t++) out.push(step(out[t] as number[], p));
  return out;
}

/**
 * Exact stationary distribution for a chain with one recurrent class: solve `pi (P - I) = 0` together
 * with `sum(pi) = 1` as a linear system.
 */
export function stationary(p: TransitionMatrix): number[] {
  const n = p.length;
  // (P^T - I) pi = 0 with the last equation replaced by sum(pi) = 1
  const a = new Matrix(n, n);
  const b = new Matrix(n, 1);
  for (let i = 0; i < n - 1; i++)
    for (let j = 0; j < n; j++) a.set(i, j, (p[j]![i] as number) - (i === j ? 1 : 0));
  for (let j = 0; j < n; j++) a.set(n - 1, j, 1);
  b.set(n - 1, 0, 1);
  return Array.from(solve(a, b).data);
}

export const totalVariation = (a: readonly number[], b: readonly number[]): number =>
  0.5 * a.reduce((s, v, i) => s + Math.abs(v - (b[i] as number)), 0);

/** Second-largest eigenvalue modulus by deflated power iteration: the per-step rate at which the chain forgets its start. */
export function convergenceRate(p: TransitionMatrix, iterations = 2000): number {
  const n = p.length;
  const pi = stationary(p);
  // iterate a vector orthogonal (in the pi-weighted sense) to the constant right eigenvector
  const rng = createRng(1);
  let v = Array.from({ length: n }, () => rng.normal());
  const project = (x: number[]) => {
    const m = x.reduce((s, xi, i) => s + xi * (pi[i] as number), 0);
    return x.map((xi) => xi - m);
  };
  v = project(v);
  let rate = 0;
  for (let it = 0; it < iterations; it++) {
    const w = new Array<number>(n).fill(0);
    for (let i = 0; i < n; i++)
      for (let j = 0; j < n; j++) w[i] = (w[i] as number) + (p[i]![j] as number) * (v[j] as number);
    const wp = project(w);
    const norm = Math.hypot(...wp);
    const vnorm = Math.hypot(...v);
    if (norm === 0 || vnorm === 0) return 0;
    rate = norm / vnorm;
    v = wp.map((x) => x / norm);
  }
  return rate;
}

/** Simulates the chain and returns the share of time spent in each state. */
export function simulateVisits(
  p: TransitionMatrix,
  start: number,
  steps: number,
  seed = 1,
): number[] {
  const rng = createRng(seed);
  const n = p.length;
  const counts = new Array<number>(n).fill(0);
  let s = start;
  for (let t = 0; t < steps; t++) {
    let u = rng.next();
    let next = n - 1;
    for (let j = 0; j < n; j++) {
      u -= p[s]![j] as number;
      if (u <= 0) {
        next = j;
        break;
      }
    }
    s = next;
    counts[s] = (counts[s] as number) + 1;
  }
  return counts.map((c) => c / steps);
}

// ---- PageRank ----------------------------------------------------------------------------------

/** `links[i]` lists the pages that page i links to. */
export type LinkGraph = number[][];

/**
 * The "Google matrix": with probability `damping` follow a uniformly random outgoing link, otherwise
 * (or if the page has no links at all) jump to a uniformly random page.
 */
export function googleMatrix(links: LinkGraph, damping: number): TransitionMatrix {
  const n = links.length;
  return links.map((out) => {
    const row = new Array<number>(n).fill((1 - damping) / n);
    if (out.length === 0) return row.map(() => 1 / n);
    for (const j of out) row[j] = (row[j] as number) + damping / out.length;
    return row;
  });
}

export function pageRank(links: LinkGraph, damping = 0.85): number[] {
  return stationary(googleMatrix(links, damping));
}

/** Power iteration for PageRank, returning the rank vector after each iteration. */
export function pageRankSteps(links: LinkGraph, damping: number, iterations: number): number[][] {
  const n = links.length;
  return evolve(new Array<number>(n).fill(1 / n), googleMatrix(links, damping), iterations);
}
