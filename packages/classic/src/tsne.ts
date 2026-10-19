import { Matrix, createRng } from '@sim/core';
import { squaredDistances } from './manifold';
import { PCA } from './pca';

/**
 * t-SNE (exact, O(n²)). Turn distances into the probability that each point would pick each other
 * point as a neighbour (a Gaussian whose width is tuned so every point has the same *perplexity*,
 * an effective number of neighbours), then arrange 2-D points so that the same probabilities, now
 * computed with a heavy-tailed Student-t kernel, match as closely as possible (minimise the KL
 * divergence by gradient descent). Conventions, constants and the optimiser follow scikit-learn.
 */
const EPS = 2.220446049250313e-16;

/** Conditional probabilities by binary search on the precision of each Gaussian (scikit-learn's algorithm). */
export function conditionalProbabilities(
  d2: Float64Array,
  n: number,
  perplexity: number,
): Float64Array {
  const p = new Float64Array(n * n);
  const desired = Math.log(perplexity);
  for (let i = 0; i < n; i++) {
    let betaMin = -Infinity;
    let betaMax = Infinity;
    let beta = 1;
    for (let step = 0; step < 100; step++) {
      let sum = 0;
      for (let j = 0; j < n; j++) {
        if (j === i) continue;
        const v = Math.exp(-(d2[i * n + j] as number) * beta);
        p[i * n + j] = v;
        sum += v;
      }
      if (sum === 0) sum = 1e-8;
      let weighted = 0;
      for (let j = 0; j < n; j++) {
        if (j === i) continue;
        p[i * n + j] = (p[i * n + j] as number) / sum;
        weighted += (d2[i * n + j] as number) * (p[i * n + j] as number);
      }
      const entropy = Math.log(sum) + beta * weighted;
      const diff = entropy - desired;
      if (Math.abs(diff) <= 1e-5) break;
      if (diff > 0) {
        betaMin = beta;
        beta = betaMax === Infinity ? beta * 2 : (beta + betaMax) / 2;
      } else {
        betaMax = beta;
        beta = betaMin === -Infinity ? beta / 2 : (beta + betaMin) / 2;
      }
    }
  }
  return p;
}

/** The symmetric joint probabilities P (flat `n × n`, summing to 1, zero diagonal). */
export function jointProbabilities(x: Matrix, perplexity: number): Float64Array {
  const n = x.rows;
  const cond = conditionalProbabilities(squaredDistances(x), n, perplexity);
  const p = new Float64Array(n * n);
  let sum = 0;
  for (let i = 0; i < n; i++)
    for (let j = 0; j < n; j++) {
      if (i === j) continue;
      p[i * n + j] = (cond[i * n + j] as number) + (cond[j * n + i] as number);
      sum += p[i * n + j] as number;
    }
  sum = Math.max(sum, EPS);
  return p.map((v, idx) => (idx % (n + 1) === 0 ? 0 : Math.max(v / sum, EPS)));
}

/** KL(P ‖ Q) and its gradient with respect to the embedding `y` (`n × d`), Student-t kernel with one degree of freedom. */
export function klAndGradient(p: Float64Array, y: Matrix): { kl: number; grad: Float64Array } {
  const n = y.rows;
  const d = y.cols;
  const w = new Float64Array(n * n); // (1 + ‖yi − yj‖²)⁻¹
  let sum = 0;
  for (let i = 0; i < n; i++)
    for (let j = i + 1; j < n; j++) {
      let s = 0;
      for (let c = 0; c < d; c++) s += (y.get(i, c) - y.get(j, c)) ** 2;
      const v = 1 / (1 + s);
      w[i * n + j] = w[j * n + i] = v;
      sum += 2 * v;
    }
  let kl = 0;
  const grad = new Float64Array(n * d);
  for (let i = 0; i < n; i++)
    for (let j = 0; j < n; j++) {
      if (i === j) continue;
      const pij = p[i * n + j] as number;
      const q = Math.max((w[i * n + j] as number) / sum, EPS);
      kl += pij * Math.log(Math.max(pij, EPS) / q);
      const coeff = 4 * (pij - q) * (w[i * n + j] as number);
      for (let c = 0; c < d; c++)
        grad[i * d + c] = (grad[i * d + c] as number) + coeff * (y.get(i, c) - y.get(j, c));
    }
  return { kl, grad };
}

export interface TsneOptions {
  perplexity?: number;
  /** Length of the early-exaggeration phase, in iterations (scikit-learn: 250). */
  earlyIterations?: number;
  earlyExaggeration?: number;
  /** Total iterations (scikit-learn: 1000). */
  iterations?: number;
  /** Learning rate; default `max(n / earlyExaggeration / 4, 50)`. */
  learningRate?: number;
  init?: 'pca' | 'random' | Matrix;
  seed?: number;
}

export interface TsneState {
  iteration: number;
  embedding: Matrix;
  /** KL divergence at this iteration (measured against the exaggerated P during the early phase). */
  kl: number;
  exaggerating: boolean;
}

/** The optimisation as a generator, one state per iteration (state 0 is the start). */
export function* tsneSteps(
  x: Matrix,
  options: TsneOptions = {},
): Generator<TsneState, void, undefined> {
  const n = x.rows;
  const {
    perplexity = Math.min(30, Math.max(2, (n - 1) / 3)),
    earlyIterations = 250,
    earlyExaggeration = 12,
    iterations = 1000,
    seed = 1,
  } = options;
  const lr = options.learningRate ?? Math.max(n / earlyExaggeration / 4, 50);
  const pBase = jointProbabilities(x, perplexity);
  let y: Matrix;
  const init = options.init ?? 'pca';
  if (init instanceof Matrix) y = new Matrix(n, init.cols, Float64Array.from(init.data));
  else if (init === 'pca') {
    const z = new PCA({ components: 2 }).fit(x).transform(x);
    let mean = 0;
    for (let i = 0; i < n; i++) mean += z.get(i, 0);
    mean /= n;
    let sd = 0;
    for (let i = 0; i < n; i++) sd += (z.get(i, 0) - mean) ** 2;
    sd = Math.sqrt(sd / n) || 1;
    y = new Matrix(
      n,
      2,
      z.data.map((v) => (v / sd) * 1e-4),
    );
  } else {
    const rng = createRng(seed);
    y = new Matrix(
      n,
      2,
      Float64Array.from({ length: n * 2 }, () => rng.normal() * 1e-4),
    );
  }
  const update = new Float64Array(y.data.length);
  const gains = new Float64Array(y.data.length).fill(1);
  yield {
    iteration: 0,
    embedding: y,
    kl: klAndGradient(
      pBase.map((v) => v * earlyExaggeration),
      y,
    ).kl,
    exaggerating: true,
  };
  for (let it = 1; it <= iterations; it++) {
    const early = it <= earlyIterations;
    const p = early ? pBase.map((v) => v * earlyExaggeration) : pBase;
    const { kl, grad } = klAndGradient(p, y);
    const momentum = early ? 0.5 : 0.8;
    const next = Float64Array.from(y.data);
    for (let i = 0; i < next.length; i++) {
      const same = (update[i] as number) * (grad[i] as number) < 0;
      gains[i] = Math.max(0.01, same ? (gains[i] as number) + 0.2 : (gains[i] as number) * 0.8);
      update[i] =
        momentum * (update[i] as number) - lr * (gains[i] as number) * (grad[i] as number);
      next[i] = (next[i] as number) + (update[i] as number);
    }
    y = new Matrix(n, y.cols, next);
    yield { iteration: it, embedding: y, kl, exaggerating: early };
  }
}

/** Runs t-SNE to the end and returns the final embedding. */
export function tsne(x: Matrix, options: TsneOptions = {}): { embedding: Matrix; kl: number } {
  let last: TsneState | null = null;
  for (const s of tsneSteps(x, options)) last = s;
  const state = last as TsneState;
  return { embedding: state.embedding, kl: state.kl };
}
