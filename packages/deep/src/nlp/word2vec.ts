import { type Rng } from '@sim/core';

/**
 * Skip-gram with negative sampling (word2vec). Every word has two vectors: `input` (what we keep
 * as its embedding) and `output` (its role as context). For a centre word u, a true context word
 * v₊ and K random words v₋:
 *
 *   loss = −log σ(u·v₊) − Σ_k log σ(−u·v₋ₖ)
 *
 * Training moves u towards the vectors of words that really appear near it and away from random
 * ones: words that share contexts end up with similar vectors. Plain SGD, gradients by hand.
 */
export interface Word2VecConfig {
  vocabulary: number;
  dim: number;
}

export const W2V_LIMITS = { dim: [2, 64], negatives: [1, 15], window: [1, 5] } as const;

const sigmoid = (x: number) => 1 / (1 + Math.exp(-x));

export class SkipGram {
  /** `vocabulary × dim`: the embeddings. */
  readonly input: Float64Array;
  /** `vocabulary × dim`: the context vectors. */
  readonly output: Float64Array;

  constructor(
    readonly config: Word2VecConfig,
    rng: Rng,
  ) {
    const { vocabulary, dim } = config;
    if (!Number.isInteger(vocabulary) || vocabulary < 2) throw new RangeError('Need ≥ 2 words');
    const [lo, hi] = W2V_LIMITS.dim;
    if (!Number.isInteger(dim) || dim < lo || dim > hi)
      throw new RangeError(`Dimension must be an integer in [${lo}, ${hi}]`);
    this.input = Float64Array.from({ length: vocabulary * dim }, () => (rng.next() - 0.5) / dim);
    this.output = new Float64Array(vocabulary * dim);
  }

  /**
   * One SGD step over a batch of (centre, context) pairs with `negatives` random words each
   * (`negs` is `pairs × negatives` word ids). Returns the mean loss before the update.
   * Gradients of the whole batch are computed first, then summed and applied with learning rate
   * `lr` per pair (the loss it reports is the batch mean).
   */
  step(
    centers: ArrayLike<number>,
    contexts: ArrayLike<number>,
    negs: ArrayLike<number>,
    negatives: number,
    lr: number,
  ): number {
    const { dim } = this.config;
    const pairs = centers.length;
    const gIn = new Map<number, Float64Array>();
    const gOut = new Map<number, Float64Array>();
    const acc = (m: Map<number, Float64Array>, id: number) => {
      let g = m.get(id);
      if (!g) {
        g = new Float64Array(dim);
        m.set(id, g);
      }
      return g;
    };
    let loss = 0;
    for (let n = 0; n < pairs; n++) {
      const c = centers[n] as number;
      const gc = acc(gIn, c);
      const targets = [contexts[n] as number];
      for (let k = 0; k < negatives; k++) targets.push(negs[n * negatives + k] as number);
      targets.forEach((w, idx) => {
        const label = idx === 0 ? 1 : 0;
        let dot = 0;
        for (let d = 0; d < dim; d++)
          dot += (this.input[c * dim + d] as number) * (this.output[w * dim + d] as number);
        const p = sigmoid(dot);
        loss -= Math.log(Math.max(1e-12, label === 1 ? p : 1 - p));
        const err = p - label; // dL/d(dot)
        const go = acc(gOut, w);
        for (let d = 0; d < dim; d++) {
          gc[d] = (gc[d] as number) + err * (this.output[w * dim + d] as number);
          go[d] = (go[d] as number) + err * (this.input[c * dim + d] as number);
        }
      });
    }
    const scale = lr; // learning rate per pair: the gradients of the batch are summed
    for (const [id, g] of gIn)
      for (let d = 0; d < dim; d++)
        this.input[id * dim + d] = (this.input[id * dim + d] as number) - scale * (g[d] as number);
    for (const [id, g] of gOut)
      for (let d = 0; d < dim; d++)
        this.output[id * dim + d] =
          (this.output[id * dim + d] as number) - scale * (g[d] as number);
    return loss / pairs;
  }

  /** Loss of a batch without changing anything (used by the gradient check). */
  loss(
    centers: ArrayLike<number>,
    contexts: ArrayLike<number>,
    negs: ArrayLike<number>,
    negatives: number,
  ): number {
    const { dim } = this.config;
    let loss = 0;
    for (let n = 0; n < centers.length; n++) {
      const c = centers[n] as number;
      for (let k = -1; k < negatives; k++) {
        const w = (k < 0 ? contexts[n] : negs[n * negatives + k]) as number;
        let dot = 0;
        for (let d = 0; d < dim; d++)
          dot += (this.input[c * dim + d] as number) * (this.output[w * dim + d] as number);
        const p = sigmoid(dot);
        loss -= Math.log(Math.max(1e-12, k < 0 ? p : 1 - p));
      }
    }
    return loss / centers.length;
  }

  vector(id: number): Float64Array {
    const { dim } = this.config;
    return this.input.slice(id * dim, (id + 1) * dim);
  }
}

/** Cosine similarity of two vectors (0 if either is all zeros). */
export function cosine(a: ArrayLike<number>, b: ArrayLike<number>): number {
  let dot = 0;
  let na = 0;
  let nb = 0;
  for (let i = 0; i < a.length; i++) {
    dot += (a[i] as number) * (b[i] as number);
    na += (a[i] as number) ** 2;
    nb += (b[i] as number) ** 2;
  }
  return na > 0 && nb > 0 ? dot / Math.sqrt(na * nb) : 0;
}

/** The `k` rows of `vectors` (`n × dim`) most similar to `query`, skipping `exclude` ids. */
export function nearestWords(
  vectors: Float64Array,
  dim: number,
  query: ArrayLike<number>,
  k: number,
  exclude: readonly number[] = [],
): { id: number; similarity: number }[] {
  const n = vectors.length / dim;
  const skip = new Set(exclude);
  const scored: { id: number; similarity: number }[] = [];
  for (let id = 0; id < n; id++) {
    if (skip.has(id)) continue;
    scored.push({ id, similarity: cosine(vectors.subarray(id * dim, (id + 1) * dim), query) });
  }
  return scored.sort((a, b) => b.similarity - a.similarity).slice(0, k);
}

/** Draws word ids with probability ∝ count^power (the word2vec noise distribution). */
export function noiseSampler(counts: ArrayLike<number>, power = 0.75): (rng: Rng) => number {
  const cumulative = new Float64Array(counts.length);
  let total = 0;
  for (let i = 0; i < counts.length; i++) {
    total += (counts[i] as number) ** power;
    cumulative[i] = total;
  }
  return (rng) => {
    const r = rng.next() * total;
    let lo = 0;
    let hi = cumulative.length - 1;
    while (lo < hi) {
      const mid = (lo + hi) >> 1;
      if ((cumulative[mid] as number) < r) lo = mid + 1;
      else hi = mid;
    }
    return lo;
  };
}
