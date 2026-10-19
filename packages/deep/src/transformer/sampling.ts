import { type Rng } from '@sim/core';

export interface SamplingOptions {
  /** > 0. Low = confident and repetitive, high = random. 0 means greedy decoding. */
  temperature: number;
  /** Keep only the k most likely tokens (0 or ≥ vocab = off). */
  topK: number;
  /** Nucleus sampling: keep the smallest set whose probability mass reaches p (1 = off). */
  topP: number;
}

export const DEFAULT_SAMPLING: SamplingOptions = { temperature: 1, topK: 0, topP: 1 };

export interface Candidate {
  id: number;
  logit: number;
  /** Probability after temperature, over the whole vocabulary. */
  prob: number;
  /** Cumulative probability in descending order (including this token). */
  cumulative: number;
  /** Survived top-k and top-p. */
  kept: boolean;
  /** Probability after filtering and renormalising (0 when dropped). This is what is sampled. */
  adjusted: number;
}

export interface NextTokenDistribution {
  /** Sorted by probability, most likely first (ties by id). */
  candidates: Candidate[];
  keptCount: number;
  /** Entropy in bits of the temperature-scaled distribution: how "undecided" the model is. */
  entropyBits: number;
}

/** Numerically stable softmax with temperature. temperature ≤ 0 returns a one-hot argmax. */
export function softmax(logits: ArrayLike<number>, temperature = 1): Float64Array {
  const n = logits.length;
  const out = new Float64Array(n);
  if (temperature <= 0) {
    let best = 0;
    for (let i = 1; i < n; i++) if ((logits[i] as number) > (logits[best] as number)) best = i;
    out[best] = 1;
    return out;
  }
  let max = -Infinity;
  for (let i = 0; i < n; i++) max = Math.max(max, logits[i] as number);
  let sum = 0;
  for (let i = 0; i < n; i++) {
    const e = Math.exp(((logits[i] as number) - max) / temperature);
    out[i] = e;
    sum += e;
  }
  for (let i = 0; i < n; i++) out[i] = (out[i] as number) / sum;
  return out;
}

/**
 * Turns raw logits into the distribution the sampler really draws from, keeping every stage
 * (temperature → top-k → top-p → renormalise) inspectable.
 */
export function nextTokenDistribution(
  logits: ArrayLike<number>,
  options: SamplingOptions = DEFAULT_SAMPLING,
): NextTokenDistribution {
  const probs = softmax(logits, options.temperature);
  const order = Array.from({ length: probs.length }, (_, i) => i).sort(
    (a, b) => (probs[b] as number) - (probs[a] as number) || a - b,
  );

  const topK = options.topK > 0 ? Math.min(options.topK, order.length) : order.length;
  const candidates: Candidate[] = [];
  let cumulative = 0;
  let kept = 0;
  let massBefore = 0;
  for (let rank = 0; rank < order.length; rank++) {
    const id = order[rank] as number;
    const prob = probs[id] as number;
    cumulative += prob;
    // Nucleus: keep a token while the mass *before* it is still below p (so at least one survives).
    const inNucleus = options.topP >= 1 || massBefore < options.topP || rank === 0;
    const keep = rank < topK && inNucleus;
    massBefore += prob;
    if (keep) kept += 1;
    candidates.push({ id, logit: logits[id] as number, prob, cumulative, kept: keep, adjusted: 0 });
  }
  let keptMass = 0;
  for (const c of candidates) if (c.kept) keptMass += c.prob;
  for (const c of candidates) c.adjusted = c.kept ? c.prob / keptMass : 0;

  let entropy = 0;
  for (const c of candidates) if (c.prob > 0) entropy -= c.prob * Math.log2(c.prob);
  return { candidates, keptCount: kept, entropyBits: entropy };
}

/** Draws one token id from the filtered distribution. */
export function sampleToken(distribution: NextTokenDistribution, rng: Rng): number {
  const r = rng.next();
  let acc = 0;
  let last = distribution.candidates[0]?.id ?? 0;
  for (const c of distribution.candidates) {
    if (c.adjusted <= 0) continue;
    acc += c.adjusted;
    last = c.id;
    if (r < acc) return c.id;
  }
  return last; // floating-point slack: fall back to the last kept token
}

export function validateSampling(options: SamplingOptions): void {
  if (!(options.temperature >= 0) || !Number.isFinite(options.temperature)) {
    throw new RangeError('temperature must be a finite number ≥ 0');
  }
  if (!Number.isInteger(options.topK) || options.topK < 0)
    throw new RangeError('topK must be an integer ≥ 0');
  if (!(options.topP > 0 && options.topP <= 1)) throw new RangeError('topP must be in (0, 1]');
}
