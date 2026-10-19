/**
 * Character n-gram language models: predict the next character from the previous n - 1 by counting.
 * `none` is the raw frequency (zero for anything unseen), `addk` adds k pseudo-counts to every character,
 * and `interpolated` builds the estimate up from the uniform guess through ever longer contexts, giving the longest one weight lambda.
 */
export type Smoothing = 'none' | 'addk' | 'interpolated';

export interface NgramModel {
  order: number;
  vocab: number;
  smoothing: Smoothing;
  /** Probability of `next` after the given context (at most order - 1 earlier tokens are used). */
  prob: (context: ArrayLike<number>, end: number, next: number) => number;
}

export interface NgramOptions {
  /** Pseudo-count for `addk`. */
  k: number;
  /** Weight on the longest context for `interpolated`; the rest is passed down to shorter ones. */
  lambda: number;
}

export const DEFAULT_NGRAM: NgramOptions = { k: 0.1, lambda: 0.7 };

const keyOf = (tokens: ArrayLike<number>, from: number, to: number): string => {
  let key = '';
  for (let i = from; i < to; i++) key += `${tokens[i]},`;
  return key;
};

export function trainNgram(
  tokens: ArrayLike<number>,
  vocab: number,
  order: number,
  smoothing: Smoothing,
  options: NgramOptions = DEFAULT_NGRAM,
): NgramModel {
  // counts[len] maps a context of `len` tokens to the counts of what followed it
  const counts: Map<string, Float64Array>[] = Array.from({ length: order }, () => new Map());
  for (let i = 0; i < tokens.length; i++)
    for (let len = 0; len < order && len <= i; len++) {
      const key = keyOf(tokens, i - len, i);
      let row = counts[len]?.get(key);
      if (!row) {
        row = new Float64Array(vocab);
        counts[len]?.set(key, row);
      }
      row[tokens[i] as number] = (row[tokens[i] as number] as number) + 1;
    }

  const raw = (
    len: number,
    context: ArrayLike<number>,
    end: number,
    next: number,
  ): number | null => {
    const row = counts[len]?.get(keyOf(context, end - len, end));
    if (!row) return null;
    let total = 0;
    for (let v = 0; v < vocab; v++) total += row[v] as number;
    return (row[next] as number) / total;
  };

  const prob = (context: ArrayLike<number>, end: number, next: number): number => {
    const longest = Math.min(order - 1, end);
    if (smoothing === 'none') return raw(longest, context, end, next) ?? 0;
    if (smoothing === 'addk') {
      const row = counts[longest]?.get(keyOf(context, end - longest, end));
      const total = row ? row.reduce((a, b) => a + b, 0) : 0;
      return (((row?.[next] as number | undefined) ?? 0) + options.k) / (total + options.k * vocab);
    }
    // interpolated: start from the uniform guess and refine it with ever longer contexts
    let p = 1 / vocab;
    for (let len = 0; len <= longest; len++) {
      const r = raw(len, context, end, next);
      if (r === null) break;
      p = len === longest ? options.lambda * r + (1 - options.lambda) * p : 0.3 * r + 0.7 * p;
    }
    return p;
  };
  return { order, vocab, smoothing, prob };
}

/** Average negative log-probability (nats per character) of the text under the model. */
export function crossEntropy(model: NgramModel, tokens: ArrayLike<number>): number {
  let sum = 0;
  let n = 0;
  for (let i = 1; i < tokens.length; i++) {
    sum -= Math.log(model.prob(tokens, i, tokens[i] as number));
    n += 1;
  }
  return sum / n;
}

export const perplexity = (nats: number): number => Math.exp(nats);
