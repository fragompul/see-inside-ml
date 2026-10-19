import type { ForwardOptions, ForwardTrace, TransformerModel } from './model';

/**
 * Head ablation: silence one attention head at a time and see how much worse the model predicts text. A head whose
 * removal barely matters is redundant for this text; one whose removal ruins it is doing real work.
 */
export const headKey = (layer: number, head: number): string => `${layer}.${head}`;

/** Mean cross-entropy (nats) of predicting each token of `ids` from the ones before it. */
export function sequenceLoss(
  model: TransformerModel,
  ids: readonly number[],
  options: ForwardOptions = {},
): number {
  const { logits } = model.forward(ids, options);
  let total = 0;
  for (let t = 0; t + 1 < ids.length; t++) {
    let top = -Infinity;
    for (let v = 0; v < logits.cols; v++) top = Math.max(top, logits.get(t, v));
    let z = 0;
    for (let v = 0; v < logits.cols; v++) z += Math.exp(logits.get(t, v) - top);
    total += top + Math.log(z) - logits.get(t, ids[t + 1] as number);
  }
  return total / (ids.length - 1);
}

/** Average loss over several texts, weighting every predicted token equally. */
export function corpusLoss(
  model: TransformerModel,
  texts: readonly (readonly number[])[],
  options: ForwardOptions = {},
): number {
  let total = 0;
  let count = 0;
  for (const ids of texts) {
    total += sequenceLoss(model, ids, options) * (ids.length - 1);
    count += ids.length - 1;
  }
  return total / count;
}

export interface HeadPattern {
  /** Average attention weight on the previous token (position i looking at i - 1). */
  previous: number;
  /** Average attention weight on the very first token. */
  first: number;
  /** Average attention weight a token pays to itself. */
  self: number;
  /** Average entropy of the attention distribution (nats): high means spread over many tokens. */
  entropy: number;
}

/** Summaries of what each head attends to, averaged over the positions of the given traces. */
export function headPatterns(traces: readonly ForwardTrace[]): HeadPattern[][] {
  const layers = (traces[0] as ForwardTrace).layers.length;
  const heads = (traces[0] as ForwardTrace).layers[0]?.probs.length as number;
  const out: HeadPattern[][] = [];
  for (let l = 0; l < layers; l++) {
    const row: HeadPattern[] = [];
    for (let h = 0; h < heads; h++) {
      let previous = 0;
      let first = 0;
      let self = 0;
      let entropy = 0;
      let positions = 0;
      for (const trace of traces) {
        const p = (trace.layers[l] as ForwardTrace['layers'][number]).probs[h]!;
        for (let i = 1; i < p.rows; i++) {
          previous += p.get(i, i - 1);
          first += p.get(i, 0);
          self += p.get(i, i);
          for (let j = 0; j <= i; j++) {
            const w = p.get(i, j);
            if (w > 0) entropy -= w * Math.log(w);
          }
          positions++;
        }
      }
      row.push({
        previous: previous / positions,
        first: first / positions,
        self: self / positions,
        entropy: entropy / positions,
      });
    }
    out.push(row);
  }
  return out;
}

export interface AblationResult {
  baseline: number;
  /** Increase in loss when only this head is silenced, per layer and head. */
  delta: number[][];
  /** Loss as heads are removed in order of how much they matter (first entry: nothing removed). */
  cumulative: { removed: string[]; loss: number }[];
}

export function ablateHeads(
  model: TransformerModel,
  texts: readonly (readonly number[])[],
  mode: 'zero' | 'mean' = 'zero',
): AblationResult {
  const { layers, heads } = model.config;
  const baseline = corpusLoss(model, texts);
  const delta = Array.from({ length: layers }, (_, l) =>
    Array.from(
      { length: heads },
      (_, h) =>
        corpusLoss(model, texts, { ablate: new Set([headKey(l, h)]), ablateMode: mode }) - baseline,
    ),
  );
  const order = delta
    .flatMap((row, l) => row.map((d, h) => ({ key: headKey(l, h), d })))
    .sort((a, b) => a.d - b.d);
  // remove the least important first, to show how many heads the model can lose
  const cumulative = [{ removed: [] as string[], loss: baseline }];
  const removed: string[] = [];
  for (const { key } of order) {
    removed.push(key);
    cumulative.push({
      removed: [...removed],
      loss: corpusLoss(model, texts, { ablate: new Set(removed), ablateMode: mode }),
    });
  }
  return { baseline, delta, cumulative };
}
