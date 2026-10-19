import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import { ablateHeads, corpusLoss, headKey, headPatterns, sequenceLoss } from './ablation';
import { createTransformer } from './generate';
import type { TransformerManifest } from './model';
import type { TokenizerData } from './tokenizer';

const asciiText = (bytes: Uint8Array): string => {
  let out = '';
  for (const b of bytes) out += String.fromCharCode(b);
  return out;
};
const dir = new URL('../../models/tiny-gpt/', import.meta.url);
const raw = readFileSync(new URL('weights.bin', dir));
const { model, tokenizer } = createTransformer(
  JSON.parse(asciiText(readFileSync(new URL('model.json', dir)))) as TransformerManifest,
  raw.buffer.slice(raw.byteOffset, raw.byteOffset + raw.byteLength),
  JSON.parse(asciiText(readFileSync(new URL('tokenizer.json', dir)))) as TokenizerData,
);
const ids = tokenizer.encode('Anna found a ball. She took it to the park.');
const hd = model.headDim;

describe('head ablation', () => {
  it('silencing nothing changes nothing', () => {
    const a = model.forward(ids).logits.data;
    const b = model.forward(ids, { ablate: new Set() }).logits.data;
    expect(Array.from(b)).toEqual(Array.from(a));
  });

  it('a zeroed head contributes zeros to its own columns and leaves the others alone', () => {
    const clean = model.forward(ids).layers[0]!;
    const cut = model.forward(ids, { ablate: new Set([headKey(0, 2)]) }).layers[0]!;
    for (let t = 0; t < ids.length; t++)
      for (let c = 0; c < model.config.d; c++) {
        const inHead = c >= 2 * hd && c < 3 * hd;
        expect(cut.headsOut.get(t, c)).toBe(inHead ? 0 : clean.headsOut.get(t, c));
      }
  });

  it('with every head silenced the attention output is just the projection bias, the same at every position', () => {
    const all = new Set(
      Array.from({ length: model.config.layers * model.config.heads }, (_, i) =>
        headKey(Math.floor(i / model.config.heads), i % model.config.heads),
      ),
    );
    const trace = model.forward(ids, { ablate: all });
    const first = Array.from({ length: model.config.d }, (_, c) =>
      trace.layers[0]!.attnOut.get(0, c),
    );
    for (let t = 1; t < ids.length; t++)
      for (let c = 0; c < model.config.d; c++)
        expect(trace.layers[0]!.attnOut.get(t, c)).toBeCloseTo(first[c]!, 12);
  });

  it('mean ablation gives every position the same output for that head', () => {
    const trace = model.forward(ids, { ablate: new Set([headKey(1, 0)]), ablateMode: 'mean' });
    for (let t = 1; t < ids.length; t++)
      for (let c = 0; c < hd; c++)
        expect(trace.layers[1]!.headsOut.get(t, c)).toBeCloseTo(
          trace.layers[1]!.headsOut.get(0, c),
          12,
        );
  });

  it('the loss is the average of -log p of each next token, and cannot be below zero', () => {
    const loss = sequenceLoss(model, ids);
    const { logits } = model.forward(ids);
    let manual = 0;
    for (let t = 0; t + 1 < ids.length; t++) {
      const row = Array.from({ length: logits.cols }, (_, v) => logits.get(t, v));
      const z = row.reduce((a, v) => a + Math.exp(v), 0);
      manual -= Math.log(Math.exp(row[ids[t + 1]!]!) / z);
    }
    expect(loss).toBeCloseTo(manual / (ids.length - 1), 10);
    expect(loss).toBeGreaterThan(0);
    expect(corpusLoss(model, [ids, ids])).toBeCloseTo(loss, 12);
  });

  it('attention summaries are averages of probabilities, and the removal curve starts at the baseline', () => {
    const patterns = headPatterns([model.forward(ids)]);
    for (const row of patterns)
      for (const p of row) {
        for (const share of [p.previous, p.first, p.self]) {
          expect(share).toBeGreaterThanOrEqual(0);
          expect(share).toBeLessThanOrEqual(1);
        }
        expect(p.entropy).toBeGreaterThan(0);
      }
    const result = ablateHeads(model, [ids]);
    expect((result.cumulative[0] as { loss: number }).loss).toBe(result.baseline);
    expect(result.cumulative).toHaveLength(model.config.layers * model.config.heads + 1);
  });
});
