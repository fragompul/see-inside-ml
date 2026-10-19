import { describe, expect, it } from 'vitest';
import { crossEntropy, perplexity, trainNgram, type Smoothing } from './ngram';

const text = Int32Array.from(
  [...'abracadabra abracadabra alakazam'].map((c) => c.charCodeAt(0) % 32),
);
const vocab = 32;

describe('n-gram language models', () => {
  it.each(['addk', 'interpolated'] as Smoothing[])(
    '%s: probabilities sum to one for seen and unseen contexts',
    (smoothing) => {
      const m = trainNgram(text, vocab, 3, smoothing);
      for (const context of [
        [1, 2],
        [5, 9],
        [31, 31],
      ]) {
        let s = 0;
        for (let v = 0; v < vocab; v++) s += m.prob(context, 2, v);
        expect(s).toBeCloseTo(1, 10);
      }
    },
  );

  it('the unigram model has the empirical entropy of the text', () => {
    const m = trainNgram(text, vocab, 1, 'none');
    const counts = new Map<number, number>();
    for (let i = 1; i < text.length; i++)
      counts.set(text[i] as number, (counts.get(text[i] as number) ?? 0) + 1);
    const n = text.length - 1;
    let h = 0;
    for (const c of counts.values()) h -= (c / n) * Math.log(c / n);
    // the model counts the first character too, so allow a small difference
    expect(crossEntropy(m, text)).toBeCloseTo(h, 1);
  });

  it('a huge pseudo-count gives the uniform model: perplexity equals the alphabet size', () => {
    const m = trainNgram(text, vocab, 3, 'addk', { k: 1e9, lambda: 0.7 });
    expect(perplexity(crossEntropy(m, text))).toBeCloseTo(vocab, 3);
  });

  it('longer contexts fit the training text better, and unseen text gets probability zero without smoothing', () => {
    const fits = [1, 2, 3, 5].map((o) => crossEntropy(trainNgram(text, vocab, o, 'none'), text));
    for (let i = 1; i < fits.length; i++) expect(fits[i]!).toBeLessThanOrEqual(fits[i - 1]! + 1e-9);
    const other = Int32Array.from([...'zzzzzz'].map((c) => c.charCodeAt(0) % 32));
    expect(crossEntropy(trainNgram(text, vocab, 2, 'none'), other)).toBe(Infinity);
    expect(Number.isFinite(crossEntropy(trainNgram(text, vocab, 2, 'addk'), other))).toBe(true);
  });
});
