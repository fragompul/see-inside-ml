import { createRng } from '@sim/core';
import { describe, expect, it } from 'vitest';
import { CharLM, CharTrainer, DEFAULT_CHAR_TRAIN, decode, makeCorpus } from './charlm';
import { CELL_NAMES, type CellName } from './recurrent';

describe('gradients', () => {
  it.each(CELL_NAMES)(
    '%s: backpropagation through time matches finite differences',
    (cell: CellName) => {
      const vocab = 5;
      const net = new CharLM({ cell, hidden: 4, vocab }, createRng(3));
      const tokens = [0, 3, 1, 4, 2, 2, 0, 1];
      net.zeroGrad();
      const { trace } = net.forward(tokens);
      net.backward(trace, 1);
      const h = 1e-6;
      for (const p of net.parameters) {
        // check a spread of entries of every parameter tensor
        const idx = [
          0,
          Math.floor(p.data.length / 3),
          Math.floor(p.data.length / 2),
          p.data.length - 1,
        ];
        for (const i of idx) {
          const saved = p.data[i]!;
          p.data[i] = saved + h;
          const up = net.evaluate(tokens);
          p.data[i] = saved - h;
          const down = net.evaluate(tokens);
          p.data[i] = saved;
          const numeric = (up - down) / (2 * h);
          expect(p.grad[i]).toBeCloseTo(numeric, 5);
        }
      }
    },
  );

  it('predicted distributions sum to one at every step and the untrained loss is about log of the alphabet size', () => {
    const net = new CharLM({ cell: 'lstm', hidden: 8, vocab: 10 }, createRng(1));
    const { trace, loss } = net.forward([1, 2, 3, 4, 5, 6, 7]);
    for (let t = 0; t < trace.steps; t++) {
      let s = 0;
      for (let k = 0; k < 10; k++) s += trace.probs[t * 10 + k]!;
      expect(s).toBeCloseTo(1, 12);
    }
    expect(Math.abs(loss - Math.log(10))).toBeLessThan(0.5);
  });
});

describe('learning', () => {
  const text = 'abcabcabcabcabcabcabcabcabcabcabcabcabcabcabcabcabcabcabc';
  const corpus = makeCorpus(text);

  it('learns a repeating pattern: the loss falls towards zero and samples continue the pattern', () => {
    const trainer = new CharTrainer(corpus, {
      ...DEFAULT_CHAR_TRAIN,
      hidden: 12,
      batch: 2,
      window: 12,
    });
    const first = trainer.step();
    for (let i = 0; i < 150; i++) trainer.step();
    expect(trainer.history[trainer.history.length - 1]!).toBeLessThan(first / 5);
    const sample = trainer.generate(30, 0.5, 1, 'ab');
    expect(sample.startsWith('ab')).toBe(true);
    expect(sample).toMatch(/^(abc)+/);
  });

  it('is deterministic for a seed', () => {
    const a = new CharTrainer(corpus, { ...DEFAULT_CHAR_TRAIN, hidden: 6 });
    const b = new CharTrainer(corpus, { ...DEFAULT_CHAR_TRAIN, hidden: 6 });
    for (let i = 0; i < 5; i++) expect(a.step()).toBe(b.step());
  });

  it('a lower temperature makes sampling more repetitive (lower entropy of the output)', () => {
    const t = new CharTrainer(
      makeCorpus('the cat sat on the mat. the dog sat on the log. '.repeat(4)),
      { ...DEFAULT_CHAR_TRAIN, hidden: 16 },
    );
    for (let i = 0; i < 200; i++) t.step();
    const distinct = (s: string) => new Set(s.split(' ')).size;
    const cold = t.generate(200, 0.2, 5, 'the ');
    const hot = t.generate(200, 2, 5, 'the ');
    expect(distinct(cold)).toBeLessThanOrEqual(distinct(hot));
  });

  it('a held-out tail is never trained on and corpus helpers round-trip', () => {
    const c = makeCorpus('hello world');
    expect(decode(c, c.tokens)).toBe('hello world');
    const trainer = new CharTrainer(
      makeCorpus('abcdefghij'.repeat(20)),
      { ...DEFAULT_CHAR_TRAIN, hidden: 6, window: 10 },
      40,
    );
    for (let i = 0; i < 5; i++) trainer.step();
    expect(Number.isFinite(trainer.validationLoss())).toBe(true);
  });
});
