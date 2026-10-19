import { describe, expect, it } from 'vitest';
import { VOCABULARY } from './corpus';
import {
  DEFAULT_NLP_LAB,
  DIVERGED_LOSS,
  NLP_RECORD_EVERY,
  NlpSession,
  validateNlpLab,
  wordId,
} from './session';

const make = (patch = {}) => new NlpSession({ ...DEFAULT_NLP_LAB, sentences: 400, ...patch });

describe('NlpSession', () => {
  it('trains deterministically for a seed and records statistics every NLP_RECORD_EVERY steps', () => {
    const a = make();
    const b = make();
    expect(a.train(60)).toHaveLength(2);
    b.train(25);
    b.train(35);
    expect(a.history).toEqual(b.history);
    expect(a.history.map((h) => h.step)).toEqual([NLP_RECORD_EVERY, 2 * NLP_RECORD_EVERY]);
    expect(
      a.history.every((h) => h.loss > 0 && h.analogy >= 0 && h.purity >= 0 && h.purity <= 1),
    ).toBe(true);
  });

  it('the loss falls and words of one category end up near each other', () => {
    const s = make({ sentences: 1000 });
    const before = s.neighbourPurity();
    s.train(1500);
    const first = s.history[0]?.loss as number;
    const last = s.history[s.history.length - 1]?.loss as number;
    expect(last).toBeLessThan(first);
    expect(s.neighbourPurity()).toBeGreaterThan(before + 0.2);
    expect(s.neighbourPurity()).toBeGreaterThan(0.85);
    expect(['dog', 'horse', 'fox', 'rabbit', 'bird']).toContain(s.nearest('cat', 1)[0]?.word);
  });

  it('counts pairs and epochs from the window and the corpus', () => {
    const narrow = make({ window: 1 });
    const wide = make({ window: 4 });
    expect(wide.pairCount).toBeGreaterThan(narrow.pairCount * 2);
    expect(narrow.epoch).toBe(0);
    narrow.train(Math.ceil(narrow.pairCount / narrow.config.batchSize) + 1);
    expect(narrow.epoch).toBe(1);
  });

  it('applies learning settings live and rebuilds on data or shape changes', () => {
    const s = make();
    s.train(30);
    expect(s.update({ lr: 0.01, negatives: 3, batchSize: 32 })).toBe('live');
    expect(s.history).toHaveLength(1);
    expect(s.update({ dim: 8 })).toBe('rebuilt');
    expect(s.history).toHaveLength(0);
    expect(s.vector('cat')).toHaveLength(8);
    expect(s.update({ window: 2 })).toBe('rebuilt');
    expect(s.update({ sentences: 600 })).toBe('rebuilt');
    expect(s.step).toBe(0);
  });

  it('rejects invalid settings and keeps the previous ones', () => {
    const s = make();
    expect(() => s.update({ dim: 1 })).toThrow();
    expect(() => s.update({ window: 9 })).toThrow();
    expect(() => s.update({ negatives: 0 })).toThrow();
    expect(() => s.update({ lr: 0 })).toThrow();
    expect(() => s.update({ sentences: 5 })).toThrow();
    expect(() => s.update({ batchSize: 2 })).toThrow();
    expect(() => validateNlpLab({ ...DEFAULT_NLP_LAB, dim: 2.5 })).toThrow();
    expect(s.config.dim).toBe(DEFAULT_NLP_LAB.dim);
  });

  it('stops cleanly when the learning rate is far too high, and a reset recovers', () => {
    const s = make({ lr: 0.5 });
    expect(s.diverged).toBe(false);
    s.train(2000);
    expect(s.diverged).toBe(true);
    expect(s.step).toBeLessThan(2000);
    const frozen = s.step;
    s.train(50);
    expect(s.step).toBe(frozen); // nothing more happens
    expect(s.vectors().every(Number.isFinite)).toBe(true);
    s.update({ lr: 0.05 });
    s.reset();
    expect(s.diverged).toBe(false);
    s.train(10);
    expect(s.step).toBe(10);
    expect(DIVERGED_LOSS).toBeGreaterThan(10);
  });

  it('nearest and analogy exclude their inputs and know every word', () => {
    const s = make();
    const near = s.nearest('king', 5);
    expect(near).toHaveLength(5);
    expect(near.map((n) => n.word)).not.toContain('king');
    expect(near[0]!.similarity).toBeGreaterThanOrEqual(near[4]!.similarity);
    const answer = s.analogy('man', 'king', 'woman', 3);
    expect(answer.map((n) => n.word)).not.toContain('man');
    expect(answer.map((n) => n.word)).not.toContain('woman');
    expect(() => s.nearest('banana')).toThrow(RangeError);
    expect(s.similarity('cat', 'cat')).toBeCloseTo(1, 12);
    expect(wordId(VOCABULARY[3] as string)).toBe(3);
  });

  it('the projection covers the content words and explains positive variance', () => {
    const s = make();
    s.train(50);
    const p = s.projection();
    expect(p.words).not.toContain('the');
    expect(p.words).toContain('king');
    expect(p.coords).toHaveLength(p.words.length * 2);
    expect(p.coords.every(Number.isFinite)).toBe(true);
    expect(p.explained[0]!).toBeGreaterThan(0);
  });

  it('inspecting the model does not disturb training', () => {
    const a = make();
    const b = make();
    a.train(30);
    b.train(30);
    a.nearest('cat');
    a.analogy('france', 'paris', 'italy');
    a.projection();
    a.analogyAccuracy();
    a.neighbourPurity();
    a.train(30);
    b.train(30);
    expect(a.vectors()).toEqual(b.vectors());
  });

  it('accuracy by group reports every group', () => {
    const groups = make().analogyAccuracyByGroup();
    expect(groups.map((g) => g.id)).toEqual(['gender', 'capital', 'royalty']);
    groups.forEach((g) => expect(g.accuracy >= 0 && g.accuracy <= 1).toBe(true));
  });
});
