import { describe, expect, it } from 'vitest';
import { DEFAULT_CHARLM, grammarValidity, runCharLm, wordValidity } from './charlmCore';
import { CORPORA, corpusOf } from './charlmData';
import { charlmExperiments } from './charlmSpec';

describe('character language-model lab', () => {
  for (const exp of charlmExperiments) {
    it(`experiment "${exp.title}" holds`, () => {
      expect(exp.claim(runCharLm({ ...DEFAULT_CHARLM, ...exp.scenario }))).toBe(true);
    }, 300_000);
  }

  it('the corpora are non-trivial and the grammar corpus obeys its own grammar', () => {
    for (const c of Object.values(CORPORA)) expect(c.text.length).toBeGreaterThan(300);
    expect(grammarValidity('. ' + CORPORA.grammar.text)).toBe(1);
    expect(corpusOf('hamlet').alphabet.length).toBeGreaterThan(20);
  });

  it('word validity is 1 for the text itself and 0 for gibberish', () => {
    expect(wordValidity(CORPORA.rhyme.text, 'rhyme')).toBe(1);
    expect(wordValidity('qzx wvk', 'rhyme')).toBe(0);
  });
});
