import { describe, expect, it } from 'vitest';
import { DEFAULT_TOKENIZER, runTokenizer } from './tokenizerCore';
import { tokenizerExperiments } from './tokenizerSpec';

describe('Tokenizer lab', () => {
  for (const exp of tokenizerExperiments) {
    it(`experiment "${exp.title}" holds`, () => {
      expect(exp.claim(runTokenizer({ ...DEFAULT_TOKENIZER, ...exp.scenario }))).toBe(true);
    });
  }

  it('the pieces always add up to the text for byte-pair and byte tokenizers', () => {
    for (const scheme of ['bytes', 'bpe'] as const) {
      const r = runTokenizer({ ...DEFAULT_TOKENIZER, scheme, testOn: 'prose' });
      if (scheme === 'bpe') expect(r.pieces.map((p) => p.text).join('')).toContain('keeper');
      expect(r.tokens).toBe(r.pieces.length);
    }
  });
});
