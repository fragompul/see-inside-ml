import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import { BpeTokenizer } from './tokenizer';
import { trainBpe } from './bpeTrain';

interface Fixture {
  texts: string[];
  cases: { merges: number; learned: [number, number][]; encoded: number[] }[];
}
/** ASCII-only decoder for the JSON file (the engine has no TextDecoder typings); the texts are escaped in the JSON. */
const asciiText = (bytes: Uint8Array): string => {
  let out = '';
  for (const b of bytes) out += String.fromCharCode(b);
  return out;
};
const fixture = JSON.parse(
  asciiText(readFileSync(new URL('../../fixtures/bpe-train.json', import.meta.url))),
) as Fixture;

describe('BPE training', () => {
  for (const c of fixture.cases)
    it(`learns the same ${c.merges} merges as the Python reference and encodes identically`, () => {
      const data = trainBpe(fixture.texts, c.merges);
      expect(data.merges).toEqual(c.learned);
      expect(new BpeTokenizer(data).encode(fixture.texts[1] as string)).toEqual(c.encoded);
    });

  it('round-trips any text, including characters it never saw', () => {
    const tok = new BpeTokenizer(trainBpe(['hello hello world'], 10));
    const text = 'héllo wörld 你好 \u{1F600}';
    expect(tok.decode(tok.encode(text))).toBe(text);
  });

  it('more merges never need more tokens for the training text', () => {
    const text = fixture.texts.join(' ');
    let last = Infinity;
    for (const n of [0, 10, 50, 200]) {
      const length = new BpeTokenizer(trainBpe([text], n)).encode(text).length;
      expect(length).toBeLessThanOrEqual(last);
      last = length;
    }
  });
});
