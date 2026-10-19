import { describe, expect, it } from 'vitest';
import { RetrievalIndex, charGrams, words } from './retrieval';

const docs = [
  'The colour of Velmora is teal.',
  'The colour of Kandrel is amber.',
  'The colour of Ostrava is teal.',
  'Nothing to do with colours at all.',
];

describe('tokenising', () => {
  it('lower-cases words and pads character grams with boundary markers', () => {
    expect(words('Hello, World 42!')).toEqual(['hello', 'world', '42']);
    expect(charGrams('Ab', 3)).toEqual(['^ab', 'ab$']);
  });
});

describe('retrieval index', () => {
  it('a document is most similar to itself, with cosine similarity exactly 1', () => {
    for (const kind of ['words', 'chars'] as const) {
      const index = new RetrievalIndex(docs, kind);
      docs.forEach((d, i) => {
        const s = index.scores(d);
        expect(s[i]).toBeCloseTo(1, 12);
        expect(index.search(d, 1)[0]).toBe(i);
      });
    }
  });

  it('a rare word is worth more than a common one: the question finds the document with the name', () => {
    const index = new RetrievalIndex(docs, 'words');
    expect(index.search('what is the colour of Kandrel', 1)[0]).toBe(1);
    // "teal" appears in two documents, so it separates them less than a name does
    const idfTeal = Math.log((4 + 1) / (2 + 1)) + 1;
    const idfName = Math.log((4 + 1) / (1 + 1)) + 1;
    expect(idfName).toBeGreaterThan(idfTeal);
  });

  it('character grams still find a name with a typo; whole words cannot', () => {
    const chars = new RetrievalIndex(docs, 'chars');
    const wordsIndex = new RetrievalIndex(docs, 'words');
    expect(chars.search('colour of Velmroa', 1)[0]).toBe(0);
    expect(wordsIndex.scores('Velmroa')).toEqual([0, 0, 0, 0]);
  });

  it('recall at k counts a question as found when its document is among the top k', () => {
    const index = new RetrievalIndex(docs, 'words');
    const qs = ['colour of Velmora', 'colour of Ostrava', 'colour of Kandrel'];
    expect(index.recallAtK(qs, [0, 2, 1], 1)).toBe(1);
    expect(index.recallAtK(qs, [3, 3, 3], 1)).toBe(0);
    expect(index.recallAtK(qs, [3, 3, 3], 4)).toBe(1);
  });

  it('ranks are deterministic: ties go to the earlier document', () => {
    const index = new RetrievalIndex(['same words', 'same words'], 'words');
    expect(index.search('same words', 2)).toEqual([0, 1]);
  });
});
