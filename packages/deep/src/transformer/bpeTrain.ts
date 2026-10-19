import { END_OF_TEXT, PRETOKEN_PATTERN, type TokenizerData } from './tokenizer';
import { encodeUtf8 } from './utf8';

/**
 * Learns byte-level BPE merges from text: repeatedly fuse the most frequent adjacent pair of tokens inside
 * words. Identical to `BPE.train` in tools/transformer/bpe.py (ties go to the smaller pair), so the merges
 * can be checked against the Python implementation. Stops early when no pair is left to fuse.
 */
export function trainBpe(texts: readonly string[], merges: number): TokenizerData {
  const counts = new Map<string, number>();
  for (const text of texts)
    for (const word of text.match(PRETOKEN_PATTERN) ?? [])
      counts.set(word, (counts.get(word) ?? 0) + 1);
  let words: { ids: number[]; count: number }[] = [...counts].map(([w, count]) => ({
    ids: encodeUtf8(w),
    count,
  }));
  const learned: [number, number][] = [];
  for (let step = 0; step < merges; step++) {
    const pairs = new Map<number, number>();
    for (const { ids, count } of words)
      for (let i = 0; i + 1 < ids.length; i++) {
        const key = (ids[i] as number) * 65536 + (ids[i + 1] as number);
        pairs.set(key, (pairs.get(key) ?? 0) + count);
      }
    if (pairs.size === 0) break;
    let best = -1;
    let bestCount = -1;
    for (const [key, c] of pairs)
      if (c > bestCount || (c === bestCount && key < best)) {
        best = key;
        bestCount = c;
      }
    const left = Math.floor(best / 65536);
    const right = best % 65536;
    learned.push([left, right]);
    const id = 256 + step;
    words = words.map(({ ids, count }) => {
      const out: number[] = [];
      for (let i = 0; i < ids.length; i++) {
        if (i + 1 < ids.length && ids[i] === left && ids[i + 1] === right) {
          out.push(id);
          i++;
        } else out.push(ids[i] as number);
      }
      return { ids: out, count };
    });
  }
  return { merges: learned, specialTokens: { [END_OF_TEXT]: 256 + learned.length } };
}
