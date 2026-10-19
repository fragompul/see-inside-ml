import { decodeUtf8, encodeUtf8 } from './utf8';

/**
 * Pre-tokenisation pattern. Keep in sync with PRETOKEN_PATTERN in tools/transformer/bpe.py.
 * Words keep their leading space (" cat"), so "cat" and " cat" are different tokens, as in GPT-2.
 */
export const PRETOKEN_PATTERN = / ?[A-Za-z]+| ?[0-9]+| ?[^\sA-Za-z0-9]+|\s+(?!\S)|\s+/g;

export const END_OF_TEXT = '<|endoftext|>';

export interface TokenizerData {
  /** Merge i fuses the pair (a, b) into token 256 + i. */
  merges: [number, number][];
  specialTokens: Record<string, number>;
}

/** One state of a word while BPE merges are being applied. */
export interface MergeStep {
  /** The pair that was fused (undefined for the initial, all-bytes state). */
  merged?: { left: number; right: number; into: number };
  /** Token ids after this step. */
  ids: number[];
}

/**
 * Byte-level BPE tokenizer, a faithful port of tools/transformer/bpe.py (parity-tested).
 * Any string can be encoded, because the 256 byte values are always in the vocabulary.
 */
export class BpeTokenizer {
  readonly vocabSize: number;
  readonly eotId: number;
  private readonly ranks = new Map<number, number>();
  private readonly bytes: number[][] = [];
  private readonly cache = new Map<string, number[]>();

  constructor(readonly data: TokenizerData) {
    for (let i = 0; i < 256; i++) this.bytes.push([i]);
    data.merges.forEach(([a, b], rank) => {
      if (!(a >= 0 && b >= 0 && a < 256 + rank && b < 256 + rank)) {
        throw new RangeError(`Merge ${rank} refers to a token that does not exist yet`);
      }
      this.ranks.set(pairKey(a, b), rank);
      this.bytes.push([...(this.bytes[a] as number[]), ...(this.bytes[b] as number[])]);
    });
    this.eotId = this.bytes.length;
    this.vocabSize = this.eotId + 1;
    const declared = data.specialTokens[END_OF_TEXT];
    if (declared !== undefined && declared !== this.eotId) {
      throw new RangeError(`${END_OF_TEXT} should be id ${this.eotId}, file says ${declared}`);
    }
  }

  /** Splits text into pre-tokens (words, numbers, punctuation, whitespace runs). */
  pretokenize(text: string): string[] {
    return text.match(PRETOKEN_PATTERN) ?? [];
  }

  private encodeWord(word: string): number[] {
    const cached = this.cache.get(word);
    if (cached) return cached;
    const ids = this.applyMerges(encodeUtf8(word)).ids;
    this.cache.set(word, ids);
    return ids;
  }

  private applyMerges(bytes: number[], record?: MergeStep[]): { ids: number[] } {
    const ids = [...bytes];
    record?.push({ ids: [...ids] });
    while (ids.length > 1) {
      let bestRank = Infinity;
      let bestAt = -1;
      for (let i = 0; i < ids.length - 1; i++) {
        const rank = this.ranks.get(pairKey(ids[i] as number, ids[i + 1] as number));
        if (rank !== undefined && rank < bestRank) {
          bestRank = rank;
          bestAt = i;
        }
      }
      if (bestAt < 0) break;
      const left = ids[bestAt] as number;
      const right = ids[bestAt + 1] as number;
      ids.splice(bestAt, 2, 256 + bestRank);
      record?.push({ merged: { left, right, into: 256 + bestRank }, ids: [...ids] });
    }
    return { ids };
  }

  encode(text: string): number[] {
    const out: number[] = [];
    text.split(END_OF_TEXT).forEach((chunk, index) => {
      if (index > 0) out.push(this.eotId);
      for (const word of this.pretokenize(chunk)) out.push(...this.encodeWord(word));
    });
    return out;
  }

  /** Like `encode`, but keeps pre-tokens apart (for display) and reports each token's text. */
  encodeDetailed(text: string): { word: string; ids: number[] }[] {
    const groups: { word: string; ids: number[] }[] = [];
    text.split(END_OF_TEXT).forEach((chunk, index) => {
      if (index > 0) groups.push({ word: END_OF_TEXT, ids: [this.eotId] });
      for (const word of this.pretokenize(chunk)) groups.push({ word, ids: this.encodeWord(word) });
    });
    return groups;
  }

  /** Every state a pre-token goes through while BPE fuses its bytes, for the merge animation. */
  mergeSteps(word: string): MergeStep[] {
    const steps: MergeStep[] = [];
    this.applyMerges(encodeUtf8(word), steps);
    return steps;
  }

  tokenBytes(id: number): number[] {
    if (id === this.eotId) return encodeUtf8(END_OF_TEXT);
    const b = this.bytes[id];
    if (!b) throw new RangeError(`Unknown token id ${id}`);
    return b;
  }

  /** Display string of one token (a lone byte of a multi-byte character shows as U+FFFD). */
  tokenString(id: number): string {
    return id === this.eotId ? END_OF_TEXT : decodeUtf8(this.tokenBytes(id));
  }

  decode(ids: readonly number[]): string {
    const bytes: number[] = [];
    for (const id of ids) bytes.push(...this.tokenBytes(id));
    return decodeUtf8(bytes);
  }
}

const pairKey = (a: number, b: number) => a * 65536 + b;
