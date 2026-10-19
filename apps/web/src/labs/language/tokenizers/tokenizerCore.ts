import { BpeTokenizer, PRETOKEN_PATTERN, encodeUtf8, trainBpe } from '@sim/deep';
import { TESTS, TRAINING, type TestText, type TrainingSet } from './tokenizerData';

export type Scheme = 'bytes' | 'words' | 'bpe';

export const SCHEME_LABEL: Record<Scheme, string> = {
  bytes: 'Bytes (no learning)',
  words: 'Whole words',
  bpe: 'Byte-pair encoding',
};

export interface TokenizerScenario {
  scheme: Scheme;
  /** Merges learned by BPE; the vocabulary is 256 bytes plus one token per merge. */
  merges: number;
  trainOn: TrainingSet;
  testOn: TestText;
}

export const DEFAULT_TOKENIZER: TokenizerScenario = {
  scheme: 'bpe',
  merges: 100,
  trainOn: 'both',
  testOn: 'prose',
};

export interface TokenizerRun {
  scenario: TokenizerScenario;
  vocab: number;
  /** Tokens used for the test text, and per 100 characters of it. */
  tokens: number;
  per100: number;
  /** Share of the test text (pieces) the vocabulary has no entry for; only whole-word vocabularies have this. */
  unknown: number;
  /** The test text cut into its tokens. `unknown` marks a piece that was replaced by the unknown token. */
  pieces: { text: string; unknown: boolean }[];
}

const bpeCache = new Map<string, BpeTokenizer>();
export function bpeFor(trainOn: TrainingSet, merges: number): BpeTokenizer {
  const key = `${trainOn}|${merges}`;
  let tok = bpeCache.get(key);
  if (!tok) {
    tok = new BpeTokenizer(trainBpe(TRAINING[trainOn], merges));
    bpeCache.set(key, tok);
  }
  return tok;
}

export function runTokenizer(s: TokenizerScenario): TokenizerRun {
  const text = TESTS[s.testOn].text;
  let pieces: { text: string; unknown: boolean }[];
  let vocab: number;
  if (s.scheme === 'bytes') {
    vocab = 256;
    // multi-byte characters are shown as the bytes they are made of
    pieces = Array.from(text).flatMap((ch) =>
      encodeUtf8(ch).length === 1
        ? [{ text: ch, unknown: false }]
        : encodeUtf8(ch).map((b) => ({ text: `<${b.toString(16)}>`, unknown: false })),
    );
  } else if (s.scheme === 'words') {
    const seen = new Set(TRAINING[s.trainOn].flatMap((t) => t.match(PRETOKEN_PATTERN) ?? []));
    vocab = seen.size + 1; // plus the unknown token
    pieces = (text.match(PRETOKEN_PATTERN) ?? []).map((w) => ({ text: w, unknown: !seen.has(w) }));
  } else {
    const tok = bpeFor(s.trainOn, s.merges);
    vocab = tok.vocabSize;
    pieces = tok.encode(text).map((id) => ({ text: tok.tokenString(id), unknown: false }));
  }
  const unknown = pieces.filter((p) => p.unknown).length;
  return {
    scenario: s,
    vocab,
    tokens: pieces.length,
    per100: (pieces.length * 100) / Array.from(text).length,
    unknown: unknown / pieces.length,
    pieces,
  };
}
