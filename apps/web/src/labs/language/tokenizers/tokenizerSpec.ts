import type { ParamSpec } from '@/shared/lab-kit';
import {
  SCHEME_LABEL,
  bpeFor,
  runTokenizer,
  type Scheme,
  type TokenizerRun,
  type TokenizerScenario,
} from './tokenizerCore';
import { TESTS, type TestText, type TrainingSet } from './tokenizerData';

export const tokenizerParams: ParamSpec[] = [
  {
    key: 'scheme',
    label: 'Tokenizer',
    kind: 'select',
    options: (Object.keys(SCHEME_LABEL) as Scheme[]).map((id) => ({
      value: id,
      label: SCHEME_LABEL[id],
    })),
  },
  {
    key: 'merges',
    label: 'Merges learned (BPE)',
    kind: 'ladder',
    values: [0, 20, 100, 400],
    format: (v) => String(v),
    hint: 'Each merge adds one token to the vocabulary.',
  },
  {
    key: 'trainOn',
    label: 'Learned from',
    kind: 'select',
    options: [
      { value: 'prose', label: 'English prose' },
      { value: 'code', label: 'Code' },
      { value: 'both', label: 'Prose and code' },
    ] satisfies { value: TrainingSet; label: string }[],
  },
  {
    key: 'testOn',
    label: 'Measured on',
    kind: 'select',
    options: (Object.keys(TESTS) as TestText[]).map((id) => ({
      value: id,
      label: (TESTS[id] as { label: string }).label,
    })),
  },
];

export interface TokenizerExperiment {
  id: string;
  title: string;
  text: string;
  scenario: Partial<TokenizerScenario>;
  claim: (run: TokenizerRun) => boolean;
}

const same = (r: TokenizerRun, change: Partial<TokenizerScenario>) =>
  runTokenizer({ ...r.scenario, ...change });

export const tokenizerExperiments: TokenizerExperiment[] = [
  {
    id: 'merges',
    title: 'Every merge shortens the text',
    text: 'With no merges a sentence costs one token per byte. Learning 400 merges from prose and code cuts new English sentences to under 40 tokens per 100 characters, at the price of a vocabulary of 657 instead of 257.',
    scenario: { scheme: 'bpe', trainOn: 'both', testOn: 'prose', merges: 400 },
    claim: (r) =>
      r.per100 < 0.4 * same(r, { merges: 0 }).per100 &&
      same(r, { merges: 100 }).per100 > r.per100 &&
      same(r, { merges: 20 }).per100 > same(r, { merges: 100 }).per100 &&
      r.vocab === 657,
  },
  {
    id: 'unknown',
    title: 'Whole words break on new words',
    text: 'A whole-word vocabulary learned from the training text has no entry for more than a fifth of the words in new English sentences, and for nearly all the words of Spanish. A byte-pair tokenizer never meets an unknown piece: it can always fall back on bytes, and the text comes back out unchanged.',
    scenario: { scheme: 'words', trainOn: 'prose', testOn: 'spanish' },
    claim: (r) => {
      const bpe = same(r, { scheme: 'bpe', merges: 100 });
      const text = TESTS[r.scenario.testOn].text;
      const tok = bpeFor(r.scenario.trainOn, 100);
      return (
        r.unknown > 0.8 &&
        same(r, { testOn: 'prose' }).unknown > 0.2 &&
        bpe.unknown === 0 &&
        tok.decode(tok.encode(text)) === text
      );
    },
  },
  {
    id: 'domain',
    title: 'A tokenizer is only as good as its training text',
    text: 'Merges learned from code shrink new code to about 40 tokens per 100 characters. Merges learned from prose, the same number of them, leave it at 85: the pieces it learned are not the ones code is made of.',
    scenario: { scheme: 'bpe', trainOn: 'code', testOn: 'code', merges: 100 },
    claim: (r) => r.per100 < 0.55 * same(r, { trainOn: 'prose' }).per100,
  },
  {
    id: 'language',
    title: 'Other languages pay a tax',
    text: 'A tokenizer trained on English prose needs more than twice as many tokens per character for Spanish: accents become separate bytes and the common endings were never learned. Every token costs compute and counts against the context window, so the same sentence costs more.',
    scenario: { scheme: 'bpe', trainOn: 'prose', testOn: 'spanish', merges: 400 },
    claim: (r) => r.per100 > 2 * same(r, { testOn: 'prose' }).per100,
  },
  {
    id: 'numbers',
    title: 'Numbers fall to pieces',
    text: 'Digits are rare in prose, so no merge ever joined them: each digit is its own token. A phone number or a date costs a token per character, and long numbers are split without regard to place value.',
    scenario: { scheme: 'bpe', trainOn: 'prose', testOn: 'numbers', merges: 400 },
    claim: (r) =>
      r.per100 > 2 * same(r, { testOn: 'prose' }).per100 &&
      r.pieces.filter((p) => /^\d$/.test(p.text)).length >= 30,
  },
];
