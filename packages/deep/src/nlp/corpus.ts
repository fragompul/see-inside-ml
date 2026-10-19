import { type Rng } from '@sim/core';

/**
 * A tiny made-up world, written so that word vectors learned from it have real structure: gender
 * (his/her), royalty (crown/kingdom against market/bag), countries and their capitals, animals and
 * food. Sentences come from a handful of templates, so the corpus is infinite, licence-free and
 * reproducible, and every claim made about it can be tested.
 */

export type WordCategory =
  'royal' | 'people' | 'country' | 'capital' | 'animal' | 'food' | 'function';

export const CATEGORY_LABEL: Record<WordCategory, string> = {
  royal: 'royalty',
  people: 'people',
  country: 'countries',
  capital: 'capitals',
  animal: 'animals',
  food: 'food',
  function: 'function words',
};

/** `[male, female]` pairs. */
export const ROYAL_PAIRS: readonly (readonly [string, string])[] = [
  ['king', 'queen'],
  ['prince', 'princess'],
];
export const PEOPLE_PAIRS: readonly (readonly [string, string])[] = [
  ['man', 'woman'],
  ['boy', 'girl'],
  ['uncle', 'aunt'],
];
/** `[country, capital]` pairs. */
export const COUNTRY_PAIRS: readonly (readonly [string, string])[] = [
  ['france', 'paris'],
  ['italy', 'rome'],
  ['spain', 'madrid'],
  ['japan', 'tokyo'],
  ['egypt', 'cairo'],
  ['peru', 'lima'],
];
export const ANIMALS = ['cat', 'dog', 'horse', 'fox', 'rabbit', 'bird'] as const;
export const FOODS = ['apple', 'bread', 'cheese', 'soup', 'rice', 'cake'] as const;
const PLACES = ['forest', 'garden', 'farm'] as const;

const FUNCTION_WORDS = [
  'the',
  'a',
  'is',
  'of',
  'in',
  'to',
  'and',
  'with',
  'we',
  'was',
  'he',
  'she',
  'his',
  'her',
  'ruled',
  'wore',
  'walked',
  'carried',
  'lived',
  'ate',
  'slept',
  'chased',
  'visited',
  'love',
  'capital',
  'city',
  'country',
  'people',
  'kingdom',
  'crown',
  'castle',
  'court',
  'market',
  'bag',
  'village',
  'family',
  'warm',
  'forest',
  'garden',
  'farm',
] as const;

/** Every word of the world, grouped by category (function words last). */
export const WORLD: Record<WordCategory, readonly string[]> = {
  royal: ROYAL_PAIRS.flat(),
  people: PEOPLE_PAIRS.flat(),
  country: COUNTRY_PAIRS.map((p) => p[0]),
  capital: COUNTRY_PAIRS.map((p) => p[1]),
  animal: ANIMALS,
  food: FOODS,
  function: FUNCTION_WORDS,
};

export const VOCABULARY: readonly string[] = Object.values(WORLD).flat();

const CATEGORY_OF = new Map<string, WordCategory>();
for (const [category, words] of Object.entries(WORLD) as [WordCategory, readonly string[]][])
  for (const w of words) CATEGORY_OF.set(w, category);

export function categoryOf(word: string): WordCategory {
  const c = CATEGORY_OF.get(word);
  if (!c) throw new RangeError(`Unknown word "${word}"`);
  return c;
}

const pick = <T>(items: readonly T[], rng: Rng): T =>
  items[Math.floor(rng.next() * items.length)] as T;

/** One random sentence (a list of words). */
export function randomSentence(rng: Rng): string[] {
  const kind = rng.next();
  if (kind < 0.3) {
    // royalty and everyday people: identical sentences except the pronoun and the nouns of their world
    const royal = rng.next() < 0.45;
    const [male, female] = pick(royal ? ROYAL_PAIRS : PEOPLE_PAIRS, rng);
    const isMale = rng.next() < 0.5;
    const w = isMale ? male : female;
    const p = isMale ? 'he' : 'she';
    const pp = isMale ? 'his' : 'her';
    const templates = royal
      ? [
          ['the', w, 'ruled', 'the', 'kingdom', 'and', p, 'wore', pp, 'crown'],
          ['the', w, 'lived', 'in', 'the', 'castle', 'with', pp, 'court'],
          ['the', w, 'ate', 'the', pick(FOODS, rng), 'in', 'the', 'castle'],
        ]
      : [
          ['the', w, 'walked', 'to', 'the', 'market', 'and', p, 'carried', pp, 'bag'],
          ['the', w, 'lived', 'in', 'the', 'village', 'with', pp, 'family'],
          ['the', w, 'ate', 'the', pick(FOODS, rng), 'in', 'the', 'village'],
        ];
    return pick(templates, rng);
  }
  if (kind < 0.55) {
    const [country, capital] = pick(COUNTRY_PAIRS, rng);
    return pick(
      [
        [capital, 'is', 'the', 'capital', 'of', country],
        ['the', 'city', 'of', capital, 'is', 'in', country],
        ['we', 'visited', 'the', 'city', 'of', capital],
        ['we', 'visited', 'the', 'country', 'of', country],
        ['the', 'people', 'of', country, 'love', country],
      ],
      rng,
    );
  }
  if (kind < 0.85) {
    const a = pick(ANIMALS, rng);
    return pick(
      [
        ['the', a, 'ate', 'the', pick(FOODS, rng)],
        ['the', a, 'slept', 'in', 'the', pick(PLACES, rng)],
        ['a', a, 'chased', 'a', pick(ANIMALS, rng)],
      ],
      rng,
    );
  }
  const f = pick(FOODS, rng);
  return pick(
    [
      ['the', f, 'was', 'warm'],
      ['we', 'ate', 'the', f, 'and', 'the', pick(FOODS, rng)],
    ],
    rng,
  );
}

export function generateCorpus(sentences: number, rng: Rng): string[][] {
  return Array.from({ length: sentences }, () => randomSentence(rng));
}

/** An analogy question "a is to b as c is to d". */
export type Analogy = readonly [string, string, string, string];

export interface AnalogyGroup {
  id: string;
  label: string;
  questions: Analogy[];
}

function fromPairs(pairs: readonly (readonly [string, string])[]): Analogy[] {
  const out: Analogy[] = [];
  for (const [a, b] of pairs) for (const [c, d] of pairs) if (a !== c) out.push([a, b, c, d]);
  return out;
}

/** All analogy questions the world supports, in named groups. */
export const ANALOGY_GROUPS: readonly AnalogyGroup[] = [
  {
    id: 'gender',
    label: 'Gender (man : woman)',
    questions: fromPairs([...ROYAL_PAIRS, ...PEOPLE_PAIRS]),
  },
  { id: 'capital', label: 'Capitals (country : capital)', questions: fromPairs(COUNTRY_PAIRS) },
  {
    id: 'royalty',
    label: 'Royalty (man : king)',
    questions: fromPairs(
      ROYAL_PAIRS.map((r, i) => [(PEOPLE_PAIRS[i] as readonly [string, string])[0], r[0]] as const),
    ).concat(
      fromPairs(
        ROYAL_PAIRS.map(
          (r, i) => [(PEOPLE_PAIRS[i] as readonly [string, string])[1], r[1]] as const,
        ),
      ),
    ),
  },
];

export const ALL_ANALOGIES: Analogy[] = ANALOGY_GROUPS.flatMap((g) => g.questions);
