/**
 * Word categories of the training stories (tools/transformer/corpus.py). Only used to colour the
 * embedding map so you can see whether the model grouped names, animals, verbs… on its own.
 * Keep the word lists in sync with the corpus generator.
 */
export type Category =
  | 'person'
  | 'animal'
  | 'object'
  | 'place'
  | 'verb'
  | 'adjective'
  | 'pronoun'
  | 'function'
  | 'punctuation'
  | 'other';

const words = (list: string) => list.split(/\s+/).filter(Boolean);

const LISTS: Record<Exclude<Category, 'punctuation' | 'other'>, Set<string>> = {
  person: new Set(
    words('Anna Mia Eva Lily Nora Zoe Ruby Ella Tom Leo Sam Max Ben Jack Finn Oscar'),
  ),
  animal: new Set(
    words(
      'cat cats dog dogs bird birds fox foxes mouse mice horse horses rabbit rabbits duck ducks',
    ),
  ),
  object: new Set(
    words('ball balls book books apple apples key keys hat hats cake cakes kite kites lamp lamps'),
  ),
  place: new Set(words('park garden kitchen school river forest farm beach')),
  verb: new Set(
    words(
      `sees see finds find found likes like liked takes take took drops drop dropped opens open
       opened carries carry carried follows follow followed runs run sleeps sleep jumps jump sings
       sing waits wait plays play went said brings`,
    ),
  ),
  adjective: new Set(words('big small red happy tired hungry quiet quick kind brave')),
  pronoun: new Set(words('she he it they She He It They her his')),
  function: new Set(words('the The a an to in at At and is are was there lot hello of on with')),
};

/** Category of a token's text (leading space ignored). */
export function categoryOf(token: string): Category {
  const word = token.trim();
  if (word === '') return 'other';
  if (/^[^A-Za-z0-9]+$/.test(word)) return 'punctuation';
  for (const [category, set] of Object.entries(LISTS)) {
    if (set.has(word)) return category as Category;
  }
  return 'other';
}

/** CSS colour variable for each category (reuses the semantic class palette). */
export const CATEGORY_COLOR: Record<Category, string> = {
  person: 'var(--class-0)',
  animal: 'var(--class-1)',
  object: 'var(--class-2)',
  place: 'var(--class-3)',
  verb: 'var(--class-4)',
  adjective: 'var(--brand)',
  pronoun: 'var(--fg)',
  function: 'var(--muted)',
  punctuation: 'var(--faint)',
  other: 'var(--faint)',
};

export const CATEGORY_LABEL: Record<Category, string> = {
  person: 'names',
  animal: 'animals',
  object: 'objects',
  place: 'places',
  verb: 'verbs',
  adjective: 'adjectives',
  pronoun: 'pronouns',
  function: 'function words',
  punctuation: 'punctuation',
  other: 'other',
};
