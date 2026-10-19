import { WORLD, type WordCategory } from '@sim/deep';

/** One theme-aware colour per word family (the design system's class colours and the brand). */
export const CATEGORY_COLOR: Record<WordCategory, string> = {
  royal: 'var(--class-3)',
  people: 'var(--class-4)',
  country: 'var(--class-1)',
  capital: 'var(--class-2)',
  animal: 'var(--class-0)',
  food: 'var(--brand)',
  function: 'var(--muted)',
};

/** Content words (everything except function words), alphabetically, for pickers. */
export const CONTENT_WORDS: string[] = (
  Object.entries(WORLD) as [WordCategory, readonly string[]][]
)
  .filter(([category]) => category !== 'function')
  .flatMap(([, words]) => [...words])
  .sort();
