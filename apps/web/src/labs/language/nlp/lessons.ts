import { DEFAULT_NLP_LAB, NlpSession, type NlpLabConfig } from '@sim/deep';

/**
 * One-click experiments. `claim.check` verifies, on a session trained for `claim.steps` steps with
 * the lesson's configuration, the statement the text makes; `lessons.test.ts` runs them all.
 */
export interface Lesson {
  id: string;
  title: string;
  config: Partial<NlpLabConfig>;
  scrollTo: string;
  text: string;
  claim: { steps: number; check: (session: NlpSession) => boolean };
}

const STEPS = 6000;

/** A sibling session with the same settings but some changes, trained for `steps`. */
function sibling(s: NlpSession, patch: Partial<NlpLabConfig>, steps = STEPS): NlpSession {
  const other = new NlpSession({ ...DEFAULT_NLP_LAB, ...s.config, ...patch });
  other.train(steps);
  return other;
}

const accuracyOf = (s: NlpSession, id: string) =>
  s.analogyAccuracyByGroup().find((g) => g.id === id)?.accuracy as number;

export const LESSONS: readonly Lesson[] = [
  {
    id: 'company',
    title: 'You shall know a word by its company',
    config: {},
    scrollTo: 'nlp-map',
    text: 'Nobody told the model that cats and dogs are animals. It only saw which words appear near which. Watch the map: untrained, the words are scattered at random; after a short while each family (animals, capitals, food, royalty…) gathers into its own cluster.',
    claim: {
      steps: 2000,
      check: (s) => {
        const trained = s.neighbourPurity();
        const fresh = new NlpSession({ ...DEFAULT_NLP_LAB, ...s.config }).neighbourPurity();
        return fresh < 0.6 && trained > 0.9;
      },
    },
  },
  {
    id: 'queen',
    title: 'King − man + woman',
    config: {},
    scrollTo: 'nlp-analogy',
    text: 'Directions in the vector space carry meaning. Take the vector from "man" to "king" (royalty) and add it to "woman": you land next to "queen". Try the calculator in section 04 with your own words.',
    claim: {
      steps: STEPS,
      check: (s) => s.analogy('man', 'king', 'woman', 1)[0]?.word === 'queen',
    },
  },
  {
    id: 'capitals',
    title: 'Capitals are a direction',
    config: {},
    scrollTo: 'nlp-analogy',
    text: 'France to Paris, Italy to Rome, Japan to Tokyo: the step from a country to its capital is roughly the same arrow every time, so "france : paris :: italy : ?" is answered correctly almost every time.',
    claim: { steps: STEPS, check: (s) => accuracyOf(s, 'capital') >= 0.8 },
  },
  {
    id: 'narrow',
    title: 'A narrow window',
    config: { window: 1 },
    scrollTo: 'nlp-analogy',
    text: 'With a window of 1 each word only sees its immediate neighbours, which is not enough to link a country with the words around its capital. Analogy accuracy stays low; widen the window (section 05) and it climbs.',
    claim: {
      steps: STEPS,
      check: (s) => s.analogyAccuracy() < 0.35 && sibling(s, { window: 4 }).analogyAccuracy() > 0.6,
    },
  },
  {
    id: 'tiny',
    title: 'Too few dimensions',
    config: { dim: 2 },
    scrollTo: 'nlp-map',
    text: 'Two numbers per word are not enough room to keep apart everything that should be apart: families blur together and analogies fail. Give the vectors more dimensions in section 05.',
    claim: {
      steps: STEPS,
      check: (s) => s.analogyAccuracy() < 0.3 && sibling(s, { dim: 16 }).analogyAccuracy() > 0.6,
    },
  },
  {
    id: 'diverge',
    title: 'Learning rate too high',
    config: { lr: 0.5 },
    scrollTo: 'nlp-train',
    text: 'With a learning rate of 0.5 the updates overshoot, the loss explodes within a few hundred steps and the vectors are ruined. Training stops with a warning; lower the rate and press Reset.',
    claim: { steps: 3000, check: (s) => s.diverged },
  },
];
