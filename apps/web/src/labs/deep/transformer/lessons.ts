import { type SamplingOptions } from '@sim/deep';

export interface Lesson {
  id: string;
  title: string;
  prompt: string;
  /** What to notice. Shown while the lesson is active. */
  text: string;
  /** Token to focus: 0-based index, or negative to count from the end (-1 = last). */
  focus: number;
  /** Section to bring into view. */
  scrollTo: 'tf-tokens' | 'tf-embeddings' | 'tf-attention' | 'tf-blocks' | 'tf-output';
  layer?: number;
  head?: number;
  /** Sampling for this lesson (defaults to the app's defaults, so lessons are reproducible). */
  sampling?: Partial<SamplingOptions>;
}

/**
 * Guided experiments. Every claim in `text` is asserted by `lessons.test.ts` against the real
 * model, so a retrained model that stops behaving this way fails the build instead of lying.
 */
export const LESSONS: readonly Lesson[] = [
  {
    id: 'pronoun-she',
    title: 'Anna → “She”',
    prompt: 'Anna found a ball.',
    focus: -1,
    scrollTo: 'tf-output',
    text: 'After the full stop the story needs a new subject. Nobody wrote a rule for it, yet “She” takes about 98% of the probability: the model learned that Anna is a girl’s name.',
  },
  {
    id: 'pronoun-he',
    title: 'Tom → “He”',
    prompt: 'Tom found a ball.',
    focus: -1,
    scrollTo: 'tf-output',
    text: 'Same sentence, different name, opposite answer. The gender of the name, seen four tokens earlier, flips the prediction from “She” to “He”.',
  },
  {
    id: 'they',
    title: 'Two names → “They”',
    prompt: 'Anna and Max went to the park.',
    focus: -1,
    scrollTo: 'tf-output',
    text: 'Two people were mentioned, so the next sentence starts with “They”. Counting names is something the model had to discover on its own.',
  },
  {
    id: 'it',
    title: '“took” → “it”',
    prompt: 'Eva found a lamp. She took',
    focus: -1,
    scrollTo: 'tf-output',
    text: 'Almost 100% for “it”: an object was just introduced, and a pronoun for objects is what fits after “took”.',
  },
  {
    id: 'where-she-comes-from',
    title: 'Who is “She”? Follow the attention',
    prompt: 'Anna found a ball.',
    focus: -1,
    layer: 1,
    head: 0,
    scrollTo: 'tf-attention',
    text: 'In layer 2, head 1, the full stop (last row of the matrix) puts almost all its weight on the first column, “Anna”. That is how the pronoun’s gender gets decided: one head fetches the name. Try the other heads and compare.',
  },
  {
    id: 'copy',
    title: 'Copying from the story',
    prompt: 'Ben, Sam and Zoe like cakes. Ben brings a',
    focus: -1,
    layer: 0,
    head: 1,
    scrollTo: 'tf-attention',
    text: 'The last token needs an object. Layer 1, head 2 points straight back at “cake”, mentioned seven tokens earlier, and the model predicts “cake” with near certainty. Copying from context is one of the first skills a transformer picks up.',
  },
  {
    id: 'greeting',
    title: 'Who does Mia greet?',
    prompt: 'Mia and Leo went to the park. Mia said hello to',
    focus: -1,
    layer: 1,
    head: 0,
    scrollTo: 'tf-attention',
    text: 'Two friends were named in the first sentence. The model fills in “Leo” with about 99%, and in layer 2, head 1 the last token looks straight back at “Leo” instead of inventing a name. It is copying a name it has seen.',
  },
  {
    id: 'plural',
    title: 'Plural subject',
    prompt: 'The big dogs',
    focus: -1,
    scrollTo: 'tf-output',
    text: 'A plural subject takes a bare verb (run, jump, sleep…). Compare with the next experiment: the candidates here do not end in “s”.',
  },
  {
    id: 'singular',
    title: 'Singular subject',
    prompt: 'The big dog',
    focus: -1,
    scrollTo: 'tf-output',
    text: 'One dog, so the verb must end in “s” (runs, sleeps, waits…). Subject-verb agreement is learned purely from the stories.',
  },
  {
    id: 'greedy',
    title: 'Temperature 0',
    prompt: 'The tired fox runs in the',
    focus: -1,
    scrollTo: 'tf-output',
    sampling: { temperature: 0 },
    text: 'Temperature 0 always picks the single most likely token, so the same prompt gives the same text every time. Press Generate twice to check.',
  },
  {
    id: 'hot',
    title: 'Temperature 2',
    prompt: 'The tired fox runs in the',
    focus: -1,
    scrollTo: 'tf-output',
    sampling: { temperature: 2 },
    text: 'At temperature 2 the distribution flattens: unlikely tokens get a real chance and the story drifts into nonsense. Slide it back down to see order return.',
  },
];
