import { createRng } from '@sim/core';
import { makeCorpus, type Corpus } from '@sim/deep';

export type CorpusId = 'rhyme' | 'hamlet' | 'grammar';

const RHYME = [
  'twinkle, twinkle, little star,',
  'how i wonder what you are!',
  'up above the world so high,',
  'like a diamond in the sky.',
  '',
].join('\n');

const HAMLET = [
  'to be, or not to be, that is the question:',
  "whether 'tis nobler in the mind to suffer",
  'the slings and arrows of outrageous fortune,',
  'or to take arms against a sea of troubles',
  'and by opposing end them. to die: to sleep;',
  'no more; and by a sleep to say we end',
  'the heart-ache and the thousand natural shocks',
  "that flesh is heir to, 'tis a consummation",
  "devoutly to be wish'd. to die, to sleep;",
  "to sleep: perchance to dream: ay, there's the rub;",
  'for in that sleep of death what dreams may come',
  '',
].join('\n');

function grammar(): string {
  const rng = createRng(21);
  const pick = <T>(v: readonly T[]): T => v[Math.floor(rng.next() * v.length)] as T;
  const lines: string[] = [];
  for (let i = 0; i < 70; i++)
    lines.push(
      `the ${pick(['cat', 'dog', 'bird'])} ${pick(['sat', 'ran', 'slept'])} on the ${pick(['mat', 'hill', 'log'])}.`,
    );
  return lines.join(' ') + ' ';
}

export const CORPORA: Record<CorpusId, { label: string; note: string; text: string }> = {
  rhyme: {
    label: 'A nursery rhyme',
    note: 'Four short lines repeated: an easy text with a clear structure.',
    text: RHYME.repeat(5),
  },
  hamlet: {
    label: 'To be or not to be',
    note: 'About 450 characters of Shakespeare: more varied, with far more to remember than the model can hold.',
    text: HAMLET,
  },
  grammar: {
    label: 'A tiny grammar',
    note: 'Sentences generated from a template, "the (cat|dog|bird) (sat|ran|slept) on the (mat|hill|log).": predictable structure, random content.',
    text: grammar(),
  },
};

export const corpusOf = (id: CorpusId): Corpus => makeCorpus(CORPORA[id].text);
