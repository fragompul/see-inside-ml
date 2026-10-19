import { type RnnLabConfig } from '@sim/deep';

/**
 * One-click experiments. Each applies a configuration, and `claim` says what the network reaches
 * on the test set after `epochs` epochs; `lessons.test.ts` runs every one and checks it.
 */
export interface Lesson {
  id: string;
  title: string;
  config: Partial<RnnLabConfig>;
  /** Where to scroll to look at the effect. */
  scrollTo: string;
  /** What to look for. */
  text: string;
  claim: { epochs: number; atLeast?: number; atMost?: number };
}

export const LESSONS: readonly Lesson[] = [
  {
    id: 'short',
    title: 'Short memory is easy',
    config: { task: 'first', seqLen: 6, cell: 'rnn' },
    scrollTo: 'rnn-train',
    text: 'Only five steps lie between the first symbol and the answer, so even a plain RNN gets there: accuracy reaches 100% within a few epochs.',
    claim: { epochs: 15, atLeast: 0.95 },
  },
  {
    id: 'forgets',
    title: 'The RNN forgets',
    config: { task: 'marker', seqLen: 12, cell: 'rnn' },
    scrollTo: 'rnn-train',
    text: 'Now the symbol to remember is hidden anywhere in 12 steps. After 25 epochs the vanilla RNN is still near chance (25%) while gated cells are done: the learning signal barely reaches the early steps (section 03). Let it run on and it does eventually get there, only far more slowly.',
    claim: { epochs: 25, atMost: 0.7 },
  },
  {
    id: 'lstm',
    title: 'Gates fix it: LSTM',
    config: { task: 'marker', seqLen: 12, cell: 'lstm' },
    scrollTo: 'rnn-train',
    text: 'Same task, same data, but with an LSTM: it learns what to write into its cell and when to keep it, and reaches ~100%. Open section 02 to watch the gates.',
    claim: { epochs: 25, atLeast: 0.95 },
  },
  {
    id: 'gru',
    title: 'A lighter fix: GRU',
    config: { task: 'marker', seqLen: 12, cell: 'gru' },
    scrollTo: 'rnn-train',
    text: 'The GRU has two gates instead of three and no separate cell, yet it solves the task just as well, usually a bit faster.',
    claim: { epochs: 25, atLeast: 0.95 },
  },
  {
    id: 'parity',
    title: 'Every token counts',
    config: { task: 'parity', seqLen: 8, cell: 'gru' },
    scrollTo: 'rnn-train',
    text: 'Even or odd depends on all eight symbols, and each one can flip the answer. A gated cell learns to toggle its state; accuracy jumps from 50% to 100% after a while.',
    claim: { epochs: 25, atLeast: 0.95 },
  },
];
