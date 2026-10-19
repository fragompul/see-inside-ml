import { type Rng } from '@sim/core';

/**
 * Sequence-classification tasks that each stress one thing a recurrent network must do. Every
 * example is a list of tokens, and the label depends on the whole list (many-to-one).
 */
export const TASK_NAMES = ['first', 'marker', 'parity'] as const;
export type TaskName = (typeof TASK_NAMES)[number];

export interface TaskInfo {
  label: string;
  /** One-line rule shown to the learner. */
  rule: string;
  /** Why this task is interesting. */
  note: string;
  /** Number of distinct tokens. */
  vocab: number;
  /** Number of classes. */
  classes: number;
  /** Printable symbol of each token. */
  symbols: readonly string[];
  /** Printable symbol of each class. */
  classSymbols: readonly string[];
  minLength: number;
  maxLength: number;
}

export const taskInfo: Record<TaskName, TaskInfo> = {
  first: {
    label: 'Remember the first',
    rule: 'The answer is the first symbol of the sequence.',
    note: 'The whole signal sits at the very start, so the network must carry it across every later step. The longer the sequence, the harder it is.',
    vocab: 4,
    classes: 4,
    symbols: ['A', 'B', 'C', 'D'],
    classSymbols: ['A', 'B', 'C', 'D'],
    minLength: 3,
    maxLength: 40,
  },
  marker: {
    label: 'After the marker',
    rule: 'The answer is the symbol that comes right after the ★.',
    note: 'The important symbol can be anywhere, and only a network that decides what to store (and when) can pick it out from the noise.',
    vocab: 5,
    classes: 4,
    symbols: ['A', 'B', 'C', 'D', '★'],
    classSymbols: ['A', 'B', 'C', 'D'],
    minLength: 4,
    maxLength: 40,
  },
  parity: {
    label: 'Even or odd',
    rule: 'The answer is whether the number of 1s is even or odd.',
    note: 'Every single token matters and the state must flip back and forth: a memory that only accumulates cannot do it.',
    vocab: 2,
    classes: 2,
    symbols: ['0', '1'],
    classSymbols: ['even', 'odd'],
    minLength: 2,
    maxLength: 24,
  },
};

export interface SequenceSet {
  length: number;
  count: number;
  /** `count × length` token ids. */
  tokens: Int32Array;
  labels: Uint8Array;
}

/** The label of a sequence under a task's rule. */
export function labelOf(task: TaskName, tokens: ArrayLike<number>): number {
  switch (task) {
    case 'first':
      return tokens[0] as number;
    case 'marker': {
      for (let i = 0; i < tokens.length - 1; i++)
        if (tokens[i] === 4) return tokens[i + 1] as number;
      throw new RangeError('A marker sequence needs a ★ that is not the last token');
    }
    case 'parity': {
      let ones = 0;
      for (let i = 0; i < tokens.length; i++) ones += tokens[i] as number;
      return ones % 2;
    }
  }
}

/** One random valid sequence. */
export function randomSequence(task: TaskName, length: number, rng: Rng): Int32Array {
  const info = taskInfo[task];
  if (!Number.isInteger(length) || length < info.minLength || length > info.maxLength) {
    throw new RangeError(`${info.label}: length must be in [${info.minLength}, ${info.maxLength}]`);
  }
  const out = new Int32Array(length);
  if (task === 'marker') {
    for (let i = 0; i < length; i++) out[i] = Math.floor(rng.next() * 4);
    out[Math.floor(rng.next() * (length - 1))] = 4; // never last, so a symbol follows it
  } else {
    for (let i = 0; i < length; i++) out[i] = Math.floor(rng.next() * info.vocab);
  }
  return out;
}

export function generateSequences(
  task: TaskName,
  length: number,
  count: number,
  rng: Rng,
): SequenceSet {
  const tokens = new Int32Array(count * length);
  const labels = new Uint8Array(count);
  for (let n = 0; n < count; n++) {
    const seq = randomSequence(task, length, rng);
    tokens.set(seq, n * length);
    labels[n] = labelOf(task, seq);
  }
  return { length, count, tokens, labels };
}
