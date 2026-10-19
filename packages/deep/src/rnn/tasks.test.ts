import { describe, expect, it } from 'vitest';
import { createRng } from '@sim/core';
import { TASK_NAMES, generateSequences, labelOf, randomSequence, taskInfo } from './tasks';

describe('tasks', () => {
  it('label rules', () => {
    expect(labelOf('first', [2, 0, 1])).toBe(2);
    expect(labelOf('marker', [0, 1, 4, 3, 2])).toBe(3);
    expect(labelOf('marker', [4, 1, 2])).toBe(1);
    expect(labelOf('parity', [1, 1, 0, 1])).toBe(1);
    expect(labelOf('parity', [1, 1, 0, 0])).toBe(0);
    expect(() => labelOf('marker', [0, 1, 2, 4])).toThrow(); // a marker that is last has nothing after it
    expect(() => labelOf('marker', [0, 1, 2])).toThrow();
  });

  it.each(TASK_NAMES)('%s: sequences are valid, balanced and reproducible', (task) => {
    const info = taskInfo[task];
    const length = info.minLength + 3;
    const a = generateSequences(task, length, 600, createRng(4));
    const b = generateSequences(task, length, 600, createRng(4));
    expect(a.tokens).toEqual(b.tokens);
    expect(a.labels).toEqual(b.labels);
    expect(a.tokens.every((t) => t >= 0 && t < info.vocab)).toBe(true);
    const counts = new Array<number>(info.classes).fill(0);
    a.labels.forEach((l) => (counts[l] = (counts[l] as number) + 1));
    counts.forEach((c) => expect(c / 600).toBeGreaterThan(0.6 / info.classes));
    for (let n = 0; n < a.count; n++)
      expect(labelOf(task, a.tokens.subarray(n * length, (n + 1) * length))).toBe(a.labels[n]);
  });

  it('marker sequences contain exactly one star and never at the end', () => {
    const rng = createRng(2);
    for (let n = 0; n < 300; n++) {
      const seq = randomSequence('marker', 8, rng);
      expect(seq.filter((t) => t === 4)).toHaveLength(1);
      expect(seq[seq.length - 1]).not.toBe(4);
    }
  });

  it('rejects lengths outside the allowed range', () => {
    expect(() => randomSequence('first', 1, createRng(1))).toThrow(RangeError);
    expect(() => randomSequence('parity', 99, createRng(1))).toThrow(RangeError);
    expect(() => randomSequence('first', 2.5, createRng(1))).toThrow(RangeError);
  });
});
