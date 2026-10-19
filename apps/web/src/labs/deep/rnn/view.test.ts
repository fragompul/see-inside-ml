import { createRng } from '@sim/core';
import { RecurrentNet } from '@sim/deep';
import { describe, expect, it } from 'vitest';
import { logPosition, logRange, relevantTokens, stepFacts, stepMean, unitMajor } from './view';

describe('unitMajor', () => {
  it('transposes time-major data and can skip the start state', () => {
    // 3 rows (start + 2 steps) × 2 units
    const timeMajor = [0, 0, 1, 2, 3, 4];
    expect(Array.from(unitMajor(timeMajor, 2, 2, 1))).toEqual([1, 3, 2, 4]);
    expect(Array.from(unitMajor([1, 2, 3, 4], 2, 2))).toEqual([1, 3, 2, 4]);
  });

  it('averages one step', () => {
    expect(stepMean([1, 2, 3, 4], 1, 2)).toBe(3.5);
  });
});

describe('log axis', () => {
  it('covers the data in whole decades and keeps a minimum height', () => {
    expect(logRange([3e-8, 0.5, 1.2])).toEqual({ lo: -8, hi: 1 });
    expect(logRange([0.2, 0.4])).toEqual({ lo: -2, hi: 0 });
    expect(logRange([0, -1, NaN])).toEqual({ lo: -2, hi: 0 });
  });

  it('places values between 0 and 1 and clamps the rest', () => {
    expect(logPosition(1e-4, -8, 0)).toBeCloseTo(0.5);
    expect(logPosition(10, -8, 0)).toBe(1);
    expect(logPosition(1e-12, -8, 0)).toBe(0);
    expect(logPosition(0, -8, 0)).toBe(0);
  });
});

describe('relevantTokens', () => {
  it('points at the tokens the rule depends on', () => {
    expect(relevantTokens('first', [2, 1, 0])).toEqual([0]);
    expect(relevantTokens('parity', [1, 0, 1])).toEqual([0, 1, 2]);
    expect(relevantTokens('marker', [0, 4, 3, 1])).toEqual([1, 2]);
    expect(relevantTokens('marker', [0, 1, 3])).toEqual([]);
  });
});

describe('stepFacts', () => {
  const trace = (cell: 'rnn' | 'lstm' | 'gru') =>
    new RecurrentNet({ cell, hidden: 4, vocab: 4, classes: 2 }, createRng(1)).run([0, 1, 2]);

  it('describes gates per cell type', () => {
    expect(stepFacts(trace('rnn'), 0).gates).toHaveLength(0);
    expect(stepFacts(trace('lstm'), 1).gates.map((g) => g.name)).toEqual([
      'forget',
      'input',
      'output',
    ]);
    expect(stepFacts(trace('gru'), 2).gates.map((g) => g.name)).toEqual(['update', 'reset']);
  });

  it('reports the token, gate means in [0, 1] and a non-negative change', () => {
    const facts = stepFacts(trace('lstm'), 2);
    expect(facts.token).toBe(2);
    facts.gates.forEach((g) => expect(g.mean >= 0 && g.mean <= 1).toBe(true));
    expect(facts.change).toBeGreaterThan(0);
  });
});
