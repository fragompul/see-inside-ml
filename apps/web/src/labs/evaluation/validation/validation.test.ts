import { createRng } from '@sim/core';
import { describe, expect, it } from 'vitest';
import { DEFAULT_VALIDATION, kFolds, runValidation, shuffled } from './validationCore';
import { validationExperiments } from './validationSpec';

describe('validation lab', () => {
  for (const exp of validationExperiments) {
    it(`experiment "${exp.title}" holds`, () => {
      expect(exp.claim(runValidation({ ...DEFAULT_VALIDATION, ...exp.scenario }))).toBe(true);
    }, 60_000);
  }

  it('folds partition the rows exactly and differ in size by at most one', () => {
    const folds = kFolds(23, 5, createRng(1));
    expect(folds.flat().sort((a, b) => a - b)).toEqual(Array.from({ length: 23 }, (_, i) => i));
    const sizes = folds.map((f) => f.length);
    expect(Math.max(...sizes) - Math.min(...sizes)).toBeLessThanOrEqual(1);
  });

  it('a shuffle is a permutation', () => {
    expect(shuffled(10, createRng(3)).sort((a, b) => a - b)).toEqual(
      Array.from({ length: 10 }, (_, i) => i),
    );
  });
});
