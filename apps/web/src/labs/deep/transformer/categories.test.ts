import { describe, expect, it } from 'vitest';
import { categoryOf } from './categories';

describe('categoryOf', () => {
  it.each([
    ['Anna', 'person'],
    [' Tom', 'person'],
    [' cat', 'animal'],
    [' mice', 'animal'],
    [' ball', 'object'],
    [' park', 'place'],
    [' found', 'verb'],
    [' carried', 'verb'],
    [' happy', 'adjective'],
    [' She', 'pronoun'],
    [' the', 'function'],
    ['.', 'punctuation'],
    [',', 'punctuation'],
    [' zzz', 'other'],
    [' ', 'other'],
  ])('%j is %s', (token, category) => {
    expect(categoryOf(token)).toBe(category);
  });
});
