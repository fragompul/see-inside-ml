import { DEFAULT_DESCENT, validateDescent } from '@sim/deep';
import { describe, expect, it } from 'vitest';
import { LESSONS } from './lessons';

describe('lessons', () => {
  it('have unique ids and a text', () => {
    expect(new Set(LESSONS.map((l) => l.id)).size).toBe(LESSONS.length);
    LESSONS.forEach((l) => expect(l.text.length).toBeGreaterThan(60));
  });

  it.each(LESSONS)('$id: configuration is valid and its claim holds', (lesson) => {
    expect(() => validateDescent({ ...DEFAULT_DESCENT, ...lesson.descent })).not.toThrow();
    expect(lesson.claim.check()).toBe(true);
  });
});
