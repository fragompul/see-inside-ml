import { DEFAULT_NLP_LAB, NlpSession, validateNlpLab } from '@sim/deep';
import { describe, expect, it } from 'vitest';
import { LESSONS } from './lessons';

describe('lessons', () => {
  it('have unique ids and a text', () => {
    expect(new Set(LESSONS.map((l) => l.id)).size).toBe(LESSONS.length);
    LESSONS.forEach((l) => expect(l.text.length).toBeGreaterThan(60));
  });

  it.each(LESSONS)(
    '$id: config is valid and its claim holds',
    (lesson) => {
      const config = { ...DEFAULT_NLP_LAB, ...lesson.config };
      expect(() => validateNlpLab(config)).not.toThrow();
      const session = new NlpSession(config);
      session.train(lesson.claim.steps);
      expect(lesson.claim.check(session)).toBe(true);
    },
    300_000,
  );
});
