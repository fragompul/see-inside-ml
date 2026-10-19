import { AutoencoderSession, DEFAULT_AE_LAB, validateAutoencoderLab } from '@sim/deep';
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
      const config = { ...DEFAULT_AE_LAB, ...lesson.config };
      expect(() => validateAutoencoderLab(config)).not.toThrow();
      const session = new AutoencoderSession(config);
      session.step(lesson.claim.epochs);
      expect(lesson.claim.check(session)).toBe(true);
    },
    240_000,
  );
});
