import { DEFAULT_RNN_LAB, RnnSession, validateRnnLab } from '@sim/deep';
import { describe, expect, it } from 'vitest';
import { LESSONS } from './lessons';

describe('lessons', () => {
  it('have unique ids and a text', () => {
    expect(new Set(LESSONS.map((l) => l.id)).size).toBe(LESSONS.length);
    LESSONS.forEach((l) => expect(l.text.length).toBeGreaterThan(40));
  });

  it.each(LESSONS)(
    '$id: config is valid and its claim holds',
    (lesson) => {
      const config = { ...DEFAULT_RNN_LAB, ...lesson.config };
      expect(() => validateRnnLab(config)).not.toThrow();
      const session = new RnnSession(config);
      session.step(lesson.claim.epochs);
      const accuracy = session.history[lesson.claim.epochs - 1]?.testAccuracy as number;
      if (lesson.claim.atLeast !== undefined)
        expect(accuracy).toBeGreaterThanOrEqual(lesson.claim.atLeast);
      if (lesson.claim.atMost !== undefined)
        expect(accuracy).toBeLessThanOrEqual(lesson.claim.atMost);
    },
    60_000,
  );
});
