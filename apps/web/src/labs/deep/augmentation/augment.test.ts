import { describe, expect, it } from 'vitest';
import { DEFAULT_AUGMENT, runAugment } from './augmentCore';
import { augmentExperiments } from './augmentSpec';

describe('Data augmentation lab', () => {
  for (const exp of augmentExperiments) {
    it(`experiment "${exp.title}" holds`, () => {
      expect(exp.claim(runAugment({ ...DEFAULT_AUGMENT, ...exp.scenario }))).toBe(true);
    }, 300_000);
  }

  it('the training set grows by the number of copies', () => {
    expect(runAugment({ ...DEFAULT_AUGMENT, augmentation: 'none' }).examples).toBe(30);
    expect(runAugment({ ...DEFAULT_AUGMENT, augmentation: 'shift', copies: 4 }).examples).toBe(150);
  });
});
