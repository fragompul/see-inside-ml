import { describe, expect, it } from 'vitest';
import { DEFAULT_KV, runKv } from './kvcacheCore';
import { kvExperiments } from './kvcacheSpec';

describe('KV cache lab', () => {
  for (const exp of kvExperiments) {
    it(`experiment "${exp.title}" holds`, () => {
      expect(exp.claim(runKv({ ...DEFAULT_KV, ...exp.scenario }))).toBe(true);
    });
  }
});
