import { describe, expect, it } from 'vitest';
import { CHANNELS, CLASSES, DEFAULT_VIS, runVis } from './visCore';
import { visExperiments } from './visSpec';

describe('Feature visualisation lab', () => {
  for (const exp of visExperiments) {
    it(`experiment "${exp.title}" holds`, () => {
      expect(exp.claim(runVis({ ...DEFAULT_VIS, ...exp.scenario }))).toBe(true);
    }, 300_000);
  }

  it('makes one picture per class or per feature map', () => {
    expect(runVis({ ...DEFAULT_VIS, kind: 'class' }).images).toHaveLength(CLASSES);
    expect(runVis({ ...DEFAULT_VIS, kind: 'channel' }).images).toHaveLength(CHANNELS);
  });
});
