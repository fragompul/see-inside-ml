import { describe, expect, it } from 'vitest';
import { DEFAULT_LORA, runLora } from './loraCore';
import { loraExperiments } from './loraSpec';

describe('LoRA lab', () => {
  for (const exp of loraExperiments) {
    it(`experiment "${exp.title}" holds`, () => {
      expect(exp.claim(runLora({ ...DEFAULT_LORA, ...exp.scenario }))).toBe(true);
    }, 180_000);
  }

  it('the number of trainable parameters follows the rank and the training loss falls', () => {
    const r2 = runLora({ ...DEFAULT_LORA, rank: 2, epochs: 20 });
    const r4 = runLora({ ...DEFAULT_LORA, rank: 4, epochs: 20 });
    expect(r4.trainable).toBe(2 * r2.trainable);
    expect(r4.trainLoss[r4.trainLoss.length - 1]!).toBeLessThan(r4.trainLoss[0]!);
  });
});
