import { readFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { createTransformer, type TokenizerData, type TransformerManifest } from '@sim/deep';
import { describe, expect, it } from 'vitest';
import { DEFAULT_ABLATION, runAblation } from './ablationCore';
import { ablationExperiments } from './ablationSpec';

const dir = resolve(
  dirname(fileURLToPath(import.meta.url)),
  '../../../../../../packages/deep/models/tiny-gpt',
);
const read = (name: string) => readFileSync(resolve(dir, name));
const json = <T>(name: string) => JSON.parse(read(name).toString('utf-8')) as T;
const raw = read('weights.bin');
const loaded = createTransformer(
  json<TransformerManifest>('model.json'),
  raw.buffer.slice(raw.byteOffset, raw.byteOffset + raw.byteLength) as ArrayBuffer,
  json<TokenizerData>('tokenizer.json'),
);

describe('Head ablation lab', () => {
  for (const exp of ablationExperiments) {
    it(`experiment "${exp.title}" holds`, () => {
      expect(exp.claim(runAblation(loaded, { ...DEFAULT_ABLATION, ...exp.scenario }), loaded)).toBe(
        true,
      );
    });
  }

  it('silencing nothing leaves the loss at the baseline', () => {
    const r = runAblation(loaded, DEFAULT_ABLATION);
    expect(r.chosenLoss).toBe(r.result.baseline);
    expect(r.result.cumulative).toHaveLength(9);
  });
});
