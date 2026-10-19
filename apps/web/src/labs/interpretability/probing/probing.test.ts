import { readFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { createTransformer, type TokenizerData, type TransformerManifest } from '@sim/deep';
import { describe, expect, it } from 'vitest';
import { DEFAULT_PROBING, runProbing, setStoryModelLoader } from './probingCore';
import { probingExperiments } from './probingSpec';

const dir = resolve(
  dirname(fileURLToPath(import.meta.url)),
  '../../../../../../packages/deep/models/tiny-gpt',
);
const read = (name: string) => readFileSync(resolve(dir, name));
setStoryModelLoader(async () => {
  const raw = read('weights.bin');
  return createTransformer(
    JSON.parse(read('model.json').toString('utf-8')) as TransformerManifest,
    raw.buffer.slice(raw.byteOffset, raw.byteOffset + raw.byteLength) as ArrayBuffer,
    JSON.parse(read('tokenizer.json').toString('utf-8')) as TokenizerData,
  );
});

describe('Probing lab', () => {
  for (const exp of probingExperiments) {
    it(`experiment "${exp.title}" holds`, async () => {
      const run = await runProbing({ ...DEFAULT_PROBING, ...exp.scenario });
      expect(await exp.claim(run)).toBe(true);
    }, 200_000);
  }

  it('every task has one score per layer', async () => {
    for (const model of ['digits', 'story'] as const) {
      const r = await runProbing({ model });
      for (const t of r.tasks) expect(t.test).toHaveLength(r.layers.length);
      expect(r.controlTrain).toHaveLength(r.layers.length);
      expect(r.widths).toHaveLength(r.layers.length);
    }
  });
});
