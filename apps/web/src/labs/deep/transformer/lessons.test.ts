import { readFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { createRng } from '@sim/core';
import {
  createTransformer,
  generate,
  nextTokenDistribution,
  type SamplingOptions,
  type TokenizerData,
  type TransformerManifest,
} from '@sim/deep';
import { describe, expect, it } from 'vitest';
import { LESSONS } from './lessons';
import { DEFAULT_SAMPLING } from './store';

const dir = resolve(
  dirname(fileURLToPath(import.meta.url)),
  '../../../../../../packages/deep/models/tiny-gpt',
);
const read = (name: string) => readFileSync(resolve(dir, name));
const json = <T>(name: string) => JSON.parse(read(name).toString('utf-8')) as T;
const raw = read('weights.bin');
const { model, tokenizer } = createTransformer(
  json<TransformerManifest>('model.json'),
  raw.buffer.slice(raw.byteOffset, raw.byteOffset + raw.byteLength) as ArrayBuffer,
  json<TokenizerData>('tokenizer.json'),
);

const lesson = (id: string) => {
  const found = LESSONS.find((l) => l.id === id);
  if (!found) throw new Error(`missing lesson ${id}`);
  return found;
};

const distribution = (prompt: string, sampling: SamplingOptions = DEFAULT_SAMPLING) =>
  nextTokenDistribution(model.nextTokenLogits(tokenizer.encode(prompt)), {
    ...sampling,
    temperature: 1, // the claims below are about the model's own probabilities
  });

const prob = (prompt: string, token: string) => {
  const id = tokenizer.encode(token)[0] as number;
  return distribution(prompt).candidates.find((c) => c.id === id)?.prob ?? 0;
};

/** Probability mass on whole verbs that end in "s" (runs, sleeps…), ignoring partial pieces. */
const massOfThirdPersonVerbs = (prompt: string) =>
  distribution(prompt)
    .candidates.filter((c) => {
      const word = tokenizer.tokenString(c.id).trim();
      return word.length >= 3 && word.endsWith('s');
    })
    .reduce((sum, c) => sum + c.prob, 0);

describe('lessons', () => {
  it('are well formed', () => {
    expect(new Set(LESSONS.map((l) => l.id)).size).toBe(LESSONS.length);
    for (const l of LESSONS) {
      const tokens = tokenizer.encode(l.prompt).length;
      expect(tokens, l.id).toBeGreaterThan(0);
      expect(tokens, l.id).toBeLessThanOrEqual(model.config.ctx - 8);
      expect(Math.abs(l.focus), l.id).toBeLessThanOrEqual(tokens);
      expect(l.text.length, l.id).toBeGreaterThan(60);
    }
  });

  // Each lesson text makes a factual claim about the model. Keep them true.
  it('"Anna → She" and "Tom → He": the pronoun follows the name', () => {
    expect(prob(lesson('pronoun-she').prompt, ' She')).toBeGreaterThan(0.9);
    expect(prob(lesson('pronoun-he').prompt, ' He')).toBeGreaterThan(0.9);
  });

  it('two names → "They"', () => {
    expect(prob(lesson('they').prompt, ' They')).toBeGreaterThan(0.9);
  });

  it('after "took" an object is "it"', () => {
    expect(prob(lesson('it').prompt, ' it')).toBeGreaterThan(0.9);
  });

  it('verbs agree with a singular or plural subject', () => {
    expect(massOfThirdPersonVerbs(lesson('singular').prompt)).toBeGreaterThan(0.6);
    expect(massOfThirdPersonVerbs(lesson('plural').prompt)).toBeLessThan(0.15);
  });

  it('attention lessons: the named head really points at the named token', () => {
    const attention = (id: string, keyText: string) => {
      const l = lesson(id);
      const ids = tokenizer.encode(l.prompt);
      const trace = model.forward(ids);
      const key = ids.findIndex((t) => tokenizer.tokenString(t).trim() === keyText);
      expect(key, `${keyText} in ${id}`).toBeGreaterThanOrEqual(0);
      const probs = trace.layers[l.layer as number]?.probs[l.head as number];
      return { weight: probs?.get(ids.length - 1, key) ?? 0, ids };
    };
    expect(attention('where-she-comes-from', 'Anna').weight).toBeGreaterThan(0.8);
    expect(attention('copy', 'cake').weight).toBeGreaterThan(0.6);
    expect(attention('greeting', 'Leo').weight).toBeGreaterThan(0.8);
    expect(prob(lesson('greeting').prompt, ' Leo')).toBeGreaterThan(0.9);
    // ...and the copied word is what the model predicts
    expect(prob(lesson('copy').prompt, ' cake')).toBeGreaterThan(0.8);
  });

  it('temperature 0 is deterministic, temperature 2 flattens the distribution', () => {
    const greedy = lesson('greedy');
    const run = (seed: number) =>
      generate(
        model,
        tokenizer.encode(greedy.prompt),
        8,
        { temperature: 0, topK: 0, topP: 1 },
        createRng(seed),
      ).map((s) => s.chosen);
    expect(run(1)).toEqual(run(99));

    const base = distribution(lesson('hot').prompt);
    const hot = nextTokenDistribution(
      model.nextTokenLogits(tokenizer.encode(lesson('hot').prompt)),
      {
        temperature: 2,
        topK: 0,
        topP: 1,
      },
    );
    expect(hot.entropyBits).toBeGreaterThan(base.entropyBits);
    expect(hot.candidates[0]?.prob).toBeLessThan(base.candidates[0]?.prob as number);
  });
});
