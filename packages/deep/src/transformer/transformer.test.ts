import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import fixtureJson from '../../fixtures/tiny-gpt-parity.json';
import { createRng } from '@sim/core';
import { generate, generateNext, createTransformer, type LoadedTransformer } from './generate';
import type { TransformerManifest } from './model';
import { BpeTokenizer, END_OF_TEXT, type TokenizerData } from './tokenizer';

/**
 * Parity with tools/transformer (NumPy): same weights, float64 math, so every tensor must match
 * to ~1e-8. Also property tests that must hold for any causal transformer.
 */

type Rows = number[][];
interface LayerFixture {
  ln1: Rows;
  q: Rows[];
  k: Rows[];
  v: Rows[];
  probs: Rows[];
  attnOut: Rows;
  residMid: Rows;
  mlpPre: Rows;
  mlpOut: Rows;
  residOut: Rows;
}
interface CaseFixture {
  prompt: string;
  ids: number[];
  tok: Rows;
  pos: Rows;
  layers: LayerFixture[];
  finalLn: Rows;
  logits: Rows;
}
interface Fixture {
  cases: CaseFixture[];
  tokenizer: { text: string; ids: number[]; decoded: string }[];
  tokenStrings: string[];
}
const fixture = fixtureJson as unknown as Fixture;

/** ASCII-only decoder for the JSON files (the engine has no TextDecoder typings). */
const asciiText = (bytes: Uint8Array): string => {
  let out = '';
  for (const b of bytes) out += String.fromCharCode(b);
  return out;
};

const modelDir = new URL('../../models/tiny-gpt/', import.meta.url);
const readText = (name: string) => asciiText(readFileSync(new URL(name, modelDir)));
const manifest = JSON.parse(readText('model.json')) as TransformerManifest;
const tokenizerData = JSON.parse(readText('tokenizer.json')) as TokenizerData;
const raw = readFileSync(new URL('weights.bin', modelDir));
const weights = raw.buffer.slice(raw.byteOffset, raw.byteOffset + raw.byteLength);
const { model, tokenizer }: LoadedTransformer = createTransformer(manifest, weights, tokenizerData);

const TOL = 1e-8;
function expectClose(actual: ArrayLike<number>, expected: number[], label: string) {
  expect(actual.length, `${label} length`).toBe(expected.length);
  let worst = 0;
  for (let i = 0; i < expected.length; i++) {
    worst = Math.max(worst, Math.abs((actual[i] as number) - (expected[i] as number)));
  }
  if (!(worst <= TOL)) throw new Error(`${label}: max |Δ| = ${worst}`);
}
const flat = (rows: Rows) => rows.flat();

describe('tokenizer parity with tools/transformer/bpe.py', () => {
  it.each(fixture.tokenizer.map((c) => [JSON.stringify(c.text), c] as const))(
    'encodes and decodes %s like Python',
    (_label, c) => {
      expect(tokenizer.encode(c.text)).toEqual(c.ids);
      expect(tokenizer.decode(c.ids)).toBe(c.decoded);
    },
  );

  it('prints every token the same way (including lone bytes of multi-byte characters)', () => {
    expect(tokenizer.vocabSize).toBe(fixture.tokenStrings.length);
    fixture.tokenStrings.forEach((s, id) => expect(tokenizer.tokenString(id)).toBe(s));
  });

  it('round-trips arbitrary text, including unseen words and non-ASCII', () => {
    const rng = createRng(3);
    const alphabet = 'abcXYZ 012.,!?\n\té€\u{1f600}-';
    for (let n = 0; n < 200; n++) {
      let s = '';
      const len = Math.floor(rng.next() * 40);
      for (let i = 0; i < len; i++)
        s += [...alphabet][Math.floor(rng.next() * [...alphabet].length)];
      expect(tokenizer.decode(tokenizer.encode(s)), JSON.stringify(s)).toBe(s);
    }
  });

  it('keeps the leading space as part of a word and treats <|endoftext|> as one token', () => {
    const ids = tokenizer.encode('Anna found a ball.');
    expect(tokenizer.tokenString(ids[1] as number)).toBe(' found');
    expect(tokenizer.encode(`a${END_OF_TEXT}b`)).toEqual([
      ...tokenizer.encode('a'),
      tokenizer.eotId,
      ...tokenizer.encode('b'),
    ]);
  });

  it('detailed encoding and merge steps agree with plain encoding', () => {
    const text = 'Zebras jump over the lamp.';
    expect(tokenizer.encodeDetailed(text).flatMap((g) => g.ids)).toEqual(tokenizer.encode(text));
    for (const word of tokenizer.pretokenize(text)) {
      const steps = tokenizer.mergeSteps(word);
      expect(steps[steps.length - 1]?.ids).toEqual(tokenizer.encode(word));
      steps.slice(1).forEach((step, i) => {
        expect(step.ids.length).toBe((steps[i]?.ids.length as number) - 1);
        expect(step.merged?.into).toBe(step.merged?.into ?? -1);
        expect(tokenizer.tokenString(step.merged?.into as number)).toBe(
          tokenizer.tokenString(step.merged?.left as number) +
            tokenizer.tokenString(step.merged?.right as number),
        );
      });
    }
  });

  it('rejects unknown ids and mismatched model files', () => {
    expect(() => tokenizer.tokenString(99999)).toThrow(/Unknown token/);
    expect(() => new BpeTokenizer({ merges: [], specialTokens: { [END_OF_TEXT]: 7 } })).toThrow(
      /should be id/,
    );
    expect(() =>
      createTransformer(manifest, weights, {
        merges: tokenizerData.merges.slice(0, -1),
        specialTokens: {},
      }),
    ).toThrow(/Tokenizer has/);
    expect(() => new BpeTokenizer({ merges: [[300, 1]], specialTokens: {} })).toThrow(
      /does not exist/,
    );
  });
});

describe('transformer forward pass parity with tools/transformer/gpt.py', () => {
  it('loads the published weights', () => {
    expect(model.config).toEqual(manifest.config);
    expect(model.parameterCount).toBeGreaterThan(50_000);
  });

  describe.each(fixture.cases.map((c) => [c.prompt, c] as const))('prompt %j', (_name, c) => {
    const trace = model.forward(c.ids);
    const { heads } = model.config;

    it('tokenises the prompt identically', () => {
      expect(tokenizer.encode(c.prompt)).toEqual(c.ids);
    });

    it('embeddings match', () => {
      expectClose(trace.tok.data, flat(c.tok), 'token embeddings');
      expectClose(trace.pos.data, flat(c.pos), 'position embeddings');
    });

    it.each(c.layers.map((_, i) => [i] as const))('block %i matches in every sub-step', (i) => {
      const t = trace.layers[i]!;
      const f = c.layers[i]!;
      expectClose(t.ln1.data, flat(f.ln1), `block ${i} ln1`);
      for (let h = 0; h < heads; h++) {
        expectClose(t.q[h]!.data, flat(f.q[h]!), `block ${i} head ${h} q`);
        expectClose(t.k[h]!.data, flat(f.k[h]!), `block ${i} head ${h} k`);
        expectClose(t.v[h]!.data, flat(f.v[h]!), `block ${i} head ${h} v`);
        expectClose(t.probs[h]!.data, flat(f.probs[h]!), `block ${i} head ${h} attention`);
      }
      expectClose(t.attnOut.data, flat(f.attnOut), `block ${i} attention output`);
      expectClose(t.residMid.data, flat(f.residMid), `block ${i} residual (mid)`);
      expectClose(t.mlpPre.data, flat(f.mlpPre), `block ${i} MLP pre-activation`);
      expectClose(t.mlpOut.data, flat(f.mlpOut), `block ${i} MLP output`);
      expectClose(t.residOut.data, flat(f.residOut), `block ${i} residual (out)`);
    });

    it('final LayerNorm and logits match', () => {
      expectClose(trace.finalLn.data, flat(c.finalLn), 'final layer norm');
      expectClose(trace.logits.data, flat(c.logits), 'logits');
    });
  });
});

describe('transformer properties', () => {
  const ids = tokenizer.encode('Mia found a key. She took it to the park.');

  it('attention is causal and every row is a probability distribution', () => {
    const trace = model.forward(ids);
    for (const layer of trace.layers) {
      for (const p of layer.probs) {
        for (let i = 0; i < p.rows; i++) {
          let sum = 0;
          for (let j = 0; j < p.cols; j++) {
            if (j > i) expect(p.get(i, j)).toBe(0);
            else expect(p.get(i, j)).toBeGreaterThanOrEqual(0);
            sum += p.get(i, j);
          }
          expect(sum).toBeCloseTo(1, 12);
        }
      }
    }
  });

  it('the future never leaks into the past: earlier logits ignore later tokens', () => {
    const full = model.forward(ids).logits;
    const prefix = model.forward(ids.slice(0, 7)).logits;
    for (let t = 0; t < 7; t++) {
      for (let j = 0; j < full.cols; j++) expect(prefix.get(t, j)).toBeCloseTo(full.get(t, j), 12);
    }
    const changed = [...ids];
    changed[10] = (changed[10] as number) === 5 ? 6 : 5;
    const other = model.forward(changed).logits;
    for (let t = 0; t < 10; t++) {
      for (let j = 0; j < full.cols; j++) expect(other.get(t, j)).toBeCloseTo(full.get(t, j), 12);
    }
  });

  it('the residual stream is the running sum of every sub-layer output', () => {
    const t = model.forward(ids);
    const first = t.layers[0]!;
    for (let i = 0; i < t.embed.data.length; i++) {
      expect(first.residMid.data[i]).toBeCloseTo(
        (t.embed.data[i] as number) + (first.attnOut.data[i] as number),
        12,
      );
      expect(first.residOut.data[i]).toBeCloseTo(
        (first.residMid.data[i] as number) + (first.mlpOut.data[i] as number),
        12,
      );
    }
  });

  it('validates its input', () => {
    expect(() => model.forward([])).toThrow(/at least one/);
    expect(() => model.forward([model.config.vocab])).toThrow(/Invalid token/);
    expect(() => model.forward([-1])).toThrow(/Invalid token/);
    expect(() => model.forward(Array(model.config.ctx + 1).fill(1))).toThrow(/exceeds/);
  });

  describe('has learned the grammar of its training stories', () => {
    const top = (text: string) => {
      const logits = model.nextTokenLogits(tokenizer.encode(text));
      let best = 0;
      for (let i = 1; i < logits.length; i++)
        if ((logits[i] as number) > (logits[best] as number)) best = i;
      return tokenizer.tokenString(best);
    };

    it('picks the pronoun that matches the name', () => {
      expect(top('Anna found a ball.')).toBe(' She');
      expect(top('Tom found a ball.')).toBe(' He');
    });

    it('says "They" after two names and "it" after "took"', () => {
      expect(top('Anna and Max went to the park.')).toBe(' They');
      expect(top('Eva found a lamp. She took')).toBe(' it');
    });

    it('copies a name or an object from earlier in the story', () => {
      expect(top('Mia and Leo went to the park. Mia said hello to')).toBe(' Leo');
      expect(top('Ben, Sam and Zoe like cakes. Ben brings a')).toBe(' cake');
    });

    it('ends stories: an end-of-text token is likely after a finished story', () => {
      const logits = model.nextTokenLogits(
        tokenizer.encode(
          'Anna found a ball. She took it to the park. At the park, Anna was happy.',
        ),
      );
      const best = logits.reduce((b, v, i, a) => (v > (a[b] as number) ? i : b), 0);
      expect(best).toBe(tokenizer.eotId);
    });
  });
});

describe('generation', () => {
  const prompt = tokenizer.encode('Anna found a ball. She');

  it('is deterministic for a seed and different for another', () => {
    const run = (seed: number) =>
      generate(model, prompt, 12, { temperature: 1, topK: 0, topP: 1 }, createRng(seed)).map(
        (s) => s.chosen,
      );
    expect(run(1)).toEqual(run(1));
    expect(run(1)).not.toEqual(run(2));
  });

  it('greedy decoding is reproducible and yields a coherent continuation', () => {
    const greedy = generate(model, prompt, 10, { temperature: 0, topK: 0, topP: 1 }, createRng(0));
    const text = tokenizer.decode([...prompt, ...greedy.map((s) => s.chosen)]);
    expect(text.startsWith('Anna found a ball. She ')).toBe(true);
    expect(text).toMatch(/She (took|carried|liked) it/);
  });

  it('stops at <|endoftext|> when asked', () => {
    const steps = generate(model, prompt, 60, { temperature: 0, topK: 0, topP: 1 }, createRng(0));
    expect(steps.length).toBeLessThan(60);
    expect(steps[steps.length - 1]?.chosen).toBe(model.eotId);
  });

  it('crops contexts longer than the window instead of failing', () => {
    const long = Array(model.config.ctx + 20).fill(tokenizer.encode(' the')[0] as number);
    const step = generateNext(model, long, { temperature: 1, topK: 0, topP: 1 }, createRng(1));
    expect(step.context).toHaveLength(model.config.ctx);
  });

  it('rejects invalid sampling settings', () => {
    expect(() =>
      generateNext(model, prompt, { temperature: -1, topK: 0, topP: 1 }, createRng(1)),
    ).toThrow(/temperature/);
  });
});
