import {
  cacheBytes,
  decodeText,
  generationMacs,
  makeDecoder,
  stepMacs,
  type CostShape,
} from '@sim/deep';

export type ModelId = 'small' | 'seven' | 'large';
export type KvSharing = 'mha' | 'gqa' | 'mqa';

export interface ModelSize {
  label: string;
  layers: number;
  d: number;
  heads: number;
}

export const MODELS: Record<ModelId, ModelSize> = {
  small: { label: '12 layers, width 768', layers: 12, d: 768, heads: 12 },
  seven: { label: '32 layers, width 4096', layers: 32, d: 4096, heads: 32 },
  large: { label: '80 layers, width 8192', layers: 80, d: 8192, heads: 64 },
};

export const SHARING_LABEL: Record<KvSharing, string> = {
  mha: 'Every head has its own keys and values',
  gqa: 'Grouped: four heads share one set',
  mqa: 'Multi-query: all heads share one set',
};

export const VOCAB = 32000;
export const BYTES = 2; // half precision

export const keyValueWidth = (m: ModelSize, sharing: KvSharing): number =>
  sharing === 'mha' ? m.d : sharing === 'gqa' ? m.d / 4 : m.d / m.heads;

export interface KvScenario {
  model: ModelId;
  sharing: KvSharing;
  prompt: number;
  steps: number;
}

export const DEFAULT_KV: KvScenario = { model: 'seven', sharing: 'mha', prompt: 1024, steps: 256 };

export const CONTEXTS = [1024, 4096, 16384, 65536, 131072];

export interface KvRun {
  scenario: KvScenario;
  /** A real tiny decoder, run both ways, wrote the same text and its counted work matched the cost model. */
  verified: boolean;
  withCache: number;
  without: number;
  speedup: number;
  /** Bytes of cache at the end of the text. */
  cacheEnd: number;
  /** Cumulative multiply-adds after each of 1..steps new tokens, sampled on a grid. */
  grid: number[];
  curveWith: number[];
  curveWithout: number[];
  /** Cache bytes at each of CONTEXTS for the three sharing schemes. */
  memory: Record<KvSharing, number[]>;
  /** Share of the work spent in attention (rather than projections) for one new token at each context, with a cache. */
  attentionShare: number[];
}

const totalMacs = (m: ModelSize, dkv: number, prompt: number, steps: number, cached: boolean) => {
  const shape: CostShape = { d: m.d, dkv, vocab: 0 };
  const t = generationMacs(shape, prompt, steps, cached);
  return m.layers * (t.projection + t.attention) + steps * VOCAB * m.d;
};

let verified: boolean | null = null;
/** Runs a tiny decoder with and without a cache: same words out, and the counted work equals the formulas. */
export function verifyCache(): boolean {
  if (verified === null) {
    const model = makeDecoder(16, 4, 20, 3);
    const prompt = [1, 5, 9, 3, 7, 2];
    const a = decodeText(model, prompt, 20, true);
    const b = decodeText(model, prompt, 20, false);
    const same = a.tokens.every((t, i) => t === b.tokens[i]);
    const model_a = generationMacs(model.shape, prompt.length, 20, true);
    const model_b = generationMacs(model.shape, prompt.length, 20, false);
    verified =
      same &&
      a.macs.projection === model_a.projection &&
      a.macs.attention === model_a.attention &&
      b.macs.projection === model_b.projection &&
      b.macs.attention === model_b.attention;
  }
  return verified;
}

export function runKv(s: KvScenario): KvRun {
  const m = MODELS[s.model];
  const dkv = keyValueWidth(m, s.sharing);
  const withCache = totalMacs(m, dkv, s.prompt, s.steps, true);
  const without = totalMacs(m, dkv, s.prompt, s.steps, false);
  const points = 24;
  const grid = Array.from({ length: points }, (_, i) =>
    Math.max(1, Math.round(((i + 1) / points) * s.steps)),
  );
  const memory = Object.fromEntries(
    (['mha', 'gqa', 'mqa'] as KvSharing[]).map((k) => [
      k,
      CONTEXTS.map((c) => cacheBytes(m.layers, keyValueWidth(m, k), c, BYTES)),
    ]),
  ) as Record<KvSharing, number[]>;
  const attentionShare = CONTEXTS.map((c) => {
    const one = stepMacs({ d: m.d, dkv, vocab: 0 }, c, 1);
    return one.attention / (one.attention + one.projection);
  });
  return {
    scenario: s,
    verified: verifyCache(),
    withCache,
    without,
    speedup: without / withCache,
    cacheEnd: cacheBytes(m.layers, dkv, s.prompt + s.steps, BYTES),
    grid,
    curveWith: grid.map((n) => totalMacs(m, dkv, s.prompt, n, true)),
    curveWithout: grid.map((n) => totalMacs(m, dkv, s.prompt, n, false)),
    memory,
    attentionShare,
  };
}

export const gigabytes = (b: number): number => b / 1024 ** 3;
