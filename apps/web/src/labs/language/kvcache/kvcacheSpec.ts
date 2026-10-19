import type { ParamSpec } from '@/shared/lab-kit';
import {
  MODELS,
  SHARING_LABEL,
  gigabytes,
  runKv,
  type KvRun,
  type KvScenario,
  type KvSharing,
  type ModelId,
} from './kvcacheCore';

export const kvParams: ParamSpec[] = [
  {
    key: 'model',
    label: 'Model size',
    kind: 'select',
    options: (Object.keys(MODELS) as ModelId[]).map((id) => ({
      value: id,
      label: MODELS[id].label,
    })),
  },
  {
    key: 'sharing',
    label: 'Keys and values',
    kind: 'select',
    options: (Object.keys(SHARING_LABEL) as KvSharing[]).map((id) => ({
      value: id,
      label: SHARING_LABEL[id],
    })),
  },
  {
    key: 'prompt',
    label: 'Prompt length (tokens)',
    kind: 'ladder',
    values: [128, 1024, 8192],
    format: (v) => String(v),
  },
  {
    key: 'steps',
    label: 'Tokens to write',
    kind: 'ladder',
    values: [16, 64, 256, 1024],
    format: (v) => String(v),
  },
];

export interface KvExperiment {
  id: string;
  title: string;
  text: string;
  scenario: Partial<KvScenario>;
  claim: (run: KvRun) => boolean;
}

export const kvExperiments: KvExperiment[] = [
  {
    id: 'same-answer',
    title: 'The cache changes the cost, not the answer',
    text: 'A real miniature decoder is run here both ways. It writes the same words, and the multiply-adds it actually counts equal the formulas used for the big models below.',
    scenario: {},
    claim: (r) => r.verified,
  },
  {
    id: 'speedup',
    title: 'The longer the text, the more the cache saves',
    text: 'Without a cache every new word re-reads the whole text so far. Writing 16 words costs about 16 times as much as with a cache; writing 1,024 costs more than 400 times as much. (A real model has more layers of work to repeat, so this understates the gap.)',
    scenario: { steps: 1024 },
    claim: (r) => r.speedup > 400 && runKv({ ...r.scenario, steps: 16 }).speedup < 20,
  },
  {
    id: 'memory',
    title: 'The price is memory',
    text: 'The cache keeps two vectors per layer for every token. For a 32-layer model of width 4096, a text of 64,000 tokens needs 32 GB in half precision, more than twice the roughly 13 GB taken by the model itself.',
    scenario: {},
    claim: (r) => Math.abs(gigabytes(r.memory.mha[3] as number) - 32) < 1e-9,
  },
  {
    id: 'share',
    title: 'Sharing keys and values shrinks the cache',
    text: 'Grouped-query attention lets four heads share one set of keys and values and cuts the cache to a quarter; multi-query attention shares one set among all 32 heads and cuts it to 1/32, with 2 GB left at 128,000 tokens instead of 64 GB.',
    scenario: { sharing: 'gqa' },
    claim: (r) =>
      r.memory.gqa.every((v, i) => v * 4 === r.memory.mha[i]) &&
      r.memory.mqa.every((v, i) => v * 32 === r.memory.mha[i]) &&
      Math.abs(gigabytes(r.memory.mqa[4] as number) - 2) < 1e-9,
  },
  {
    id: 'long-context',
    title: 'At long range attention is the whole cost',
    text: 'With a cache, a new word costs a fixed amount of projection work plus work proportional to the length of the text. At 1,024 tokens attention is about a tenth of the total; at 128,000 it is over nine tenths.',
    scenario: {},
    claim: (r) => (r.attentionShare[0] as number) < 0.15 && (r.attentionShare[4] as number) > 0.9,
  },
];
