import {
  ablateHeads,
  corpusLoss,
  headKey,
  headPatterns,
  type AblationResult,
  type HeadPattern,
  type LoadedTransformer,
} from '@sim/deep';
import { STORIES } from './ablationTexts';

export type AblationMode = 'zero' | 'mean';

export interface AblationScenario {
  mode: AblationMode;
  /** Heads silenced by hand, as "layer.head". */
  silenced: string[];
  prompt: string;
}

export const DEFAULT_ABLATION: AblationScenario = {
  mode: 'zero',
  silenced: [],
  prompt: 'Anna found a ball.',
};

export interface AblationRun {
  scenario: AblationScenario;
  result: AblationResult;
  patterns: HeadPattern[][];
  /** Loss on the stories with the chosen heads silenced. */
  chosenLoss: number;
  /** Probability of the most likely next tokens after the prompt, with the chosen heads silenced. */
  next: { token: string; p: number }[];
  tokens: number;
}

const cache = new Map<string, { result: AblationResult; patterns: HeadPattern[][] }>();

export function storyIds(loaded: LoadedTransformer): number[][] {
  return STORIES.map((s) => loaded.tokenizer.encode(s));
}

export function runAblation(loaded: LoadedTransformer, s: AblationScenario): AblationRun {
  const ids = storyIds(loaded);
  let shared = cache.get(s.mode);
  if (!shared) {
    shared = {
      result: ablateHeads(loaded.model, ids, s.mode),
      patterns: headPatterns(ids.map((x) => loaded.model.forward(x))),
    };
    cache.set(s.mode, shared);
  }
  const options = { ablate: new Set(s.silenced), ablateMode: s.mode };
  const prompt = loaded.tokenizer.encode(s.prompt);
  const { logits } = loaded.model.forward(prompt, options);
  const last = Array.from({ length: logits.cols }, (_, v) => logits.get(logits.rows - 1, v));
  const top = Math.max(...last);
  const z = last.reduce((a, v) => a + Math.exp(v - top), 0);
  const ranked = last
    .map((v, i) => ({ token: loaded.tokenizer.tokenString(i), p: Math.exp(v - top) / z }))
    .sort((a, b) => b.p - a.p)
    .slice(0, 4);
  return {
    scenario: s,
    result: shared.result,
    patterns: shared.patterns,
    chosenLoss: s.silenced.length ? corpusLoss(loaded.model, ids, options) : shared.result.baseline,
    next: ranked,
    tokens: ids.reduce((a, x) => a + x.length - 1, 0),
  };
}

export { headKey };
