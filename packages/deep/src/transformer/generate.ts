import { type Rng } from '@sim/core';
import { BpeTokenizer, type TokenizerData } from './tokenizer';
import { TransformerModel, type TransformerManifest } from './model';
import {
  nextTokenDistribution,
  sampleToken,
  validateSampling,
  type NextTokenDistribution,
  type SamplingOptions,
} from './sampling';

export interface LoadedTransformer {
  model: TransformerModel;
  tokenizer: BpeTokenizer;
}

/** Wires model weights and tokenizer together, checking that they belong to each other. */
export function createTransformer(
  manifest: TransformerManifest,
  weights: ArrayBuffer,
  tokenizerData: TokenizerData,
): LoadedTransformer {
  const tokenizer = new BpeTokenizer(tokenizerData);
  const model = TransformerModel.fromBuffer(manifest, weights);
  if (tokenizer.vocabSize !== model.config.vocab) {
    throw new RangeError(
      `Tokenizer has ${tokenizer.vocabSize} tokens but the model expects ${model.config.vocab}`,
    );
  }
  if (tokenizer.eotId !== manifest.eotId) throw new RangeError('End-of-text id mismatch');
  return { model, tokenizer };
}

export interface GenerationStep {
  /** Context the model saw (already cropped to its window). */
  context: number[];
  distribution: NextTokenDistribution;
  chosen: number;
}

/** One decoding step: run the model on the (cropped) context and sample the next token. */
export function generateNext(
  model: TransformerModel,
  ids: readonly number[],
  sampling: SamplingOptions,
  rng: Rng,
): GenerationStep {
  validateSampling(sampling);
  const context = ids.slice(-model.config.ctx);
  const distribution = nextTokenDistribution(model.nextTokenLogits(context), sampling);
  return { context, distribution, chosen: sampleToken(distribution, rng) };
}

/** Generates up to `maxNew` tokens, stopping early at <|endoftext|> when asked to. */
export function generate(
  model: TransformerModel,
  promptIds: readonly number[],
  maxNew: number,
  sampling: SamplingOptions,
  rng: Rng,
  stopAtEnd = true,
): GenerationStep[] {
  const ids = [...promptIds];
  const steps: GenerationStep[] = [];
  for (let i = 0; i < maxNew; i++) {
    const step = generateNext(model, ids, sampling, rng);
    steps.push(step);
    ids.push(step.chosen);
    if (stopAtEnd && step.chosen === model.eotId) break;
  }
  return steps;
}
