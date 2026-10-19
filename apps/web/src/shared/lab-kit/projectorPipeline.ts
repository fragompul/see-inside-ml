import type { Matrix } from '@sim/core';
import { continuity, neighbourPreservation, trustworthiness } from '@sim/classic';
import type { ProjectionData } from './projectionData';

/**
 * The pure part of the dimensionality-reduction labs. An embedding method is anything that turns the
 * data into a stream of 2-D layouts: one-shot methods (PCA, Isomap, LLE) yield once, iterative ones
 * (t-SNE, UMAP, MDS) yield as they improve, so the page can animate them and the tests can run
 * them to the end with the same code.
 */
export interface EmbedFrame {
  embedding: Matrix;
  /** A short status line (iteration, divergence, stress…). */
  info?: string;
  /** 0 … 1 while an iterative method is running. */
  progress?: number;
}

export interface EmbedModel {
  run(x: Matrix): Generator<EmbedFrame, void, undefined>;
}

/** Wraps a method that computes its answer in one go. */
export function oneShot(compute: (x: Matrix) => { embedding: Matrix; info?: string }): EmbedModel {
  return {
    *run(x) {
      yield { ...compute(x), progress: 1 };
    },
  };
}

export interface EmbeddingScores {
  /** Of the 5 nearest neighbours in the embedding, how many are really near? (1 = none are false) */
  trustworthiness: number;
  /** Of the 5 nearest original neighbours, how many are still near in the embedding? */
  continuity: number;
  /** Share of each point's 5 nearest neighbours that are the same in both spaces. */
  preservation: number;
}

export function scoreEmbedding(x: Matrix, embedding: Matrix, k = 5): EmbeddingScores {
  const kk = Math.min(k, Math.floor((x.rows - 2) / 2));
  return {
    trustworthiness: trustworthiness(x, embedding, kk),
    continuity: continuity(x, embedding, kk),
    preservation: neighbourPreservation(x, embedding, kk),
  };
}

export interface ProjectorRun {
  data: ProjectionData;
  embedding: Matrix;
  info: string | undefined;
  scores: EmbeddingScores;
}

/** Runs an embedding method to its last frame and scores the result. */
export function runProjector(model: EmbedModel, data: ProjectionData): ProjectorRun {
  let last: EmbedFrame | null = null;
  for (const frame of model.run(data.x)) last = frame;
  if (!last) throw new Error('The method produced no embedding');
  return {
    data,
    embedding: last.embedding,
    info: last.info,
    scores: scoreEmbedding(data.x, last.embedding),
  };
}
