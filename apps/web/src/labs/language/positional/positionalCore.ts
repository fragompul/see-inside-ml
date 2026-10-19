import {
  diagonalSpread,
  meanAtOffsets,
  positionMatrix,
  type PositionMatrix,
  type PositionScheme,
} from '@sim/deep';

export interface PositionalScenario {
  scheme: PositionScheme;
  dim: number;
  length: number;
  /** Sequence length seen in training: a learned table has nothing beyond it. */
  trained: number;
}

export const DEFAULT_POSITIONAL: PositionalScenario = {
  scheme: 'sinusoidal',
  dim: 32,
  length: 48,
  trained: 48,
};

export interface PositionalRun {
  scenario: PositionalScenario;
  matrix: PositionMatrix;
  /** Share of position pairs for which the scheme gives a score. */
  definedShare: number;
  /** Mean score at offsets 1–4, 10–20 and 30 or more. */
  near: number;
  mid: number;
  far: number;
  /** 0 when the score depends only on how far apart two positions are. */
  spread: number;
  /** Mean score at each offset 0, 1, 2, … */
  byOffset: number[];
}

export function runPositional(s: PositionalScenario): PositionalRun {
  const matrix = positionMatrix(s.scheme, s.length, s.dim, s.trained);
  const defined = matrix.values.filter((v) => !Number.isNaN(v)).length;
  // ALiBi only scores the past, so the share is taken over the pairs the model would ever look at
  const possible = s.scheme === 'alibi' ? (s.length * (s.length + 1)) / 2 : s.length * s.length;
  return {
    scenario: s,
    matrix,
    definedShare: defined / possible,
    near: meanAtOffsets(matrix, 1, 4),
    mid: meanAtOffsets(matrix, 10, 20),
    far: meanAtOffsets(matrix, 30, s.length),
    spread: diagonalSpread(matrix),
    byOffset: Array.from({ length: s.length }, (_, k) => meanAtOffsets(matrix, k, k)),
  };
}
