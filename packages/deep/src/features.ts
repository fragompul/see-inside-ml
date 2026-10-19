import { Matrix } from '@sim/core';

export interface FeatureDefinition {
  id: string;
  /** Short display label (plain text; the UI may render it with math typography). */
  label: string;
  /** KaTeX source. */
  formula: string;
  compute: (x1: number, x2: number) => number;
}

/**
 * Hand-crafted input features. Feeding x₁² or sin(x₁) to even a linear model lets it carve
 * non-linear boundaries: the core lesson of feature engineering.
 */
export const FEATURES: readonly FeatureDefinition[] = [
  { id: 'X1', label: 'x₁', formula: 'x_1', compute: (a) => a },
  { id: 'X2', label: 'x₂', formula: 'x_2', compute: (_, b) => b },
  { id: 'X1^2', label: 'x₁²', formula: 'x_1^2', compute: (a) => a * a },
  { id: 'X2^2', label: 'x₂²', formula: 'x_2^2', compute: (_, b) => b * b },
  { id: 'X1*X2', label: 'x₁x₂', formula: 'x_1 x_2', compute: (a, b) => a * b },
  { id: 'sin(X1)', label: 'sin x₁', formula: '\\sin x_1', compute: (a) => Math.sin(a) },
  { id: 'sin(X2)', label: 'sin x₂', formula: '\\sin x_2', compute: (_, b) => Math.sin(b) },
  { id: 'cos(X1)', label: 'cos x₁', formula: '\\cos x_1', compute: (a) => Math.cos(a) },
  { id: 'cos(X2)', label: 'cos x₂', formula: '\\cos x_2', compute: (_, b) => Math.cos(b) },
  {
    id: 'sgn(X1)',
    label: 'sgn x₁',
    formula: '\\operatorname{sgn} x_1',
    compute: (a) => Math.sign(a),
  },
  {
    id: 'sgn(X2)',
    label: 'sgn x₂',
    formula: '\\operatorname{sgn} x_2',
    compute: (_, b) => Math.sign(b),
  },
  {
    id: 'RBF_Gaussian',
    label: 'e^(−r²/4)',
    formula: 'e^{-(x_1^2+x_2^2)/4}',
    compute: (a, b) => Math.exp(-(a * a + b * b) / 4),
  },
];

export type FeatureId = string;

const byId = new Map(FEATURES.map((f) => [f.id, f]));

export function isFeatureId(id: string): boolean {
  return byId.has(id);
}

/** Maps an n×2 coordinate matrix to an n×len(ids) feature matrix. */
export function extractFeatures(X: Matrix, ids: readonly FeatureId[]): Matrix {
  if (X.cols !== 2) throw new RangeError('extractFeatures expects n×2 coordinates');
  const defs = ids.map((id) => {
    const def = byId.get(id);
    if (!def) throw new RangeError(`Unknown feature "${id}"`);
    return def;
  });
  const out = new Matrix(X.rows, defs.length);
  for (let i = 0; i < X.rows; i++) {
    const a = X.get(i, 0);
    const b = X.get(i, 1);
    defs.forEach((def, j) => out.set(i, j, def.compute(a, b)));
  }
  return out;
}
