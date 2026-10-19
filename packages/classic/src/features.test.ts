import { Matrix, createRng, makeClassification, accuracy } from '@sim/core';
import { describe, expect, it } from 'vitest';
import { PolynomialFeatureClassifier, polynomialExpand, polynomialSize } from './features';
import { LogisticRegression } from './logistic';

describe('polynomial feature expansion', () => {
  it('uses scikit-learn’s column order', () => {
    const f = polynomialExpand(Matrix.from([[2, 3]]), 3);
    // a, b, a², ab, b², a³, a²b, ab², b³
    expect(Array.from(f.data)).toEqual([2, 3, 4, 6, 9, 8, 12, 18, 27]);
  });

  it('predicts its own size', () => {
    for (const [d, deg] of [
      [2, 1],
      [2, 3],
      [3, 4],
      [5, 2],
    ] as const) {
      expect(polynomialExpand(Matrix.from([new Array(d).fill(1)]), deg).cols).toBe(
        polynomialSize(d, deg),
      );
    }
  });
});

describe('a linear model on polynomial features', () => {
  const accuracyOn = (kind: 'xor' | 'circles', degree: number) => {
    const t = makeClassification(kind, 200, 0.2, createRng(3));
    const m = new PolynomialFeatureClassifier(new LogisticRegression({ C: 100 }), degree).fit(
      t.x,
      t.y,
    );
    return accuracy(t.y, m.predict(t.x));
  };

  it('cannot separate XOR with straight boundaries but can with degree 2', () => {
    expect(accuracyOn('xor', 1)).toBeLessThan(0.7);
    expect(accuracyOn('xor', 2)).toBeGreaterThan(0.9);
  });

  it('solves circles with degree 2', () => {
    expect(accuracyOn('circles', 1)).toBeLessThan(0.7);
    expect(accuracyOn('circles', 2)).toBeGreaterThan(0.9);
  });
});
