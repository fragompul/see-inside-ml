import { Matrix, Standardizer, type Classifier } from '@sim/core';

/**
 * All monomials of the columns of `x` up to `degree`, without the constant term, in scikit-learn's
 * `PolynomialFeatures` order: for features (a, b) and degree 2 the columns are a, b, a², ab, b².
 */
export function polynomialExpand(x: Matrix, degree: number): Matrix {
  const combos: number[][] = [];
  const build = (start: number, left: number, current: number[]) => {
    if (left === 0) {
      combos.push([...current]);
      return;
    }
    for (let j = start; j < x.cols; j++) build(j, left - 1, [...current, j]);
  };
  for (let d = 1; d <= degree; d++) build(0, d, []);
  const out = new Matrix(x.rows, combos.length);
  for (let i = 0; i < x.rows; i++)
    combos.forEach((combo, c) => {
      let v = 1;
      for (const j of combo) v *= x.get(i, j);
      out.set(i, c, v);
    });
  return out;
}

/** Number of columns `polynomialExpand` produces. */
export function polynomialSize(features: number, degree: number): number {
  let total = 0;
  for (let d = 1; d <= degree; d++) {
    // multisets of size d from `features` items
    let num = 1;
    for (let i = 1; i <= d; i++) num = (num * (features + i - 1)) / i;
    total += num;
  }
  return total;
}

/**
 * Feeds a classifier polynomial features of its inputs (standardised, so the penalty and the
 * optimiser see comparable columns). A linear model on these is a curved model on the originals,
 * which is how logistic regression solves XOR and circles.
 */
export class PolynomialFeatureClassifier implements Classifier {
  private scaler = new Standardizer();

  constructor(
    readonly inner: Classifier,
    readonly degree: number,
  ) {}

  get classes(): number {
    return this.inner.classes;
  }

  private map(x: Matrix): Matrix {
    return this.scaler.transform(polynomialExpand(x, this.degree));
  }

  fit(x: Matrix, y: ArrayLike<number>): this {
    const expanded = polynomialExpand(x, this.degree);
    this.inner.fit(this.scaler.fit(expanded).transform(expanded), y);
    return this;
  }

  predict(x: Matrix): Int32Array {
    return this.inner.predict(this.map(x));
  }

  predictProba(x: Matrix): Matrix {
    if (!this.inner.predictProba) throw new Error('The wrapped classifier has no probabilities');
    return this.inner.predictProba(this.map(x));
  }
}
