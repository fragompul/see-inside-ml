import { Matrix } from '@sim/core';

export interface PcaResult {
  /** n × k coordinates of every row on the principal axes. */
  projection: Matrix;
  /** Variance captured by each axis (eigenvalues of the covariance matrix). */
  variance: number[];
  /** Fraction of the total variance each axis explains. */
  explained: number[];
}

/**
 * Principal component analysis by power iteration with deflation. Dependency-free and fully
 * deterministic (fixed start vector, sign-normalised axes), which is all the embedding explorer
 * needs: a stable 2-D picture of a few hundred high-dimensional vectors.
 */
export function pca(data: Matrix, components = 2, iterations = 200): PcaResult {
  const { rows: n, cols: d } = data;
  if (n < 2) throw new RangeError('PCA needs at least two rows');
  const k = Math.min(components, d);

  const mean = new Float64Array(d);
  for (let i = 0; i < n; i++)
    for (let j = 0; j < d; j++) mean[j] = (mean[j] as number) + data.get(i, j) / n;
  const centered = new Matrix(n, d);
  for (let i = 0; i < n; i++)
    for (let j = 0; j < d; j++) centered.set(i, j, data.get(i, j) - (mean[j] as number));

  // covariance C = XᵀX / n  (d × d)
  const cov = new Float64Array(d * d);
  for (let i = 0; i < n; i++) {
    for (let a = 0; a < d; a++) {
      const xa = centered.get(i, a);
      for (let b = 0; b < d; b++)
        cov[a * d + b] = (cov[a * d + b] as number) + (xa * centered.get(i, b)) / n;
    }
  }
  let totalVariance = 0;
  for (let j = 0; j < d; j++) totalVariance += cov[j * d + j] as number;

  const axes: Float64Array[] = [];
  const variance: number[] = [];
  for (let c = 0; c < k; c++) {
    // deterministic, non-degenerate start vector
    let v = Float64Array.from({ length: d }, (_, j) => 1 + ((j * 7 + c * 3) % 5) * 0.1);
    let eigenvalue = 0;
    for (let it = 0; it < iterations; it++) {
      const w = new Float64Array(d);
      for (let a = 0; a < d; a++) {
        let sum = 0;
        for (let b = 0; b < d; b++) sum += (cov[a * d + b] as number) * (v[b] as number);
        w[a] = sum;
      }
      let norm = 0;
      for (let a = 0; a < d; a++) norm += (w[a] as number) ** 2;
      norm = Math.sqrt(norm);
      if (norm < 1e-300) break;
      for (let a = 0; a < d; a++) w[a] = (w[a] as number) / norm;
      v = w;
      eigenvalue = norm;
    }
    // sign convention: the largest-magnitude loading is positive
    let big = 0;
    for (let a = 1; a < d; a++) if (Math.abs(v[a] as number) > Math.abs(v[big] as number)) big = a;
    if ((v[big] as number) < 0) for (let a = 0; a < d; a++) v[a] = -(v[a] as number);
    axes.push(v);
    variance.push(eigenvalue);
    // deflate: C ← C − λ v vᵀ
    for (let a = 0; a < d; a++) {
      for (let b = 0; b < d; b++) {
        cov[a * d + b] =
          (cov[a * d + b] as number) - eigenvalue * (v[a] as number) * (v[b] as number);
      }
    }
  }

  const projection = new Matrix(n, k);
  for (let i = 0; i < n; i++) {
    for (let c = 0; c < k; c++) {
      let sum = 0;
      for (let j = 0; j < d; j++)
        sum += centered.get(i, j) * ((axes[c] as Float64Array)[j] as number);
      projection.set(i, c, sum);
    }
  }
  return {
    projection,
    variance,
    explained: variance.map((v) => (totalVariance > 0 ? v / totalVariance : 0)),
  };
}

export interface Neighbour {
  index: number;
  similarity: number;
}

/** The `k` rows most similar (cosine) to row `index`, best first, excluding itself. */
export function nearestByCosine(data: Matrix, index: number, k: number): Neighbour[] {
  if (index < 0 || index >= data.rows) throw new RangeError('Row out of range');
  const norms = new Float64Array(data.rows);
  for (let i = 0; i < data.rows; i++) {
    let s = 0;
    for (let j = 0; j < data.cols; j++) s += data.get(i, j) ** 2;
    norms[i] = Math.sqrt(s);
  }
  const out: Neighbour[] = [];
  for (let i = 0; i < data.rows; i++) {
    if (i === index) continue;
    let dot = 0;
    for (let j = 0; j < data.cols; j++) dot += data.get(index, j) * data.get(i, j);
    const denom = (norms[index] as number) * (norms[i] as number);
    out.push({ index: i, similarity: denom > 0 ? dot / denom : 0 });
  }
  return out.sort((a, b) => b.similarity - a.similarity || a.index - b.index).slice(0, k);
}
