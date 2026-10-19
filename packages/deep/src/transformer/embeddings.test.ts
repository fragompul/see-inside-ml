import { describe, expect, it } from 'vitest';
import { Matrix, createRng } from '@sim/core';
import { nearestByCosine, pca } from './embeddings';

/** Points with strong variance along known axes inside a higher-dimensional space. */
function synthetic(n = 300, d = 6, seed = 1): Matrix {
  const rng = createRng(seed);
  const m = new Matrix(n, d);
  for (let i = 0; i < n; i++) {
    const a = rng.normal() * 5; // dominant direction: axis 0 + 0.5·axis 1
    const b = rng.normal() * 2; // second direction: axis 2
    m.set(i, 0, a + 3);
    m.set(i, 1, a * 0.5 - 1);
    m.set(i, 2, b);
    for (let j = 3; j < d; j++) m.set(i, j, rng.normal() * 0.1);
  }
  return m;
}

describe('pca', () => {
  const data = synthetic();
  const result = pca(data, 2);

  it('finds the dominant directions and orders them by variance', () => {
    expect(result.variance[0]).toBeGreaterThan(result.variance[1] as number);
    // the first axis carries (1, 0.5) / |(1, 0.5)|: variance ≈ 25 · (1 + 0.25)
    expect(result.variance[0]).toBeCloseTo(25 * 1.25, -1);
    expect(result.variance[1]).toBeCloseTo(4, 0);
    expect((result.explained[0] as number) + (result.explained[1] as number)).toBeGreaterThan(0.99);
  });

  it('projects onto uncorrelated, zero-mean axes whose variance matches the eigenvalues', () => {
    const n = data.rows;
    let cross = 0;
    let meanX = 0;
    let meanY = 0;
    let varX = 0;
    let varY = 0;
    for (let i = 0; i < n; i++) {
      const x = result.projection.get(i, 0);
      const y = result.projection.get(i, 1);
      meanX += x / n;
      meanY += y / n;
      varX += (x * x) / n;
      varY += (y * y) / n;
      cross += (x * y) / n;
    }
    expect(Math.abs(meanX)).toBeLessThan(1e-9);
    expect(Math.abs(meanY)).toBeLessThan(1e-9);
    expect(Math.abs(cross)).toBeLessThan(1e-6);
    expect(varX).toBeCloseTo(result.variance[0] as number, 6);
    expect(varY).toBeCloseTo(result.variance[1] as number, 6);
  });

  it('is deterministic', () => {
    expect(pca(synthetic(), 2).projection.data).toEqual(result.projection.data);
  });

  it('caps the number of components at the dimension and validates input', () => {
    expect(pca(synthetic(50, 3), 5).projection.cols).toBe(3);
    expect(() => pca(new Matrix(1, 3))).toThrow(/two rows/);
  });
});

describe('nearestByCosine', () => {
  const m = Matrix.from([
    [1, 0],
    [0.9, 0.1],
    [0, 1],
    [-1, 0],
    [0, 0],
  ]);

  it('ranks by cosine similarity and excludes the query row', () => {
    const out = nearestByCosine(m, 0, 3);
    expect(out.map((n) => n.index)).toEqual([1, 2, 4]);
    expect(out[0]?.similarity).toBeGreaterThan(0.99);
    expect(out.find((n) => n.index === 3)).toBeUndefined();
    expect(nearestByCosine(m, 0, 4).at(-1)?.similarity).toBeCloseTo(-1, 12);
  });

  it('treats a zero vector as similar to nothing and rejects a bad row', () => {
    expect(nearestByCosine(m, 4, 1)[0]?.similarity).toBe(0);
    expect(() => nearestByCosine(m, 9, 1)).toThrow(/out of range/);
  });
});
