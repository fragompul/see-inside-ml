import { describe, expect, it } from 'vitest';
import {
  alibiSlope,
  diagonalSpread,
  dot,
  meanAtOffsets,
  positionMatrix,
  rotate,
  sinusoidalEncoding,
} from './positional';

describe('position encodings', () => {
  it('sinusoidal codes follow the definition and have a constant norm', () => {
    const e = sinusoidalEncoding(5, 8);
    expect(e[0]).toBeCloseTo(Math.sin(5), 12);
    expect(e[1]).toBeCloseTo(Math.cos(5), 12);
    expect(e[2]).toBeCloseTo(Math.sin(5 * 10000 ** (-2 / 8)), 12);
    expect(dot(e, e)).toBeCloseTo(4, 12);
  });

  it('the dot product of two sinusoidal codes depends only on the offset', () => {
    expect(diagonalSpread(positionMatrix('sinusoidal', 40, 32, 40))).toBeLessThan(1e-9);
  });

  it('rotation preserves length, and rotated dot products depend only on the offset', () => {
    const x = [0.3, -1.2, 0.8, 0.5, -0.4, 2];
    expect(dot(rotate(x, 7), rotate(x, 7))).toBeCloseTo(dot(x, x), 12);
    const q = [1, 2, -1, 0.5];
    const k = [0.2, -0.7, 1.5, 0.9];
    expect(dot(rotate(q, 10), rotate(k, 4))).toBeCloseTo(dot(rotate(q, 16), rotate(k, 10)), 10);
  });

  it('with identical content RoPE and sinusoidal codes give the same position scores', () => {
    const a = positionMatrix('sinusoidal', 20, 16, 20);
    const b = positionMatrix('rope', 20, 16, 20);
    for (let i = 0; i < a.values.length; i++) expect(a.values[i]).toBeCloseTo(b.values[i]!, 10);
  });

  it('ALiBi is zero on the diagonal, linear in the distance and causal', () => {
    const m = positionMatrix('alibi', 10, 8, 10);
    const slope = alibiSlope(0, 8);
    expect(slope).toBeCloseTo(0.5, 12);
    expect(m.values[5 * 10 + 5]).toBeCloseTo(0, 12);
    expect(m.values[5 * 10 + 2]).toBeCloseTo(-3 * slope, 12);
    expect(m.values[2 * 10 + 5]).toBeNaN();
    expect(diagonalSpread(m)).toBeLessThan(1e-12);
  });

  it('a learned table is undefined beyond its trained length', () => {
    const m = positionMatrix('learned', 12, 16, 8);
    expect(m.values[3 * 12 + 3]).not.toBeNaN();
    expect(m.values[9 * 12 + 9]).toBeNaN();
    expect(m.values[3 * 12 + 9]).toBeNaN();
    expect(Number.isNaN(meanAtOffsets(m, 0, 11))).toBe(false);
  });
});
