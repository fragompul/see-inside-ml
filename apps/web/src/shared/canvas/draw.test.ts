import { createRng } from '@sim/core';
import { IMAGE_SIZE, renderDigit } from '@sim/deep';
import { describe, expect, it } from 'vitest';
import { inkStats, normalizeDrawing, paint, stroke } from './draw';

const N = IMAGE_SIZE;
const blank = () => new Float64Array(N * N);
const at = (img: Float64Array, x: number, y: number) => img[y * N + x] as number;

describe('brush', () => {
  it('is brightest at the centre, fades with distance and stays in [0, 1]', () => {
    const img = blank();
    paint(img, 8, 8);
    expect(at(img, 7, 7)).toBeGreaterThan(0.9);
    expect(at(img, 7, 7)).toBeGreaterThan(at(img, 9, 7));
    expect(at(img, 7, 7)).toBeGreaterThan(at(img, 11, 7));
    expect(at(img, 14, 14)).toBe(0);
    expect(img.every((v) => v >= 0 && v <= 1)).toBe(true);
  });

  it('never erases: painting twice keeps the brightest value', () => {
    const img = blank();
    paint(img, 8, 8);
    const once = Float64Array.from(img);
    paint(img, 8.4, 8.2);
    img.forEach((v, i) => expect(v).toBeGreaterThanOrEqual(once[i] as number));
  });

  it('copes with positions at and beyond the border', () => {
    const img = blank();
    expect(() => paint(img, -3, -3)).not.toThrow();
    expect(() => paint(img, 40, 8)).not.toThrow();
    paint(img, 0, 0);
    expect(at(img, 0, 0)).toBeGreaterThan(0.5);
  });
});

describe('stroke', () => {
  it('leaves no gaps along a long fast movement', () => {
    const img = blank();
    stroke(img, [2, 2], [13, 13]);
    for (let i = 2; i <= 13; i++) expect(at(img, i, i)).toBeGreaterThan(0.5);
  });

  it('a stroke of zero length is a dot', () => {
    const img = blank();
    stroke(img, [5, 5], [5, 5]);
    expect(at(img, 4, 4)).toBeGreaterThan(0.5);
  });
});

describe('inkStats', () => {
  it('reports nothing for an empty image', () => {
    expect(inkStats(blank())).toEqual({ mass: 0, centroid: null, box: null });
  });

  it('finds the centroid and the bounding box of a drawing', () => {
    const img = blank();
    stroke(img, [3, 4], [9, 4]);
    const { centroid, box, mass } = inkStats(img);
    expect(mass).toBeGreaterThan(5);
    // pixel-space coordinates: the stroke runs along y = 4 from x = 3 to x = 9
    expect(centroid?.[0]).toBeCloseTo(6, 0);
    expect(centroid?.[1]).toBeCloseTo(4, 0);
    expect(box?.[1]).toBeLessThanOrEqual(4);
    expect(box?.[3]).toBeGreaterThanOrEqual(4);
  });
});

describe('normalizeDrawing', () => {
  it('moves an off-centre drawing to the middle', () => {
    const img = blank();
    stroke(img, [10, 9], [13, 13]);
    stroke(img, [13, 9], [10, 13]); // a small "x" in the bottom-right corner
    const { centroid } = inkStats(normalizeDrawing(img));
    expect(centroid?.[0]).toBeCloseTo(N / 2, 0);
    expect(centroid?.[1]).toBeCloseTo(N / 2, 0);
  });

  it('scales a small drawing up to fill the training box, but never shrinks a fitting one much', () => {
    const small = blank();
    stroke(small, [6, 6], [9, 9]);
    const grown = inkStats(normalizeDrawing(small)).box as number[];
    expect((grown[2] as number) - (grown[0] as number)).toBeGreaterThan(4);

    const digit = renderDigit(8, createRng(1), { jitter: 0.02, tilt: 0.1, noise: 0 });
    const before = inkStats(digit).box as number[];
    const after = inkStats(normalizeDrawing(digit)).box as number[];
    const heightBefore = (before[3] as number) - (before[1] as number);
    const heightAfter = (after[3] as number) - (after[1] as number);
    expect(heightAfter).toBeGreaterThanOrEqual(heightBefore - 1.5);
    expect(heightAfter).toBeLessThanOrEqual(N - 2);
  });

  it('keeps values in range, returns a copy for empty input, and does not mutate its argument', () => {
    const img = blank();
    stroke(img, [3, 3], [12, 12]);
    const snapshot = Float64Array.from(img);
    const out = normalizeDrawing(img);
    expect(img).toEqual(snapshot);
    expect(out.every((v) => v >= 0 && v <= 1)).toBe(true);
    const empty = blank();
    const copy = normalizeDrawing(empty);
    expect(copy).toEqual(empty);
    expect(copy).not.toBe(empty);
  });
});
