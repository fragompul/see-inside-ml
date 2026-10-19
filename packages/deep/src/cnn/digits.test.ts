import { describe, expect, it } from 'vitest';
import { createRng } from '@sim/core';
import { DIGIT_CLASSES, IMAGE_SIZE, SKELETONS, generateDigits, renderDigit } from './digits';

const ink = (image: Float64Array) => image.reduce((a, b) => a + b, 0);

describe('digit rendering', () => {
  it('has a skeleton for every digit', () => {
    expect(SKELETONS).toHaveLength(DIGIT_CLASSES);
    for (const strokes of SKELETONS) {
      expect(strokes.length).toBeGreaterThan(0);
      for (const s of strokes) {
        expect(s.length).toBeGreaterThanOrEqual(3);
        for (const [x, y] of s) {
          expect(x).toBeGreaterThanOrEqual(0);
          expect(x).toBeLessThanOrEqual(1);
          expect(y).toBeGreaterThanOrEqual(0);
          expect(y).toBeLessThanOrEqual(1);
        }
      }
    }
  });

  it('draws pixels in [0, 1] with a meaningful amount of ink', () => {
    const rng = createRng(1);
    for (let d = 0; d < DIGIT_CLASSES; d++) {
      const image = renderDigit(d, rng);
      expect(image).toHaveLength(IMAGE_SIZE * IMAGE_SIZE);
      expect(image.every((v) => v >= 0 && v <= 1)).toBe(true);
      expect(ink(image)).toBeGreaterThan(18);
      expect(ink(image)).toBeLessThan(150);
    }
  });

  it('is deterministic for a seed and varies between draws', () => {
    expect(renderDigit(3, createRng(5))).toEqual(renderDigit(3, createRng(5)));
    const rng = createRng(5);
    expect(renderDigit(3, rng)).not.toEqual(renderDigit(3, rng));
  });

  it('keeps the margin empty in nearly all samples so strokes are not cut off', () => {
    const rng = createRng(2);
    let clean = 0;
    const n = 200;
    for (let k = 0; k < n; k++) {
      const image = renderDigit(k % 10, rng, { jitter: 0.025, tilt: 0.2, noise: 0 });
      let border = 0;
      for (let i = 0; i < IMAGE_SIZE; i++) {
        border = Math.max(
          border,
          image[i] as number,
          image[(IMAGE_SIZE - 1) * IMAGE_SIZE + i] as number,
          image[i * IMAGE_SIZE] as number,
          image[i * IMAGE_SIZE + IMAGE_SIZE - 1] as number,
        );
      }
      if (border < 0.2) clean += 1;
    }
    expect(clean / n).toBeGreaterThan(0.85);
  });

  it('digits look different: 1 uses far less ink than 8, and classes differ on average', () => {
    const rng = createRng(3);
    const mean = (d: number) => {
      const acc = new Float64Array(IMAGE_SIZE * IMAGE_SIZE);
      for (let n = 0; n < 40; n++)
        renderDigit(d, rng).forEach((v, i) => (acc[i] = (acc[i] as number) + v / 40));
      return acc;
    };
    const means = Array.from({ length: 10 }, (_, d) => mean(d));
    expect(ink(means[1] as Float64Array)).toBeLessThan(ink(means[8] as Float64Array) * 0.7);
    for (let a = 0; a < 10; a++) {
      for (let b = a + 1; b < 10; b++) {
        let diff = 0;
        (means[a] as Float64Array).forEach(
          (v, i) => (diff += Math.abs(v - (means[b] as Float64Array)[i]!)),
        );
        expect(diff, `${a} vs ${b}`).toBeGreaterThan(8);
      }
    }
  });

  it('rejects an unknown digit', () => {
    expect(() => renderDigit(10, createRng(1))).toThrow(/Digit/);
  });
});

describe('generateDigits', () => {
  const set = generateDigits(12, createRng(9));

  it('is balanced and shuffled', () => {
    expect(set.count).toBe(120);
    expect(set.images).toHaveLength(120 * IMAGE_SIZE * IMAGE_SIZE);
    const counts = new Array(10).fill(0) as number[];
    set.labels.forEach((l) => (counts[l] = (counts[l] as number) + 1));
    expect(counts.every((c) => c === 12)).toBe(true);
    // not sorted by class
    expect(Array.from(set.labels.slice(0, 12)).some((l, i, a) => i > 0 && l !== a[0])).toBe(true);
  });

  it('is reproducible', () => {
    const again = generateDigits(12, createRng(9));
    expect(again.images).toEqual(set.images);
    expect(again.labels).toEqual(set.labels);
    expect(generateDigits(12, createRng(10)).images).not.toEqual(set.images);
  });
});
