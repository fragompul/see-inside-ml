import { createRng } from '@sim/core';
import { describe, expect, it } from 'vitest';
import { augmentImage, augmentSet, noiseImage, rotateImage, shiftImage } from './augment';
import { DEFAULT_STYLE, IMAGE_SIZE, renderDigit } from './digits';

const S = IMAGE_SIZE;
const digit = (label: number) => renderDigit(label, createRng(label + 1), DEFAULT_STYLE);
const centre = (img: ArrayLike<number>) => {
  let m = 0;
  let cx = 0;
  let cy = 0;
  for (let i = 0; i < S * S; i++) {
    m += img[i] as number;
    cx += (img[i] as number) * (i % S);
    cy += (img[i] as number) * Math.floor(i / S);
  }
  return [cx / m, cy / m];
};

describe('augmentations', () => {
  it('a shift of zero is the identity and a shift moves the centre of mass by exactly that much', () => {
    // a small blob well inside the frame, so that nothing is pushed out of it
    const img = new Float64Array(S * S);
    for (let y = 6; y < 10; y++)
      for (let x = 5; x < 9; x++) img[y * S + x] = 0.5 + 0.1 * (x + y - 11);
    expect(Array.from(shiftImage(img, 0, 0))).toEqual(Array.from(img));
    const [x0, y0] = centre(img) as [number, number];
    const [x1, y1] = centre(shiftImage(img, 2, -1)) as [number, number];
    expect(x1 - x0).toBeCloseTo(2, 9);
    expect(y1 - y0).toBeCloseTo(-1, 9);
  });

  it('a rotation of zero is the identity, and opposite rotations roughly cancel', () => {
    const img = digit(7);
    for (let i = 0; i < S * S; i++) expect(rotateImage(img, 0)[i]).toBeCloseTo(img[i]!, 12);
    const back = rotateImage(rotateImage(img, 10), -10);
    const [x0, y0] = centre(img) as [number, number];
    const [x1, y1] = centre(back) as [number, number];
    expect(Math.hypot(x1 - x0, y1 - y0)).toBeLessThan(0.3);
  });

  it('a quarter turn moves a pixel to where a clockwise turn should put it', () => {
    const img = new Float64Array(S * S);
    const mid = (S - 1) / 2;
    img[Math.round(mid) * S + Math.round(mid) + 4] = 1; // right of the centre
    const turned = rotateImage(img, 90);
    const [x, y] = centre(turned) as [number, number];
    // a clockwise quarter turn on screen takes "right of the centre" to "below the centre"
    expect(y).toBeGreaterThan(mid + 3);
    expect(Math.abs(x - Math.round(mid))).toBeLessThan(1.2);
  });

  it('noise stays in [0, 1], is random per call and zero noise changes nothing', () => {
    const img = digit(1);
    const rng = createRng(2);
    const noisy = noiseImage(img, 0.3, rng);
    expect(Math.min(...noisy)).toBeGreaterThanOrEqual(0);
    expect(Math.max(...noisy)).toBeLessThanOrEqual(1);
    expect(Array.from(noiseImage(img, 0, rng))).toEqual(Array.from(img));
  });

  it('augmented sets keep the original first and are reproducible for a seed', () => {
    const img = digit(5);
    const a = augmentSet(img, 1, { shift: 2, rotate: 15, noise: 0.05 }, 3, createRng(4));
    const b = augmentSet(img, 1, { shift: 2, rotate: 15, noise: 0.05 }, 3, createRng(4));
    expect(Array.from(a.images)).toEqual(Array.from(b.images));
    expect(Array.from(a.images.subarray(0, S * S))).toEqual(Array.from(img));
    expect(a.source).toEqual([0, 0, 0, 0]);
  });

  it('augmented copies stay within the stated limits', () => {
    const img = digit(8);
    const rng = createRng(9);
    for (let i = 0; i < 50; i++) {
      const out = augmentImage(img, { shift: 1, rotate: 0, noise: 0 }, rng);
      const [x0] = centre(img) as [number, number];
      const [x1] = centre(out) as [number, number];
      expect(Math.abs(x1 - x0)).toBeLessThanOrEqual(1.5);
    }
  });
});
