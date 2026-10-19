import { describe, expect, it } from 'vitest';
import { actionTriangle, bestActions, cellAt, cellXY, normalise, valueRange } from './view';

describe('valueRange / normalise', () => {
  it('ignores skipped cells, widens a flat range and clamps', () => {
    const values = [5, -3, 100, 0, -50];
    const range = valueRange(values, (s) => s === 2 || s === 4); // skip the 100 and the -50
    expect(range).toEqual({ lo: -3, hi: 5 });
    expect(valueRange([2, 2], () => false)).toEqual({ lo: 1.5, hi: 2.5 });
    expect(valueRange([1, 2], () => true)).toEqual({ lo: 0, hi: 1 });
    expect(normalise(1, { lo: -3, hi: 5 })).toBe(0.5);
    expect(normalise(99, { lo: -3, hi: 5 })).toBe(1);
    expect(normalise(-9, { lo: -3, hi: 5 })).toBe(0);
  });
});

describe('bestActions', () => {
  it('lists the best action(s), and none before anything is learned', () => {
    const q = [0, 0, 0, 0, 1, 3, 3, 2];
    expect(bestActions(q, 0)).toEqual([]);
    expect(bestActions(q, 1)).toEqual([1, 2]);
    expect(bestActions([-5, -1, -9, -1], 0)).toEqual([1, 3]);
  });
});

describe('actionTriangle', () => {
  it('has its apex at the centre and its base on the side the action points to', () => {
    expect(actionTriangle(0, 40)).toBe('20,20 0,0 40,0'); // up
    expect(actionTriangle(1, 40)).toBe('20,20 40,0 40,40'); // right
    expect(actionTriangle(2, 40)).toBe('20,20 40,40 0,40'); // down
    expect(actionTriangle(3, 40)).toBe('20,20 0,40 0,0'); // left
    expect(actionTriangle(0, 40, 2)).toBe('20,20 2,2 38,2');
  });
});

describe('cell geometry', () => {
  it('converts between cell indices and pixels', () => {
    expect(cellXY(14, 12)).toEqual([2, 1]);
    expect(cellAt(5, 5, 12, 4, 40)).toBe(0);
    expect(cellAt(45, 5, 12, 4, 40)).toBe(1);
    expect(cellAt(5, 45, 12, 4, 40)).toBe(12);
    expect(cellAt(479, 159, 12, 4, 40)).toBe(47);
    expect(cellAt(-1, 5, 12, 4, 40)).toBeNull();
    expect(cellAt(5, 160, 12, 4, 40)).toBeNull();
    expect(cellAt(480, 5, 12, 4, 40)).toBeNull();
  });
});
