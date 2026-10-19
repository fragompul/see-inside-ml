import { describe, expect, it } from 'vitest';
import {
  CLASSIFICATION_DATASETS,
  REGRESSION_DATASETS,
  generateClassification,
  generateDataset,
  datasetInfo,
  generateRegression,
  splitIndices,
  toTargets,
} from './datasets';
import { FEATURES, extractFeatures } from './features';
import { Matrix, createRng } from '@sim/core';

describe('classification datasets', () => {
  it.each(CLASSIFICATION_DATASETS.map((d) => [d] as const))('%s is well formed', (name) => {
    const [lo, hi] = datasetInfo[name].classes;
    const classes = Math.min(hi, Math.max(lo, 3));
    const d = generateClassification(name, { samples: 150, noise: 0.2, classes }, createRng(1));
    expect(d.X.cols).toBe(2);
    expect(d.labels.length).toBe(d.X.rows);
    expect(d.X.rows).toBeGreaterThan(100);
    expect(d.numClasses).toBe(classes);
    for (const label of d.labels) {
      expect(label).toBeGreaterThanOrEqual(0);
      expect(label).toBeLessThan(classes);
    }
    expect(d.X.data.every(Number.isFinite)).toBe(true);
    // every class is represented
    expect(new Set(d.labels).size).toBe(classes);
  });

  it('is deterministic per seed and differs across seeds', () => {
    const make = (seed: number) =>
      generateClassification('spiral', { samples: 90, noise: 0.3, classes: 3 }, createRng(seed));
    expect(make(5).X.data).toEqual(make(5).X.data);
    expect(make(5).X.data).not.toEqual(make(6).X.data);
  });

  it('xor is the classic quadrant problem: opposite corners share a class', () => {
    const d = generateClassification('xor', { samples: 200, noise: 0, classes: 2 }, createRng(1));
    d.labels.forEach((label, i) => {
      expect(label).toBe(d.X.get(i, 0) > 0 !== d.X.get(i, 1) > 0 ? 1 : 0);
    });
  });

  it('clamps unsupported class counts', () => {
    const moons = generateClassification(
      'moons',
      { samples: 60, noise: 0, classes: 5 },
      createRng(1),
    );
    expect(moons.numClasses).toBe(2);
    const circle = generateClassification(
      'circle',
      { samples: 60, noise: 0, classes: 99 },
      createRng(1),
    );
    expect(circle.numClasses).toBe(5);
  });

  it('circles are concentric: radius grows with the class index (noise-free)', () => {
    const d = generateClassification('circle', { samples: 90, noise: 0, classes: 3 }, createRng(1));
    const meanRadius = [0, 0, 0];
    const count = [0, 0, 0];
    d.labels.forEach((label, i) => {
      meanRadius[label] = (meanRadius[label] as number) + Math.hypot(d.X.get(i, 0), d.X.get(i, 1));
      count[label] = (count[label] as number) + 1;
    });
    const radii = meanRadius.map((r, i) => r / (count[i] as number));
    expect(radii[0]).toBeCloseTo(5 / 3, 5);
    expect(radii[1]).toBeCloseTo(10 / 3, 5);
    expect(radii[2]).toBeCloseTo(5, 5);
  });
});

describe('regression datasets', () => {
  it.each(REGRESSION_DATASETS.map((d) => [d] as const))('%s is well formed', (name) => {
    const d = generateRegression(name, { samples: 120, noise: 0 }, createRng(2));
    expect(d.problem).toBe('regression');
    expect(d.values.length).toBe(120);
    expect(d.values.every(Number.isFinite)).toBe(true);
  });

  it('plane follows 0.5·x₁ + 0.5·x₂ exactly without noise', () => {
    const d = generateRegression('plane', { samples: 50, noise: 0 }, createRng(3));
    for (let i = 0; i < 50; i++) {
      expect(d.values[i]).toBeCloseTo(0.5 * d.X.get(i, 0) + 0.5 * d.X.get(i, 1), 10);
    }
  });
});

describe('targets and splits', () => {
  it('one-hot encodes classification labels', () => {
    const d = generateDataset('xor', { samples: 40, noise: 0, classes: 2 }, createRng(1));
    const y = toTargets(d);
    expect(y.cols).toBe(2);
    for (let i = 0; i < y.rows; i++) {
      expect(y.get(i, 0) + y.get(i, 1)).toBe(1);
      expect(y.get(i, d.labels[i] as number)).toBe(1);
    }
  });

  it('splits into disjoint sets covering every sample, keeping both sides non-empty', () => {
    const { trainIndices, testIndices } = splitIndices(100, 75, createRng(4));
    expect(trainIndices.length).toBe(75);
    expect(testIndices.length).toBe(25);
    expect(new Set([...trainIndices, ...testIndices]).size).toBe(100);

    const tiny = splitIndices(2, 99, createRng(4));
    expect(tiny.trainIndices.length).toBe(1);
    expect(tiny.testIndices.length).toBe(1);
  });
});

describe('features', () => {
  it('computes the documented transforms', () => {
    const X = Matrix.from([[2, -3]]);
    const ids = FEATURES.map((f) => f.id);
    const out = extractFeatures(X, ids).toArray()[0] as number[];
    const at = (id: string) => out[ids.indexOf(id)] as number;
    expect(at('X1')).toBe(2);
    expect(at('X2')).toBe(-3);
    expect(at('X1^2')).toBe(4);
    expect(at('X2^2')).toBe(9);
    expect(at('X1*X2')).toBe(-6);
    expect(at('sgn(X2)')).toBe(-1);
    expect(at('sin(X1)')).toBeCloseTo(Math.sin(2), 12);
    expect(at('cos(X2)')).toBeCloseTo(Math.cos(-3), 12);
    expect(at('RBF_Gaussian')).toBeCloseTo(Math.exp(-13 / 4), 12);
  });

  it('rejects unknown features', () => {
    expect(() => extractFeatures(Matrix.from([[1, 2]]), ['nope'])).toThrow(/Unknown feature/);
  });
});
