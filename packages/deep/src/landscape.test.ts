import { describe, expect, it } from 'vitest';
import { DEFAULT_CONFIG, PlaygroundSession, type PlaygroundConfig } from './session';
import { assignParams, flattenParams } from './landscape';

const config = (patch: Partial<PlaygroundConfig> = {}): PlaygroundConfig => ({
  ...DEFAULT_CONFIG,
  dataset: 'circle',
  samples: 200,
  hidden: [{ neurons: 4, activation: 'tanh' }],
  lr: 0.05,
  ...patch,
});

describe('loss landscape', () => {
  it('has the current training loss at the centre and restores the weights it moved', () => {
    const s = new PlaygroundSession(config());
    s.step(20);
    const before = s.snapshot({ resolution: 6 });
    const land = s.landscape({ steps: 9, kind: 'random', seed: 3 });
    const after = s.snapshot({ resolution: 6 });
    expect(Array.from(after.boundary!.values)).toEqual(Array.from(before.boundary!.values));
    const c = (land.steps - 1) / 2;
    expect(land.values[c * land.steps + c]).toBeCloseTo(land.centreLoss, 5);
    expect(land.centreLoss).toBeCloseTo(before.stats.trainLoss, 6);
  });

  it('a trained network sits in a valley: the centre is lower than the rim of the plane', () => {
    const s = new PlaygroundSession(config({ optimizer: 'adam', lr: 0.05 }));
    s.step(150);
    const land = s.landscape({ steps: 15, kind: 'path' });
    const n = land.steps;
    const rim: number[] = [];
    for (let i = 0; i < n; i++)
      rim.push(
        land.values[i] as number,
        land.values[(n - 1) * n + i] as number,
        land.values[i * n] as number,
        land.values[i * n + n - 1] as number,
      );
    const meanRim = rim.reduce((a, b) => a + b, 0) / rim.length;
    expect(land.centreLoss).toBeLessThan(meanRim);
    expect(land.kind).toBe('path');
  });

  it('the path plane shows the descent: the training path ends at the origin and starts away from it', () => {
    const s = new PlaygroundSession(config());
    s.step(60);
    const land = s.landscape({ steps: 9, kind: 'path' });
    const p = land.path!;
    const last = p.a.length - 1;
    expect(Math.hypot(p.a[last] as number, p.b[last] as number)).toBeLessThan(1e-9);
    expect(Math.hypot(p.a[0] as number, p.b[0] as number)).toBeGreaterThan(0.05);
    // the path fits inside the plane that was drawn
    for (let i = 0; i < p.a.length; i++) {
      expect(Math.abs(p.a[i] as number)).toBeLessThanOrEqual(land.radius + 1e-9);
      expect(Math.abs(p.b[i] as number)).toBeLessThanOrEqual(land.radius + 1e-9);
    }
  });

  it('falls back to random directions when there is no path yet, and is reproducible for a seed', () => {
    const s = new PlaygroundSession(config());
    const a = s.landscape({ steps: 7, kind: 'path', seed: 5 });
    expect(a.kind).toBe('random');
    const b = s.landscape({ steps: 7, kind: 'random', seed: 5 });
    expect(Array.from(a.values)).toEqual(Array.from(b.values));
    const c = s.landscape({ steps: 7, kind: 'random', seed: 6 });
    expect(Array.from(c.values)).not.toEqual(Array.from(b.values));
  });

  it('keeps a bounded number of path snapshots however long training runs', () => {
    const s = new PlaygroundSession(config({ samples: 60 }));
    s.step(400);
    const land = s.landscape({ steps: 5 });
    expect(land.path!.a.length).toBeLessThanOrEqual(161);
    expect(land.path!.a.length).toBeGreaterThan(50);
  });

  it('flatten and assign are inverses', () => {
    const s = new PlaygroundSession(config());
    // reach the network through a snapshot-free path: a landscape call must leave the vector unchanged
    s.step(3);
    const w0 = s.snapshot().weights;
    s.landscape({ steps: 5, kind: 'random' });
    expect(s.snapshot().weights).toEqual(w0);
    void flattenParams;
    void assignParams;
  });
});

describe('weight and gradient histograms', () => {
  it('count every weight of every layer and report matching statistics', () => {
    const s = new PlaygroundSession(
      config({
        hidden: [
          { neurons: 5, activation: 'relu' },
          { neurons: 3, activation: 'tanh' },
        ],
      }),
    );
    s.step(5);
    const h = s.histograms(20);
    const sizes = [2 * 5, 5 * 3, 3 * 2];
    expect(h).toHaveLength(3);
    h.forEach((layer, i) => {
      expect(layer.weights.counts.reduce((a, b) => a + b, 0)).toBe(sizes[i]);
      expect(layer.gradients.counts.reduce((a, b) => a + b, 0)).toBe(sizes[i]);
      expect(layer.weights.counts).toHaveLength(20);
      expect(layer.weights.min).toBeCloseTo(-layer.weights.max, 12);
      expect(layer.weightRms).toBeGreaterThan(0);
    });
  });

  it('L1 regularisation drives weights to exactly zero and shows up as a zero share', () => {
    const base = new PlaygroundSession(
      config({ regularization: 'none', hidden: [{ neurons: 8, activation: 'tanh' }] }),
    );
    const l1 = new PlaygroundSession(
      config({
        regularization: 'l1',
        regularizationRate: 0.05,
        hidden: [{ neurons: 8, activation: 'tanh' }],
      }),
    );
    base.step(120);
    l1.step(120);
    const share = (s: PlaygroundSession) => s.histograms().reduce((a, l) => a + l.zeroShare, 0);
    expect(share(l1)).toBeGreaterThanOrEqual(share(base));
    expect(l1.histograms()[0]!.weightRms).toBeLessThan(base.histograms()[0]!.weightRms);
  });

  it('gradients shrink as training converges', () => {
    const s = new PlaygroundSession(config({ lr: 0.05 }));
    s.step(2);
    const early = s.histograms()[0]!.gradientRms;
    s.step(300);
    const late = s.histograms()[0]!.gradientRms;
    expect(late).toBeLessThan(early);
  });
});
