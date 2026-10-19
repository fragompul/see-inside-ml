import { createRng } from '@sim/core';
import { describe, expect, it } from 'vitest';
import { slicedWasserstein } from '../gan/metrics';
import {
  BLOBS,
  DEFAULT_FIELD_TRAINING,
  FieldNet,
  integrateField,
  nearestBlob,
  sampleBlobs,
  sampleFlowMatching,
  sampleGuided,
  trainFlowMatching,
  trainGuidedDiffusion,
} from './variants';

describe('field network', () => {
  it('backpropagation matches finite differences', () => {
    const net = new FieldNet(3, 6, createRng(2));
    const x = [0.3, -0.5, 1.1, 0.2];
    const t = [0.2, 0.7];
    const labels = [1, -1];
    const target = [0.4, -0.2, 0.9, 0.1];
    net.gradients(x, t, 1, labels, target);
    const h = 1e-6;
    const loss = () => {
      const out = net.run(x, t, 1, labels, false);
      return out.data.reduce((a, v, i) => a + (v - target[i]!) ** 2, 0) / out.data.length;
    };
    for (const p of net.parameters) {
      const grad = Float64Array.from(p.grad);
      for (const i of [0, Math.floor(p.data.length / 2), p.data.length - 1]) {
        const saved = p.data[i]!;
        p.data[i] = saved + h;
        const up = loss();
        p.data[i] = saved - h;
        const down = loss();
        p.data[i] = saved;
        expect(grad[i]).toBeCloseTo((up - down) / (2 * h), 5);
      }
    }
  });
});

describe('flow matching', () => {
  it('integrating the exact velocity field of a single data point lands exactly on it', () => {
    const target = [1.5, -0.7];
    const field = (x: Float64Array, t: number) =>
      Float64Array.from(x, (v, i) => (target[i % 2]! - v) / (1 - t));
    const out = integrateField(field, 20, 3, 8);
    for (let i = 0; i < 20; i++) {
      expect(out[2 * i]).toBeCloseTo(1.5, 10);
      expect(out[2 * i + 1]).toBeCloseTo(-0.7, 10);
    }
  });

  it('a trained field moves noise towards the data', () => {
    const data = sampleBlobs(1000, createRng(1)).points;
    const model = trainFlowMatching(data, { ...DEFAULT_FIELD_TRAINING, steps: 400 });
    const rng = createRng(9);
    const noise = Float64Array.from({ length: 800 }, () => rng.normal());
    const real = sampleBlobs(400, createRng(2)).points;
    const generated = sampleFlowMatching(model, 400, 5, 10);
    const before = slicedWasserstein(noise, real);
    expect(slicedWasserstein(generated, real)).toBeLessThan(before);
  });
});

describe('guided diffusion', () => {
  it('is deterministic for a seed and follows the class it is asked for better than chance', () => {
    const model = trainGuidedDiffusion({ ...DEFAULT_FIELD_TRAINING, steps: 500, dropout: 0.15 });
    const a = sampleGuided(model, 2, 0, 60, 4);
    const b = sampleGuided(model, 2, 0, 60, 4);
    expect(Array.from(a)).toEqual(Array.from(b));
    let right = 0;
    for (let i = 0; i < 60; i++) if (nearestBlob(a[2 * i]!, a[2 * i + 1]!) === 2) right++;
    expect(right / 60).toBeGreaterThan(1 / BLOBS + 0.2);
  });
});
