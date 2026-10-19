import { describe, expect, it } from 'vitest';
import { blurImage, totalVariation, visualise, DEFAULT_VISUALISE } from './featurevis';
import { CnnSession, DEFAULT_CNN_LAB } from './session';

const session = new CnnSession({ ...DEFAULT_CNN_LAB, perClass: 20, seed: 3 });
session.step(3);
const net = session.network;
const { image } = session.sample(7);

describe('channel gradient', () => {
  it('matches finite differences of the mean activation of a feature map', () => {
    const channel = 2;
    const { grad } = net.channelGradient(image, channel);
    const h = 1e-6;
    for (const i of [20, 100, 140, 230]) {
      const up = Float64Array.from(image);
      const down = Float64Array.from(image);
      up[i] = up[i]! + h;
      down[i] = down[i]! - h;
      const numeric =
        (net.channelGradient(up, channel).activation -
          net.channelGradient(down, channel).activation) /
        (2 * h);
      expect(grad[i]).toBeCloseTo(numeric, 5);
    }
  });

  it('leaves the network as it was', () => {
    const before = net.predict(image);
    net.channelGradient(image, 1);
    expect(net.predict(image)).toEqual(before);
  });
});

describe('feature visualisation', () => {
  it('gradient ascent raises the score of the unit and keeps pixels in range', () => {
    for (const target of [
      { kind: 'class', index: 3 },
      { kind: 'channel', index: 1 },
    ] as const) {
      const v = visualise(net, target, { ...DEFAULT_VISUALISE, steps: 40 });
      expect(v.endScore).toBeGreaterThan(v.startScore);
      expect(Math.min(...v.image)).toBeGreaterThanOrEqual(0);
      expect(Math.max(...v.image)).toBeLessThanOrEqual(1);
      expect(v.trace).toHaveLength(41);
    }
  });

  it('blurring lowers the total variation, and a flat image has none', () => {
    const noisy = Float64Array.from({ length: 256 }, (_, i) => ((i * 37) % 11) / 10);
    expect(totalVariation(blurImage(noisy, 1))).toBeLessThan(totalVariation(noisy));
    expect(totalVariation(new Float64Array(256).fill(0.3))).toBe(0);
    expect(blurImage(noisy, 0)).toBe(noisy);
  });
});
