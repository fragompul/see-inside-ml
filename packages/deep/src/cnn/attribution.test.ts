import { describe, expect, it } from 'vitest';
import {
  attribute,
  deletionCurve,
  gradientTimesInput,
  integratedGradients,
  rankPixels,
  saliency,
  scoreGap,
  smoothGrad,
} from './attribution';
import { DEFAULT_CNN_LAB, CnnSession } from './session';

const session = new CnnSession({ ...DEFAULT_CNN_LAB, perClass: 20, seed: 3 });
session.step(3);
const net = session.network;
const { image, label } = session.sample(5);

describe('logit gradient', () => {
  it('matches finite differences of the class score', () => {
    const { grad } = net.logitGradient(image, label);
    const h = 1e-6;
    for (const i of [10, 85, 130, 200]) {
      const up = Float64Array.from(image);
      const down = Float64Array.from(image);
      up[i] = up[i]! + h;
      down[i] = down[i]! - h;
      const numeric =
        (net.logitGradient(up, label).logit - net.logitGradient(down, label).logit) / (2 * h);
      expect(grad[i]).toBeCloseTo(numeric, 5);
    }
  });

  it('leaves the network exactly as it was', () => {
    const before = net.predict(image);
    net.logitGradient(image, 3);
    expect(net.predict(image)).toEqual(before);
  });
});

describe('attribution methods', () => {
  it('integrated gradients add up to the change in score, more exactly with more steps', () => {
    const gap = scoreGap(net, image, label);
    const error = (steps: number) =>
      Math.abs(integratedGradients(net, image, label, steps).reduce((a, b) => a + b, 0) - gap);
    expect(error(256)).toBeLessThan(0.02 * Math.abs(gap) + 1e-3);
    expect(error(256)).toBeLessThan(error(2) + 1e-12);
  });

  it('gradient times input gives one finite value per pixel', () => {
    const map = gradientTimesInput(net, image, label);
    expect(map).toHaveLength(256);
    expect(Number.isFinite(map.reduce((a, b) => a + b, 0))).toBe(true);
  });

  it('saliency is non-negative and SmoothGrad is reproducible', () => {
    expect(Math.min(...saliency(net, image, label))).toBeGreaterThanOrEqual(0);
    expect(Array.from(smoothGrad(net, image, label, 5, 0.1, 2))).toEqual(
      Array.from(smoothGrad(net, image, label, 5, 0.1, 2)),
    );
  });

  it('attribute defaults to the class the network predicts', () => {
    const { target } = attribute('integrated', net, image);
    const probs = net.predict(image);
    expect(probs[target]).toBe(Math.max(...probs));
  });
});

describe('deletion curve', () => {
  it('starts at the score of the image, has the requested number of points and ranks pixels by value', () => {
    const order = rankPixels(integratedGradients(net, image, label, 16));
    const curve = deletionCurve(net, image, label, order, 10);
    expect(curve.scores).toHaveLength(11);
    expect(curve.scores[0]).toBeCloseTo(net.logitGradient(image, label).logit, 10);
    expect(rankPixels([0.1, 0.9, 0.5])).toEqual([1, 2, 0]);
  });

  it('removing the most important pixels first hurts more than removing them last', () => {
    const map = integratedGradients(net, image, label, 16);
    const order = rankPixels(map);
    const first = deletionCurve(net, image, label, order, 10).area;
    const last = deletionCurve(net, image, label, [...order].reverse(), 10).area;
    expect(first).toBeLessThan(last);
  });
});
