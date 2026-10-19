import { describe, expect, it } from 'vitest';
import { Matrix, createRng } from '@sim/core';
import { DIGIT_CLASSES, IMAGE_SIZE, generateDigits } from './digits';
import { CnnNetwork, argmax, occlusionMap, validateCnn } from './network';
import { CNN_LAB_LIMITS, CnnSession, DEFAULT_CNN_LAB, type CnnLabConfig } from './session';

const pixels = IMAGE_SIZE * IMAGE_SIZE;
const lab = (patch: Partial<CnnLabConfig> = {}): CnnLabConfig => ({ ...DEFAULT_CNN_LAB, ...patch });

const oneHot = (labels: ArrayLike<number>) => {
  const m = new Matrix(labels.length, DIGIT_CLASSES);
  for (let i = 0; i < labels.length; i++) m.set(i, labels[i] as number, 1);
  return m;
};

describe('CnnNetwork', () => {
  const set = generateDigits(2, createRng(1));
  const batch = 4;
  const images = set.images.slice(0, batch * pixels);
  const targets = oneHot(set.labels.slice(0, batch));

  it('outputs a probability distribution per image and counts its parameters', () => {
    const net = new CnnNetwork({ filters1: 6, filters2: 12, kernel: 3 }, createRng(2));
    const probs = net.forward(images, batch);
    expect(probs.rows).toBe(batch);
    for (let i = 0; i < batch; i++) {
      let sum = 0;
      for (let j = 0; j < DIGIT_CLASSES; j++) sum += probs.get(i, j);
      expect(sum).toBeCloseTo(1, 12);
    }
    // conv1 6·9+6, conv2 12·6·9+12, dense (12·5·5)·10+10
    expect(net.parameterCount).toBe(60 + 660 + 3010);
  });

  it.each([
    [{ filters1: 2, filters2: 2, kernel: 3 as const }],
    [{ filters1: 3, filters2: 2, kernel: 5 as const }],
  ])('backward matches finite differences for every parameter (%j)', (config) => {
    const net = new CnnNetwork(config, createRng(3));
    // non-zero biases keep ReLU pre-activations off the kink
    const rng = createRng(4);
    for (const p of net.parameters)
      if (!p.decay) for (let i = 0; i < p.data.length; i++) p.data[i] = rng.normal() * 0.1;

    const probs = net.forward(images, batch);
    net.backward(targets, probs);
    const analytic = net.parameters.map((p) => p.grad.slice());
    const loss = () => net.loss(net.forward(images, batch), targets);
    const eps = 1e-6;
    let worst = 0;
    net.parameters.forEach((p, pi) => {
      const picks = new Set([0, 1, Math.floor(p.data.length / 2), p.data.length - 1]);
      for (const i of picks) {
        const original = p.data[i] as number;
        p.data[i] = original + eps;
        const plus = loss();
        p.data[i] = original - eps;
        const minus = loss();
        p.data[i] = original;
        const numeric = (plus - minus) / (2 * eps);
        const a = (analytic[pi] as Float64Array)[i] as number;
        worst = Math.max(
          worst,
          Math.abs(a - numeric) / Math.max(1e-4, Math.abs(a) + Math.abs(numeric)),
        );
      }
    });
    expect(worst).toBeLessThan(1e-4);
  });

  it('trace agrees with forward and does not disturb a pending backward', () => {
    const net = new CnnNetwork({ filters1: 4, filters2: 4, kernel: 3 }, createRng(5));
    const probs = net.forward(images, batch);
    net.backward(targets, probs);
    const grads = net.parameters.map((p) => p.grad.slice());

    const probe = set.images.slice(5 * pixels, 6 * pixels);
    const trace = net.trace(probe);
    const alone = new CnnNetwork({ filters1: 4, filters2: 4, kernel: 3 }, createRng(5));
    alone.parameters.forEach((p, i) =>
      p.data.set((net.parameters[i] as { data: Float64Array }).data),
    );
    expect(trace.probs.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 12);
    alone.forward(probe, 1).data.forEach((v, i) => expect(trace.probs[i]).toBeCloseTo(v, 12));

    expect(trace.shapes).toEqual({ conv1: [4, 14], pool1: [4, 7], conv2: [4, 5] });
    expect(trace.conv1).toHaveLength(4 * 14 * 14);
    expect(trace.pool1).toHaveLength(4 * 7 * 7);
    expect(trace.conv2).toHaveLength(4 * 5 * 5);
    expect(trace.conv1.every((v) => v >= 0)).toBe(true); // post-ReLU
    // the gradients computed before the trace are intact
    net.parameters.forEach((p, i) =>
      expect(Array.from(p.grad)).toEqual(Array.from(grads[i] as Float64Array)),
    );
  });

  it('validates its configuration', () => {
    expect(() => validateCnn({ filters1: 1, filters2: 4, kernel: 3 })).toThrow(/filters1/);
    expect(() => validateCnn({ filters1: 4, filters2: 40, kernel: 3 })).toThrow(/filters2/);
    expect(() => validateCnn({ filters1: 4, filters2: 4, kernel: 4 as 3 })).toThrow(/kernel/);
  });
});

describe('CnnSession', () => {
  it('learns to read digits: well above chance within a few epochs', () => {
    const s = new CnnSession(lab());
    const first = s.step(1)[0];
    const history = s.step(9);
    const last = history[history.length - 1];
    expect(first?.testAccuracy).toBeLessThan(0.9);
    expect(last?.trainLoss).toBeLessThan((first?.trainLoss as number) * 0.5);
    expect(last?.testAccuracy).toBeGreaterThan(0.9);
    expect(s.epoch).toBe(10);
  }, 60_000);

  it('is reproducible for a seed and reset replays the same run', () => {
    const a = new CnnSession(lab({ perClass: 20 }));
    const b = new CnnSession(lab({ perClass: 20 }));
    expect(a.step(2)).toEqual(b.step(2));
    const before = a.step(1);
    a.reset();
    a.step(2);
    expect(a.step(1)).toEqual(before);
  });

  it('applies learning settings live and rebuilds on data or architecture changes', () => {
    const s = new CnnSession(lab({ perClass: 20 }));
    s.step(2);
    expect(s.update({ lr: 0.02, optimizer: 'sgd', batchSize: 32 })).toBe('live');
    expect(s.epoch).toBe(2);
    expect(s.update({ filters1: 4 })).toBe('rebuilt');
    expect(s.epoch).toBe(0);
    expect(s.network.parameterCount).toBeLessThan(3730);
    const before = s.sample(0, 'train').image.slice();
    expect(s.update({ noise: 0.2 })).toBe('rebuilt');
    expect(s.sample(0, 'train').image).not.toEqual(before);
  });

  it('builds a confusion matrix whose rows sum to the test images per digit', () => {
    const s = new CnnSession(lab({ perClass: 20 }));
    s.step(3);
    const m = s.confusion();
    expect(m).toHaveLength(10);
    m.forEach((row) => expect(row.reduce((a, b) => a + b, 0)).toBe(30));
  });

  it('training can advance a few batches at a time and still completes whole epochs', () => {
    const s = new CnnSession(lab({ perClass: 20, batchSize: 40 })); // 200 images → 5 batches per epoch
    expect(s.trainBatches(3)).toEqual([]);
    expect(s.epoch).toBe(0);
    const done = s.trainBatches(2);
    expect(done).toHaveLength(1);
    expect(done[0]?.epoch).toBe(1);
    expect(s.trainBatches(10)).toHaveLength(2); // two more full epochs
    expect(s.epoch).toBe(3);
    expect(s.history).toHaveLength(3);
  });

  it('batch-wise and epoch-wise training give the same result', () => {
    const a = new CnnSession(lab({ perClass: 20, batchSize: 40 }));
    const b = new CnnSession(lab({ perClass: 20, batchSize: 40 }));
    a.step(2);
    b.trainBatches(5);
    b.trainBatches(5);
    expect(b.history).toEqual(a.history);
  });

  it('exposes learned kernels, traces and samples', () => {
    const s = new CnnSession(lab({ filters1: 3, filters2: 5 }));
    const k = s.kernels();
    expect(k.conv1).toHaveLength(3);
    expect(k.conv1[0]).toHaveLength(9);
    expect(k.conv2).toHaveLength(5);
    expect(k.conv2[0]).toHaveLength(3);
    expect(k.conv2[0]?.[0]).toHaveLength(9);

    const { image, label } = s.sample(0);
    expect(image).toHaveLength(pixels);
    expect(label).toBeGreaterThanOrEqual(0);
    expect(s.trace(image).probs).toHaveLength(10);
    expect(() => s.trace(new Float64Array(5))).toThrow(/pixels/);
    expect(() => s.sample(99999)).toThrow(/range/);
  });

  it('validates configuration changes without breaking the session', () => {
    const s = new CnnSession(lab({ perClass: 20 }));
    expect(() => s.update({ lr: 0 })).toThrow(/Learning rate/);
    expect(() => s.update({ perClass: 5 })).toThrow(/perClass/);
    expect(() => s.update({ noise: 1 })).toThrow(/noise/);
    expect(() => new CnnSession(lab({ batchSize: 0 }))).toThrow(/batchSize/);
    expect(s.config.perClass).toBe(20);
    expect(s.step(1)).toHaveLength(1);
    expect(CNN_LAB_LIMITS.perClass[0]).toBeLessThan(CNN_LAB_LIMITS.perClass[1]);
  });
});

describe('occlusion sensitivity', { timeout: 60_000 }, () => {
  it('highlights the pixels the decision depends on, and nothing for a blank image', () => {
    const s = new CnnSession(lab());
    s.step(8);
    const { image, label } = s.sample(0);
    const guess = argmax(s.trace(image).probs);
    expect(guess).toBe(label); // a trained net reads this one correctly
    const map = occlusionMap(s.network, image, guess, 3);
    expect(map).toHaveLength(pixels);
    // covering the ink hurts far more than covering empty background
    let onInk = 0;
    let onInkN = 0;
    let off = 0;
    let offN = 0;
    image.forEach((v, i) => {
      if (v > 0.5) {
        onInk += map[i] as number;
        onInkN += 1;
      } else if (v < 0.02) {
        off += map[i] as number;
        offN += 1;
      }
    });
    expect(onInk / onInkN).toBeGreaterThan(off / offN);

    const blank = occlusionMap(s.network, new Float64Array(pixels), 3, 3);
    expect(Math.max(...blank.map(Math.abs))).toBeLessThan(0.5);
  });
});
