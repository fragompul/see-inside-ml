import { beforeAll, describe, expect, it } from 'vitest';
import { createRng } from '@sim/core';
import { attack, evidenceMap, gradCam, robustAccuracy, upsample } from './attacks';
import { IMAGE_SIZE, generateDigits } from './digits';
import { CnnNetwork, argmax } from './network';
import { CnnSession, DEFAULT_CNN_LAB } from './session';

const px = IMAGE_SIZE * IMAGE_SIZE;

describe('input gradient', () => {
  const net = new CnnNetwork({ filters1: 3, filters2: 3, kernel: 3 }, createRng(3));
  const rng = createRng(4);
  for (const p of net.parameters)
    if (!p.decay) for (let i = 0; i < p.data.length; i++) p.data[i] = rng.normal() * 0.1;
  const set = generateDigits(2, createRng(1));
  const image = set.images.slice(0, px);

  it('matches finite differences on the pixels', () => {
    const { grad, loss } = net.inputGradient(image, 4);
    expect(loss).toBeGreaterThan(0);
    const f = (x: Float64Array) => net.inputGradient(x, 4).loss;
    let worst = 0;
    for (const i of [0, 37, 100, 129, 200, 255]) {
      const h = 1e-6;
      const up = image.slice();
      up[i] = (up[i] as number) + h;
      const down = image.slice();
      down[i] = (down[i] as number) - h;
      const numeric = (f(up) - f(down)) / (2 * h);
      const a = grad[i] as number;
      worst = Math.max(
        worst,
        Math.abs(a - numeric) / Math.max(1e-4, Math.abs(a) + Math.abs(numeric)),
      );
    }
    expect(worst).toBeLessThan(1e-4);
  });

  it('leaves the parameter gradients and caches exactly as they were', () => {
    const tiny = new CnnNetwork({ filters1: 2, filters2: 2, kernel: 3 }, createRng(9));
    const probs = tiny.forward(set.images.slice(0, 2 * px), 2);
    const targets = (() => {
      const m = probs.constructor as unknown as new (r: number, c: number) => typeof probs;
      const t = new m(2, 10);
      t.set(0, 3, 1);
      t.set(1, 5, 1);
      return t;
    })();
    tiny.backward(targets, probs);
    const grads = tiny.parameters.map((p) => p.grad.slice());
    tiny.inputGradient(image, 7);
    tiny.parameters.forEach((p, i) =>
      expect(Array.from(p.grad)).toEqual(Array.from(grads[i] as Float64Array)),
    );
  });
});

describe('attacks and Grad-CAM on a trained network', () => {
  let net: CnnNetwork;
  let images: Float64Array;
  let labels: ArrayLike<number>;

  beforeAll(() => {
    const s = new CnnSession({ ...DEFAULT_CNN_LAB, perClass: 40 });
    s.step(8);
    net = s.network;
    const test = generateDigits(8, createRng(77));
    images = test.images;
    labels = test.labels;
  }, 120_000);

  const sample = (i: number) => images.slice(i * px, (i + 1) * px);
  const clean = () => robustAccuracy(net, images, labels, { method: 'fgsm', epsilon: 0 });

  it('a zero budget changes nothing; every attack stays inside the ε-ball and the pixel range', () => {
    const none = attack(net, sample(0), labels[0] as number, {
      method: 'pgd',
      epsilon: 0,
      steps: 5,
    });
    expect(Array.from(none.adversarial)).toEqual(Array.from(sample(0)));
    for (const method of ['fgsm', 'pgd'] as const) {
      const r = attack(net, sample(3), labels[3] as number, { method, epsilon: 0.15, steps: 8 });
      for (let i = 0; i < px; i++) {
        expect(Math.abs(r.perturbation[i] as number)).toBeLessThanOrEqual(0.15 + 1e-12);
        expect(r.adversarial[i]).toBeGreaterThanOrEqual(0);
        expect(r.adversarial[i]).toBeLessThanOrEqual(1);
      }
    }
  });

  it('the trained network reads clean digits well', () => {
    expect(clean()).toBeGreaterThan(0.9);
  });

  it('larger budgets break more predictions, and PGD is at least as strong as FGSM', () => {
    const acc = (method: 'fgsm' | 'pgd', epsilon: number) =>
      robustAccuracy(net, images, labels, { method, epsilon, steps: 10 });
    const f = [0.05, 0.15, 0.3].map((e) => acc('fgsm', e));
    expect(f[0]).toBeLessThanOrEqual(clean());
    expect(f[1]).toBeLessThan(f[0] as number);
    expect(f[2]).toBeLessThan(clean() - 0.2);
    expect(acc('pgd', 0.15)).toBeLessThanOrEqual((f[1] as number) + 0.02);
  });

  it('a small FGSM step raises the loss of the true class for most images', () => {
    let up = 0;
    const n = 40;
    for (let i = 0; i < n; i++) {
      const before = net.inputGradient(sample(i), labels[i] as number).loss;
      const r = attack(net, sample(i), labels[i] as number, { method: 'fgsm', epsilon: 0.05 });
      if (net.inputGradient(r.adversarial, labels[i] as number).loss > before) up++;
    }
    expect(up / n).toBeGreaterThan(0.9);
  });

  it('a targeted PGD attack can make the network say the digit we pick', () => {
    let hits = 0;
    let tried = 0;
    for (let i = 0; i < 30; i++) {
      const target = ((labels[i] as number) + 3) % 10;
      const r = attack(net, sample(i), labels[i] as number, {
        method: 'pgd',
        epsilon: 0.35,
        steps: 25,
        targetClass: target,
      });
      tried++;
      if (r.fooled) hits++;
    }
    expect(hits / tried).toBeGreaterThan(0.5);
  });

  it('Grad-CAM is a non-negative map scaled to [0, 1] (or empty), enlarged to the image size', () => {
    for (let i = 0; i < 20; i++) {
      const cam = gradCam(net, sample(i));
      expect(cam.map).toHaveLength(px);
      expect(cam.coarse).toHaveLength(25);
      expect(Math.min(...cam.coarse)).toBeGreaterThanOrEqual(0);
      expect(Math.max(...cam.map)).toBeLessThanOrEqual(1 + 1e-12);
      expect(cam.target).toBe(argmax(net.predict(sample(i))));
      // averaging position-specific weights can cancel the evidence completely: then the map is empty
      const top = Math.max(...cam.coarse);
      expect(top === 0 || Math.abs(top - 1) < 1e-12).toBe(true);
    }
  });

  it('the evidence map adds up to the class score exactly', () => {
    for (let i = 0; i < 10; i++) {
      const img = sample(i);
      const ev = evidenceMap(net, img);
      const logit = net.trace(img).logits[ev.target] as number;
      const bias = net.dense.biases.get(0, ev.target);
      expect(ev.score + bias).toBeCloseTo(logit, 9);
    }
  });

  it('hiding the pixels the evidence map highlights hurts the class more than hiding the ones it ignores', () => {
    let topDrop = 0;
    let lowDrop = 0;
    let used = 0;
    for (let i = 0; i < 40; i++) {
      const img = sample(i);
      const cls = argmax(net.predict(img));
      const ev = evidenceMap(net, img, cls);
      const ink = Array.from({ length: px }, (_, k) => k).filter((k) => (img[k] as number) > 0.3);
      if (ink.length < 20) continue;
      const order = [...ink].sort((a, b) => (ev.map[b] as number) - (ev.map[a] as number));
      const k = Math.floor(ink.length / 3);
      const hide = (idx: number[]) => {
        const masked = img.slice();
        for (const j of idx) masked[j] = 0;
        return (net.predict(img)[cls] as number) - (net.predict(masked)[cls] as number);
      };
      topDrop += hide(order.slice(0, k));
      lowDrop += hide(order.slice(-k));
      used++;
    }
    expect(used).toBeGreaterThan(20);
    expect(topDrop / used).toBeGreaterThan(lowDrop / used);
  });
});

describe('upsample', () => {
  it('keeps a constant map constant and stays within the range of the input', () => {
    const flat = new Float64Array(25).fill(0.4);
    for (const v of upsample(flat, 5, 16)) expect(v).toBeCloseTo(0.4, 12);
    const ramp = Float64Array.from({ length: 25 }, (_, i) => i / 24);
    const big = upsample(ramp, 5, 16);
    expect(Math.min(...big)).toBeGreaterThanOrEqual(0);
    expect(Math.max(...big)).toBeLessThanOrEqual(1);
  });
});
