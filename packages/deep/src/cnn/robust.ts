import { Matrix, Optimizer, createRng, permutation } from '@sim/core';
import { attack, robustAccuracy, type AttackMethod } from './attacks';
import { DIGIT_CLASSES, IMAGE_SIZE, generateDigits, type DigitSet } from './digits';
import { CnnNetwork, argmax, type CnnConfig } from './network';

/**
 * Adversarial training: every training batch is made of the images as they are and of the same images after an
 * attack against the current network, so the network is taught to keep its answer inside the budget the
 * attacker has. The price is paid in accuracy on untouched images and in training time (an attack per image).
 */
const PIXELS = IMAGE_SIZE * IMAGE_SIZE;

export interface RobustTraining {
  config: CnnConfig;
  epochs: number;
  perClass: number;
  batchSize: number;
  learningRate: number;
  /** Budget of the attack used while training; 0 trains normally. */
  epsilon: number;
  /** Attack used while training. */
  method: AttackMethod;
  noise: number;
  seed: number;
}

export const DEFAULT_ROBUST: RobustTraining = {
  config: { filters1: 6, filters2: 12, kernel: 3 },
  epochs: 8,
  perClass: 40,
  batchSize: 32,
  learningRate: 0.003,
  epsilon: 0,
  method: 'fgsm',
  noise: 0.1,
  seed: 7,
};

export interface RobustRun {
  net: CnnNetwork;
  test: DigitSet;
  /** Accuracy on the untouched test set after each epoch. */
  accuracy: number[];
}

const accuracyOn = (net: CnnNetwork, set: DigitSet): number => {
  let correct = 0;
  for (let i = 0; i < set.count; i++)
    if (argmax(net.predict(set.images.slice(i * PIXELS, (i + 1) * PIXELS))) === set.labels[i])
      correct += 1;
  return correct / set.count;
};

export function trainRobust(o: RobustTraining): RobustRun {
  const style = { jitter: 0.025, tilt: 0.2, noise: o.noise };
  const train = generateDigits(o.perClass, createRng(o.seed * 10 + 1), style);
  const test = generateDigits(30, createRng(o.seed * 10 + 2), style);
  const net = new CnnNetwork(o.config, createRng(o.seed * 10 + 3));
  const optimizer = new Optimizer('adam');
  const shuffle = createRng(o.seed * 10 + 4);
  const accuracy: number[] = [];
  for (let epoch = 0; epoch < o.epochs; epoch++) {
    const order = permutation(train.count, shuffle);
    for (let start = 0; start < order.length; start += o.batchSize) {
      const idx = order.subarray(start, Math.min(order.length, start + o.batchSize));
      const rows = o.epsilon > 0 ? idx.length * 2 : idx.length;
      const images = new Float64Array(rows * PIXELS);
      const targets = new Matrix(rows, DIGIT_CLASSES);
      idx.forEach((src, n) => {
        const clean = train.images.slice(src * PIXELS, (src + 1) * PIXELS);
        const label = train.labels[src] as number;
        images.set(clean, n * PIXELS);
        targets.set(n, label, 1);
        if (o.epsilon > 0) {
          const adv = attack(net, clean, label, { method: o.method, epsilon: o.epsilon, steps: 5 });
          images.set(adv.adversarial, (idx.length + n) * PIXELS);
          targets.set(idx.length + n, label, 1);
        }
      });
      const probs = net.forward(images, rows);
      net.backward(targets, probs);
      optimizer.step(net.parameters, o.learningRate);
    }
    accuracy.push(accuracyOn(net, test));
  }
  return { net, test, accuracy };
}

/** Accuracy of a trained network on the test set under attack, for each budget. */
export function robustnessCurve(
  run: RobustRun,
  budgets: readonly number[],
  method: AttackMethod,
): number[] {
  return budgets.map((epsilon) =>
    epsilon === 0
      ? accuracyOn(run.net, run.test)
      : robustAccuracy(run.net, run.test.images, run.test.labels, { method, epsilon, steps: 10 }),
  );
}
