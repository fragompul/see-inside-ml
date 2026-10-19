import { describe, expect, it } from 'vitest';
import {
  HIDDEN_ACTIVATIONS,
  type ActivationName,
  type LossType,
  type Problem,
  Matrix,
  createRng,
} from '@sim/core';
import { NeuralNetwork } from './network';

/**
 * Gradient checking: analytic backprop gradients vs central finite differences of the loss.
 * Independent of the Python reference, so it also guards against errors we might have
 * faithfully copied from it.
 */

function setup(
  problem: Problem,
  hidden: ActivationName,
  loss: LossType | undefined,
  skip: boolean,
  seed = 11,
) {
  const rng = createRng(seed);
  const sizes = problem === 'classification' ? [3, 4, 4, 3] : [3, 4, 4, 1];
  const net = new NeuralNetwork(
    {
      layerSizes: sizes,
      activations: [hidden, hidden, problem === 'classification' ? 'softmax' : 'linear'],
      problem,
      loss,
      skipConnections: skip,
    },
    rng,
  );
  // Non-zero biases keep pre-activations off ReLU's kink (z = 0 exactly), where the derivative is undefined.
  for (const layer of net.layers) {
    for (let j = 0; j < layer.biases.data.length; j++) layer.biases.data[j] = rng.normal() * 0.1;
  }
  const m = 9;
  const X = new Matrix(m, 3);
  for (let i = 0; i < X.data.length; i++) X.data[i] = rng.normal();
  const outCols = sizes[sizes.length - 1] as number;
  const y = new Matrix(m, outCols);
  if (problem === 'classification') {
    for (let i = 0; i < m; i++) y.set(i, Math.floor(rng.next() * outCols), 1);
  } else {
    for (let i = 0; i < m; i++) y.set(i, 0, rng.normal() * 2);
  }
  return { net, X, y };
}

function maxRelativeError(net: NeuralNetwork, X: Matrix, y: Matrix): number {
  const out = net.forward(X);
  net.backward(y, out);
  const eps = 1e-6;
  let worst = 0;
  for (const param of net.parameters) {
    const analytic = param.grad.slice();
    for (let i = 0; i < param.data.length; i++) {
      const original = param.data[i] as number;
      param.data[i] = original + eps;
      const plus = net.evaluate(net.forward(X, false), y).loss;
      param.data[i] = original - eps;
      const minus = net.evaluate(net.forward(X, false), y).loss;
      param.data[i] = original;
      const numeric = (plus - minus) / (2 * eps);
      const a = analytic[i] as number;
      const denom = Math.max(1e-4, Math.abs(a) + Math.abs(numeric));
      worst = Math.max(worst, Math.abs(a - numeric) / denom);
    }
  }
  return worst;
}

describe('gradient check', () => {
  describe.each(HIDDEN_ACTIVATIONS.map((a) => [a] as const))(
    'hidden activation %s',
    (activation) => {
      it('classification (softmax + cross-entropy)', () => {
        const { net, X, y } = setup('classification', activation, undefined, false);
        expect(maxRelativeError(net, X, y)).toBeLessThan(1e-5);
      });
    },
  );

  it.each(['mse', 'huber', 'mae'] as const)('regression with %s loss', (loss) => {
    const { net, X, y } = setup('regression', 'tanh', loss, false);
    expect(maxRelativeError(net, X, y)).toBeLessThan(1e-4);
  });

  it('skip connections (classification)', () => {
    const { net, X, y } = setup('classification', 'tanh', undefined, true);
    expect(maxRelativeError(net, X, y)).toBeLessThan(1e-5);
  });

  it('skip connections (regression)', () => {
    const { net, X, y } = setup('regression', 'swish', 'mse', true);
    expect(maxRelativeError(net, X, y)).toBeLessThan(1e-5);
  });
});
