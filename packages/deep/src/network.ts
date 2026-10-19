import {
  isActivationName,
  type ActivationName,
  defaultLoss,
  isValidLoss,
  lossValue,
  outputGradient,
  type LossType,
  type Problem,
  type Matrix,
  hconcat,
  sliceCols,
  NO_REGULARIZATION,
  type Parameter,
  type Regularization,
  type Rng,
} from '@sim/core';
import { DenseLayer } from './dense';

export interface NetworkConfig {
  /** [inputs, hidden…, outputs]. */
  layerSizes: number[];
  /** One activation per weight layer (so `layerSizes.length - 1` entries). */
  activations: ActivationName[];
  problem: Problem;
  /** Defaults to cross-entropy for classification and MSE for regression. */
  loss?: LossType;
  /** Feed the raw inputs to every hidden layer too (ResNet-flavoured). */
  skipConnections?: boolean;
}

export interface LayerWeights {
  W: number[][];
  b: number[];
}

export interface Evaluation {
  /** Loss including the regularisation penalty. */
  loss: number;
  /** Accuracy for classification, MSE for regression. */
  metric: number;
}

export function validateConfig(config: NetworkConfig): void {
  const { layerSizes, activations, problem } = config;
  if (layerSizes.length < 2)
    throw new RangeError('A network needs at least input and output sizes');
  if (!layerSizes.every((n) => Number.isInteger(n) && n >= 1)) {
    throw new RangeError('Layer sizes must be positive integers');
  }
  if (activations.length !== layerSizes.length - 1) {
    throw new RangeError(
      `Expected ${layerSizes.length - 1} activations, received ${activations.length}`,
    );
  }
  activations.forEach((name, i) => {
    if (!isActivationName(name)) throw new RangeError(`Unknown activation "${name}"`);
    const isOutput = i === activations.length - 1;
    if (name === 'softmax' && !isOutput) {
      throw new RangeError('Softmax is only valid on the output layer');
    }
  });
  const output = activations[activations.length - 1];
  const outputSize = layerSizes[layerSizes.length - 1] as number;
  if (problem === 'classification') {
    if (output !== 'softmax') throw new RangeError('Classification needs a softmax output layer');
    if (outputSize < 2) throw new RangeError('Classification needs at least 2 output classes');
  } else {
    if (output !== 'linear') throw new RangeError('Regression needs a linear output layer');
    if (outputSize !== 1) throw new RangeError('Regression needs exactly 1 output');
  }
  const loss = config.loss ?? defaultLoss(problem);
  if (!isValidLoss(problem, loss)) {
    throw new RangeError(`Loss "${loss}" is not valid for ${problem}`);
  }
}

/**
 * Multilayer perceptron with manual backpropagation.
 * Mirrors the reference NumPy implementation operation-for-operation (verified by parity tests).
 */
export class NeuralNetwork {
  readonly layers: DenseLayer[] = [];
  readonly problem: Problem;
  readonly loss: LossType;
  readonly skipConnections: boolean;
  readonly inputDim: number;
  private lastInput: Matrix | null = null;

  constructor(
    readonly config: NetworkConfig,
    rng: Rng,
  ) {
    validateConfig(config);
    this.problem = config.problem;
    this.loss = config.loss ?? defaultLoss(config.problem);
    this.skipConnections = config.skipConnections ?? false;
    this.inputDim = config.layerSizes[0] as number;

    for (let i = 0; i < config.layerSizes.length - 1; i++) {
      // Skip connections widen every layer after the first by the raw input size.
      const width =
        (config.layerSizes[i] as number) + (this.skipConnections && i > 0 ? this.inputDim : 0);
      this.layers.push(
        new DenseLayer(
          width,
          config.layerSizes[i + 1] as number,
          config.activations[i] as ActivationName,
          rng,
        ),
      );
    }
  }

  get parameters(): Parameter[] {
    return this.layers.flatMap((layer) => layer.parameters);
  }

  get parameterCount(): number {
    return this.parameters.reduce((sum, p) => sum + p.data.length, 0);
  }

  /**
   * Forward pass. With `cache: false` nothing is stored, so it is safe to call on a
   * visualisation grid without disturbing the activations of the last training batch.
   */
  forward(X: Matrix, cache = true): Matrix {
    if (X.cols !== this.inputDim) {
      throw new RangeError(`Expected ${this.inputDim} input features, received ${X.cols}`);
    }
    if (cache) this.lastInput = X;
    let a = X;
    this.layers.forEach((layer, i) => {
      const input = this.skipConnections && i > 0 ? hconcat(a, X) : a;
      a = layer.forward(input, cache);
    });
    return a;
  }

  /**
   * Activations of every layer for `X`, without touching any cache. Powers per-neuron
   * visualisations (e.g. what each neuron "sees" across the whole input plane).
   */
  trace(X: Matrix): Matrix[] {
    if (X.cols !== this.inputDim) {
      throw new RangeError(`Expected ${this.inputDim} input features, received ${X.cols}`);
    }
    const activations: Matrix[] = [];
    let a = X;
    this.layers.forEach((layer, i) => {
      a = layer.forward(this.skipConnections && i > 0 ? hconcat(a, X) : a, false);
      activations.push(a);
    });
    return activations;
  }

  /** Backprop for the batch of the last cached `forward`; fills every parameter's gradient. */
  backward(y: Matrix, out: Matrix): void {
    if (!this.lastInput) throw new Error('backward() called before forward()');
    const last = this.layers[this.layers.length - 1] as DenseLayer;
    let dInputs = last.backwardFromPreActivation(outputGradient(this.loss, out, y));
    if (this.skipConnections && this.layers.length > 1) {
      dInputs = sliceCols(
        dInputs,
        0,
        (this.layers[this.layers.length - 2] as DenseLayer).outputDim,
      );
    }
    for (let i = this.layers.length - 2; i >= 0; i--) {
      dInputs = (this.layers[i] as DenseLayer).backward(dInputs);
      if (this.skipConnections && i > 0) {
        dInputs = sliceCols(dInputs, 0, (this.layers[i - 1] as DenseLayer).outputDim);
      }
    }
  }

  /** Loss (plus L1/L2 penalty on weights) and the problem's headline metric. */
  evaluate(out: Matrix, y: Matrix, reg: Regularization = NO_REGULARIZATION): Evaluation {
    let loss = lossValue(this.loss, out, y);
    for (const layer of this.layers) {
      const w = layer.weights.data;
      if (reg.l1 > 0) {
        let s = 0;
        for (let i = 0; i < w.length; i++) s += Math.abs(w[i] as number);
        loss += reg.l1 * s;
      }
      if (reg.l2 > 0) {
        let s = 0;
        for (let i = 0; i < w.length; i++) s += (w[i] as number) ** 2;
        loss += 0.5 * reg.l2 * s;
      }
    }
    return { loss, metric: this.metric(out, y) };
  }

  metric(out: Matrix, y: Matrix): number {
    if (this.problem === 'regression') {
      let s = 0;
      for (let i = 0; i < out.data.length; i++) {
        s += ((out.data[i] as number) - (y.data[i] as number)) ** 2;
      }
      return s / out.data.length;
    }
    let correct = 0;
    for (let i = 0; i < out.rows; i++) {
      if (argmaxRow(out, i) === argmaxRow(y, i)) correct += 1;
    }
    return correct / out.rows;
  }

  getWeights(): LayerWeights[] {
    return this.layers.map((l) => ({ W: l.weights.toArray(), b: Array.from(l.biases.data) }));
  }

  setWeights(weights: readonly LayerWeights[]): void {
    if (weights.length !== this.layers.length) throw new RangeError('Layer count mismatch');
    weights.forEach((w, i) => {
      const layer = this.layers[i] as DenseLayer;
      if (w.W.length !== layer.inputDim || w.W[0]?.length !== layer.outputDim) {
        throw new RangeError(`Weight shape mismatch at layer ${i}`);
      }
      if (w.b.length !== layer.outputDim) throw new RangeError(`Bias shape mismatch at layer ${i}`);
      layer.weights.data.set(w.W.flat());
      layer.biases.data.set(w.b);
    });
  }
}

export function argmaxRow(m: Matrix, row: number): number {
  let best = 0;
  let bestValue = -Infinity;
  for (let j = 0; j < m.cols; j++) {
    const v = m.get(row, j);
    if (v > bestValue) {
      bestValue = v;
      best = j;
    }
  }
  return best;
}
