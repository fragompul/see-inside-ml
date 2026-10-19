import {
  activate,
  derivative,
  usesHeInit,
  type ActivationName,
  Matrix,
  addRowVector,
  colSums,
  matmul,
  matmulAT,
  matmulBT,
  mul,
  Parameter,
  type Rng,
} from '@sim/core';

/** Fully-connected layer: a = f(x·W + b), with manual backprop. */
export class DenseLayer {
  readonly weights: Matrix;
  readonly biases: Matrix;
  readonly weightParam: Parameter;
  readonly biasParam: Parameter;

  /** Cached by the last cached forward pass; used by backprop and by visualisations. */
  inputs: Matrix | null = null;
  z: Matrix | null = null;
  a: Matrix | null = null;
  /** dL/dZ from the last backward pass (kept for step-by-step inspection). */
  delta: Matrix | null = null;

  constructor(
    readonly inputDim: number,
    readonly outputDim: number,
    readonly activation: ActivationName,
    rng: Rng,
  ) {
    this.weights = new Matrix(inputDim, outputDim);
    this.biases = new Matrix(1, outputDim);
    if (usesHeInit(activation)) {
      const std = Math.sqrt(2 / inputDim);
      for (let i = 0; i < this.weights.data.length; i++) this.weights.data[i] = rng.normal() * std;
    } else {
      const limit = Math.sqrt(6 / (inputDim + outputDim));
      for (let i = 0; i < this.weights.data.length; i++) {
        this.weights.data[i] = rng.uniform(-limit, limit);
      }
    }
    this.weightParam = new Parameter(this.weights.data, true);
    this.biasParam = new Parameter(this.biases.data, false);
  }

  get parameters(): Parameter[] {
    return [this.weightParam, this.biasParam];
  }

  /** `cache: false` leaves the stored activations untouched (used for grid predictions). */
  forward(inputs: Matrix, cache = true): Matrix {
    const z = matmul(inputs, this.weights);
    addRowVector(z, this.biases.data);
    const a = activate(this.activation, z);
    if (cache) {
      this.inputs = inputs;
      this.z = z;
      this.a = a;
    }
    return a;
  }

  /** Backprop from dL/dA. Returns dL/dInputs. */
  backward(dA: Matrix): Matrix {
    if (!this.z || !this.a) throw new Error('backward() called before forward()');
    return this.backwardFromPreActivation(mul(dA, derivative(this.activation, this.z, this.a)));
  }

  /** Backprop from dL/dZ (used at the output layer, where loss and activation are fused). */
  backwardFromPreActivation(dZ: Matrix): Matrix {
    if (!this.inputs) throw new Error('backward() called before forward()');
    this.delta = dZ;
    const m = this.inputs.rows;
    const dW = matmulAT(this.inputs, dZ);
    const db = colSums(dZ);
    for (let i = 0; i < dW.data.length; i++) this.weightParam.grad[i] = (dW.data[i] as number) / m;
    for (let j = 0; j < db.length; j++) this.biasParam.grad[j] = (db[j] as number) / m;
    return matmulBT(dZ, this.weights);
  }
}
