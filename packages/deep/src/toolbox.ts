import { Matrix, createRng, type Rng } from '@sim/core';

/**
 * A small multi-layer perceptron with the training tricks that make deep networks work: weight
 * initialisation schemes, dropout, batch normalisation, weight decay, early stopping and a learning-rate
 * finder. Hand-written forward and backward passes so that each trick can be switched on and off.
 */
export type Activation = 'relu' | 'tanh';
export type InitScheme = 'zeros' | 'small' | 'xavier' | 'he' | 'large';
export type ToolOptimizer = 'sgd' | 'momentum' | 'adam';

export interface ToolboxConfig {
  /** Layer sizes including input and output, e.g. [2, 32, 32, 2]. */
  layers: number[];
  activation: Activation;
  init: InitScheme;
  /** Probability of zeroing a hidden unit during training. */
  dropout: number;
  batchNorm: boolean;
  weightDecay: number;
  seed: number;
}

export const DEFAULT_TOOLBOX: ToolboxConfig = {
  layers: [2, 32, 32, 2],
  activation: 'relu',
  init: 'he',
  dropout: 0,
  batchNorm: false,
  weightDecay: 0,
  seed: 1,
};

export interface Tensor {
  data: Float64Array;
  grad: Float64Array;
  /** Whether weight decay applies. */
  decay: boolean;
  /** Frozen tensors are never updated (the pre-trained weights under a LoRA adapter). */
  frozen?: boolean;
  m: Float64Array;
  v: Float64Array;
}

const tensor = (data: Float64Array, decay: boolean): Tensor => ({
  data,
  grad: new Float64Array(data.length),
  decay,
  m: new Float64Array(data.length),
  v: new Float64Array(data.length),
});

interface Cache {
  /** Input of each linear layer. */
  inputs: Matrix[];
  /** Pre-normalisation output of each linear layer. */
  pre: Matrix[];
  /** Normalised values and statistics of each batch-norm layer. */
  normalized: Matrix[];
  invStd: Float64Array[];
  /** Pre-activation (after batch norm) of each hidden layer. */
  act: Matrix[];
  masks: (Matrix | null)[];
  probs: Matrix;
}

export interface Adapter {
  layer: number;
  rank: number;
  fanIn: number;
  fanOut: number;
  /** The pre-trained weights, frozen. */
  base: Float64Array;
  /** Low-rank factors: the layer uses base + a (fanIn x rank) times b (rank x fanOut). */
  a: Tensor;
  b: Tensor;
}

export class ToolboxMLP {
  readonly weights: Tensor[] = [];
  readonly biases: Tensor[] = [];
  readonly gammas: Tensor[] = [];
  readonly betas: Tensor[] = [];
  readonly runMean: Float64Array[] = [];
  readonly runVar: Float64Array[] = [];
  readonly adapters: Adapter[] = [];
  private step = 0;
  private cache: Cache | null = null;

  constructor(readonly config: ToolboxConfig) {
    const rng = createRng(config.seed);
    const { layers } = config;
    for (let l = 0; l + 1 < layers.length; l++) {
      const fanIn = layers[l] as number;
      const fanOut = layers[l + 1] as number;
      const w = new Float64Array(fanIn * fanOut);
      const sd =
        config.init === 'zeros'
          ? 0
          : config.init === 'small'
            ? 0.01
            : config.init === 'large'
              ? 1.5
              : config.init === 'xavier'
                ? Math.sqrt(2 / (fanIn + fanOut))
                : Math.sqrt(2 / fanIn);
      for (let i = 0; i < w.length; i++) w[i] = rng.normal() * sd;
      this.weights.push(tensor(w, true));
      this.biases.push(tensor(new Float64Array(fanOut), false));
      if (config.batchNorm && l + 2 < layers.length) {
        this.gammas.push(tensor(new Float64Array(fanOut).fill(1), false));
        this.betas.push(tensor(new Float64Array(fanOut), false));
        this.runMean.push(new Float64Array(fanOut));
        this.runVar.push(new Float64Array(fanOut).fill(1));
      }
    }
  }

  /**
   * LoRA: freeze the network and learn, for every layer, a low-rank update A B added to its weights.
   * B starts at zero so the network initially computes exactly what it did before.
   */
  addAdapters(rank: number, seed = 1): void {
    const rng = createRng(seed);
    for (const t of [...this.weights, ...this.biases, ...this.gammas, ...this.betas])
      t.frozen = true;
    this.weights.forEach((w, layer) => {
      const fanIn = this.config.layers[layer] as number;
      const fanOut = this.config.layers[layer + 1] as number;
      const a = new Float64Array(fanIn * rank);
      for (let i = 0; i < a.length; i++) a[i] = rng.normal() * 0.1;
      this.adapters.push({
        layer,
        rank,
        fanIn,
        fanOut,
        base: Float64Array.from(w.data),
        a: tensor(a, false),
        b: tensor(new Float64Array(rank * fanOut), false),
      });
    });
  }

  /** Number of trainable parameters (frozen ones do not count). */
  get trainableCount(): number {
    return this.tensors().reduce((s, t) => s + (t.frozen ? 0 : t.data.length), 0);
  }

  private syncAdapters(): void {
    for (const ad of this.adapters) {
      const w = this.weights[ad.layer] as Tensor;
      const { fanIn, fanOut, rank } = ad;
      for (let j = 0; j < fanIn; j++)
        for (let k = 0; k < fanOut; k++) {
          let s = ad.base[j * fanOut + k] as number;
          for (let q = 0; q < rank; q++)
            s += (ad.a.data[j * rank + q] as number) * (ad.b.data[q * fanOut + k] as number);
          w.data[j * fanOut + k] = s;
        }
    }
  }

  private adapterGradients(): void {
    for (const ad of this.adapters) {
      const w = this.weights[ad.layer] as Tensor;
      const { fanIn, fanOut, rank } = ad;
      for (let j = 0; j < fanIn; j++)
        for (let q = 0; q < rank; q++) {
          let s = 0;
          for (let k = 0; k < fanOut; k++)
            s += (w.grad[j * fanOut + k] as number) * (ad.b.data[q * fanOut + k] as number);
          ad.a.grad[j * rank + q] = (ad.a.grad[j * rank + q] as number) + s;
        }
      for (let q = 0; q < rank; q++)
        for (let k = 0; k < fanOut; k++) {
          let s = 0;
          for (let j = 0; j < fanIn; j++)
            s += (ad.a.data[j * rank + q] as number) * (w.grad[j * fanOut + k] as number);
          ad.b.grad[q * fanOut + k] = (ad.b.grad[q * fanOut + k] as number) + s;
        }
    }
  }

  /** Folds the adapters into the weights and removes them: the network is the same size as before. */
  mergeAdapters(): void {
    this.syncAdapters();
    this.adapters.length = 0;
  }

  get depth(): number {
    return this.weights.length;
  }

  /** Every learnable tensor, in a fixed order. */
  tensors(): Tensor[] {
    return [
      ...this.weights,
      ...this.biases,
      ...this.gammas,
      ...this.betas,
      ...this.adapters.flatMap((a) => [a.a, a.b]),
    ];
  }

  zeroGrad(): void {
    for (const t of this.tensors()) t.grad.fill(0);
  }

  private act(x: number): number {
    return this.config.activation === 'relu' ? (x > 0 ? x : 0) : Math.tanh(x);
  }

  private actGrad(pre: number, out: number): number {
    return this.config.activation === 'relu' ? (pre > 0 ? 1 : 0) : 1 - out * out;
  }

  /** What leaves each hidden layer (rows x width) for the batch `x`, from an evaluation-mode forward pass. */
  hiddenActivations(x: Matrix): Matrix[] {
    this.forward(x, false);
    return (this.cache as Cache).inputs.slice(1);
  }

  /** Class probabilities. In training mode, dropout and batch statistics are active. */
  forward(x: Matrix, training: boolean, rng?: Rng): Matrix {
    this.syncAdapters();
    const { dropout, batchNorm } = this.config;
    const inputs: Matrix[] = [];
    const pre: Matrix[] = [];
    const normalized: Matrix[] = [];
    const invStd: Float64Array[] = [];
    const act: Matrix[] = [];
    const masks: (Matrix | null)[] = [];
    let a = x;
    const L = this.depth;
    for (let l = 0; l < L; l++) {
      inputs.push(a);
      const w = this.weights[l] as Tensor;
      const b = this.biases[l] as Tensor;
      const fanIn = a.cols;
      const fanOut = b.data.length;
      const z = new Matrix(a.rows, fanOut);
      for (let i = 0; i < a.rows; i++)
        for (let k = 0; k < fanOut; k++) {
          let s = b.data[k] as number;
          for (let j = 0; j < fanIn; j++) s += a.get(i, j) * (w.data[j * fanOut + k] as number);
          z.set(i, k, s);
        }
      pre.push(z);
      if (l === L - 1) {
        // softmax output
        const out = new Matrix(a.rows, fanOut);
        for (let i = 0; i < a.rows; i++) {
          let mx = -Infinity;
          for (let k = 0; k < fanOut; k++) mx = Math.max(mx, z.get(i, k));
          let sum = 0;
          for (let k = 0; k < fanOut; k++) {
            const e = Math.exp(z.get(i, k) - mx);
            out.set(i, k, e);
            sum += e;
          }
          for (let k = 0; k < fanOut; k++) out.set(i, k, out.get(i, k) / sum);
        }
        this.cache = { inputs, pre, normalized, invStd, act, masks, probs: out };
        return out;
      }
      let h = z;
      if (batchNorm) {
        const gamma = this.gammas[l] as Tensor;
        const beta = this.betas[l] as Tensor;
        const n = z.rows;
        const norm = new Matrix(n, fanOut);
        const inv = new Float64Array(fanOut);
        const out = new Matrix(n, fanOut);
        for (let k = 0; k < fanOut; k++) {
          let mean: number;
          let variance: number;
          if (training) {
            mean = 0;
            for (let i = 0; i < n; i++) mean += z.get(i, k) / n;
            variance = 0;
            for (let i = 0; i < n; i++) variance += (z.get(i, k) - mean) ** 2 / n;
            (this.runMean[l] as Float64Array)[k] =
              0.9 * (this.runMean[l] as Float64Array)[k]! + 0.1 * mean;
            (this.runVar[l] as Float64Array)[k] =
              0.9 * (this.runVar[l] as Float64Array)[k]! + 0.1 * variance;
          } else {
            mean = (this.runMean[l] as Float64Array)[k] as number;
            variance = (this.runVar[l] as Float64Array)[k] as number;
          }
          inv[k] = 1 / Math.sqrt(variance + 1e-5);
          for (let i = 0; i < n; i++) {
            const nz = (z.get(i, k) - mean) * (inv[k] as number);
            norm.set(i, k, nz);
            out.set(i, k, (gamma.data[k] as number) * nz + (beta.data[k] as number));
          }
        }
        normalized.push(norm);
        invStd.push(inv);
        h = out;
      }
      act.push(h);
      const out = new Matrix(h.rows, fanOut);
      for (let i = 0; i < h.rows; i++)
        for (let k = 0; k < fanOut; k++) out.set(i, k, this.act(h.get(i, k)));
      let next = out;
      if (training && dropout > 0 && rng) {
        const mask = new Matrix(h.rows, fanOut);
        const keep = 1 - dropout;
        for (let i = 0; i < h.rows; i++)
          for (let k = 0; k < fanOut; k++) mask.set(i, k, rng.next() < keep ? 1 / keep : 0);
        masks.push(mask);
        next = new Matrix(h.rows, fanOut);
        for (let i = 0; i < h.rows; i++)
          for (let k = 0; k < fanOut; k++) next.set(i, k, out.get(i, k) * mask.get(i, k));
      } else masks.push(null);
      a = next;
    }
    throw new Error('unreachable');
  }

  /** Cross-entropy loss of the last forward pass against integer labels, and the gradients of every parameter. */
  backward(labels: ArrayLike<number>, scale = 1): number {
    const c = this.cache as Cache;
    const n = c.probs.rows;
    let loss = 0;
    const dz = new Matrix(n, c.probs.cols);
    for (let i = 0; i < n; i++) {
      loss -= Math.log(Math.max(c.probs.get(i, labels[i] as number), 1e-300)) / n;
      for (let k = 0; k < c.probs.cols; k++)
        dz.set(i, k, ((c.probs.get(i, k) - (k === labels[i] ? 1 : 0)) / n) * scale);
    }
    this.propagate(dz);
    return loss;
  }

  /** Cross-entropy against a full probability distribution per row (soft targets, as in distillation). */
  backwardSoft(targets: Matrix, scale = 1): number {
    const c = this.cache as Cache;
    const n = c.probs.rows;
    let loss = 0;
    const dz = new Matrix(n, c.probs.cols);
    for (let i = 0; i < n; i++)
      for (let k = 0; k < c.probs.cols; k++) {
        loss -= (targets.get(i, k) * Math.log(Math.max(c.probs.get(i, k), 1e-300))) / n;
        dz.set(i, k, ((c.probs.get(i, k) - targets.get(i, k)) / n) * scale);
      }
    this.propagate(dz);
    return loss;
  }

  private propagate(dzIn: Matrix): void {
    const c = this.cache as Cache;
    const L = this.depth;
    const n = c.probs.rows;
    let dz = dzIn;
    for (let l = L - 1; l >= 0; l--) {
      const w = this.weights[l] as Tensor;
      const b = this.biases[l] as Tensor;
      const input = c.inputs[l] as Matrix;
      const fanIn = input.cols;
      const fanOut = b.data.length;
      for (let i = 0; i < n; i++)
        for (let k = 0; k < fanOut; k++) {
          const d = dz.get(i, k);
          b.grad[k] = (b.grad[k] as number) + d;
          for (let j = 0; j < fanIn; j++)
            w.grad[j * fanOut + k] = (w.grad[j * fanOut + k] as number) + input.get(i, j) * d;
        }
      if (l === 0) break;
      // gradient into the input of this layer = output of the previous (dropout, activation, batch norm)
      const da = new Matrix(n, fanIn);
      for (let i = 0; i < n; i++)
        for (let j = 0; j < fanIn; j++) {
          let s = 0;
          for (let k = 0; k < fanOut; k++) s += dz.get(i, k) * (w.data[j * fanOut + k] as number);
          da.set(i, j, s);
        }
      const mask = c.masks[l - 1] as Matrix | null;
      const hPrev = c.act[l - 1] as Matrix;
      const dh = new Matrix(n, fanIn);
      for (let i = 0; i < n; i++)
        for (let j = 0; j < fanIn; j++) {
          const out = this.act(hPrev.get(i, j));
          let g = da.get(i, j) * (mask ? mask.get(i, j) : 1);
          g *= this.actGrad(hPrev.get(i, j), out);
          dh.set(i, j, g);
        }
      if (this.config.batchNorm) {
        const gamma = this.gammas[l - 1] as Tensor;
        const beta = this.betas[l - 1] as Tensor;
        const norm = c.normalized[l - 1] as Matrix;
        const inv = c.invStd[l - 1] as Float64Array;
        const dzPrev = new Matrix(n, fanIn);
        for (let j = 0; j < fanIn; j++) {
          let sumD = 0;
          let sumDn = 0;
          for (let i = 0; i < n; i++) {
            const g = dh.get(i, j);
            gamma.grad[j] = (gamma.grad[j] as number) + g * norm.get(i, j);
            beta.grad[j] = (beta.grad[j] as number) + g;
            sumD += g;
            sumDn += g * norm.get(i, j);
          }
          for (let i = 0; i < n; i++) {
            const dn = dh.get(i, j) * (gamma.data[j] as number);
            // standard batch-norm backward: (1/N) inv (N dn - sum dn - n_hat sum(dn n_hat))
            dzPrev.set(
              i,
              j,
              ((inv[j] as number) / n) *
                (n * dn -
                  (gamma.data[j] as number) * sumD -
                  norm.get(i, j) * (gamma.data[j] as number) * sumDn),
            );
          }
        }
        dz = dzPrev;
      } else dz = dh;
    }
    this.adapterGradients();
  }

  /** One optimiser step on the accumulated gradients. */
  update(optimizer: ToolOptimizer, learningRate: number): void {
    this.step += 1;
    const wd = this.config.weightDecay;
    for (const t of this.tensors()) {
      if (t.frozen) continue;
      for (let i = 0; i < t.data.length; i++) {
        let g = t.grad[i] as number;
        if (t.decay && wd > 0) g += wd * (t.data[i] as number);
        if (optimizer === 'sgd') t.data[i] = (t.data[i] as number) - learningRate * g;
        else if (optimizer === 'momentum') {
          t.m[i] = 0.9 * (t.m[i] as number) + g;
          t.data[i] = (t.data[i] as number) - learningRate * (t.m[i] as number);
        } else {
          t.m[i] = 0.9 * (t.m[i] as number) + 0.1 * g;
          t.v[i] = 0.999 * (t.v[i] as number) + 0.001 * g * g;
          const mh = (t.m[i] as number) / (1 - 0.9 ** this.step);
          const vh = (t.v[i] as number) / (1 - 0.999 ** this.step);
          t.data[i] = (t.data[i] as number) - (learningRate * mh) / (Math.sqrt(vh) + 1e-8);
        }
      }
    }
  }

  predict(x: Matrix): Int32Array {
    const p = this.forward(x, false);
    return Int32Array.from({ length: p.rows }, (_, i) => {
      let best = 0;
      for (let k = 1; k < p.cols; k++) if (p.get(i, k) > p.get(i, best)) best = k;
      return best;
    });
  }

  /** Mean cross-entropy and accuracy in evaluation mode. */
  evaluate(x: Matrix, y: ArrayLike<number>): { loss: number; accuracy: number } {
    const p = this.forward(x, false);
    let loss = 0;
    let ok = 0;
    for (let i = 0; i < p.rows; i++) {
      loss -= Math.log(Math.max(p.get(i, y[i] as number), 1e-300)) / p.rows;
      let best = 0;
      for (let k = 1; k < p.cols; k++) if (p.get(i, k) > p.get(i, best)) best = k;
      if (best === y[i]) ok++;
    }
    return { loss, accuracy: ok / p.rows };
  }

  /** Number of distinct hidden units in the first hidden layer (units with identical weights count once). */
  distinctFirstLayerUnits(tolerance = 1e-9): number {
    const w = this.weights[0] as Tensor;
    const fanIn = this.config.layers[0] as number;
    const fanOut = this.config.layers[1] as number;
    const rows: number[][] = [];
    for (let k = 0; k < fanOut; k++) {
      const col = Array.from({ length: fanIn }, (_, j) => w.data[j * fanOut + k] as number);
      if (!rows.some((r) => r.every((v, j) => Math.abs(v - (col[j] as number)) < tolerance)))
        rows.push(col);
    }
    return rows.length;
  }
}

export interface ToolboxTraining {
  epochs: number;
  batchSize: number;
  learningRate: number;
  optimizer: ToolOptimizer;
  /** Stop when the validation loss has not improved for this many epochs (0 = never). */
  patience: number;
  seed: number;
}

export interface ToolboxEpoch {
  epoch: number;
  trainLoss: number;
  trainAccuracy: number;
  validationLoss: number;
  validationAccuracy: number;
}

export interface ToolboxResult {
  history: ToolboxEpoch[];
  /** Epoch with the best validation loss, and the weights from then if early stopping was on. */
  bestEpoch: number;
  stoppedEarly: boolean;
  model: ToolboxMLP;
}

export interface LabeledSet {
  x: Matrix;
  y: ArrayLike<number>;
}

const subset = (x: Matrix, idx: readonly number[]): Matrix => {
  const out = new Matrix(idx.length, x.cols);
  idx.forEach((r, i) => out.data.set(x.data.subarray(r * x.cols, (r + 1) * x.cols), i * x.cols));
  return out;
};

export function trainToolbox(
  config: ToolboxConfig,
  train: LabeledSet,
  validation: LabeledSet,
  options: ToolboxTraining,
): ToolboxResult {
  const model = new ToolboxMLP(config);
  const rng = createRng(options.seed);
  const n = train.x.rows;
  const history: ToolboxEpoch[] = [];
  let best = Infinity;
  let bestEpoch = 0;
  let snapshot: Float64Array[] | null = null;
  let snapshotStats: { mean: Float64Array[]; variance: Float64Array[] } | null = null;
  let stoppedEarly = false;
  const copyState = () => {
    snapshot = model.tensors().map((t) => Float64Array.from(t.data));
    snapshotStats = {
      mean: model.runMean.map((m) => Float64Array.from(m)),
      variance: model.runVar.map((m) => Float64Array.from(m)),
    };
  };
  for (let epoch = 1; epoch <= options.epochs; epoch++) {
    const order = Array.from({ length: n }, (_, i) => i);
    for (let i = n - 1; i > 0; i--) {
      const j = Math.floor(rng.next() * (i + 1));
      [order[i], order[j]] = [order[j] as number, order[i] as number];
    }
    for (let s = 0; s < n; s += options.batchSize) {
      const idx = order.slice(s, s + options.batchSize);
      if (idx.length < 2 && config.batchNorm) continue;
      const xb = subset(train.x, idx);
      model.zeroGrad();
      model.forward(xb, true, rng);
      model.backward(idx.map((i) => train.y[i] as number));
      model.update(options.optimizer, options.learningRate);
    }
    const tr = model.evaluate(train.x, train.y);
    const va = model.evaluate(validation.x, validation.y);
    history.push({
      epoch,
      trainLoss: tr.loss,
      trainAccuracy: tr.accuracy,
      validationLoss: va.loss,
      validationAccuracy: va.accuracy,
    });
    if (!Number.isFinite(tr.loss)) break;
    if (va.loss < best - 1e-9) {
      best = va.loss;
      bestEpoch = epoch;
      if (options.patience > 0) copyState();
    } else if (options.patience > 0 && epoch - bestEpoch >= options.patience) {
      stoppedEarly = true;
      break;
    }
  }
  if (stoppedEarly && snapshot && snapshotStats) {
    // roll back to the weights of the best epoch
    const tensors = model.tensors();
    (snapshot as Float64Array[]).forEach((d, i) => (tensors[i] as Tensor).data.set(d));
    (snapshotStats as { mean: Float64Array[]; variance: Float64Array[] }).mean.forEach((m, i) =>
      (model.runMean[i] as Float64Array).set(m),
    );
    (snapshotStats as { mean: Float64Array[]; variance: Float64Array[] }).variance.forEach((m, i) =>
      (model.runVar[i] as Float64Array).set(m),
    );
  }
  return { history, bestEpoch, stoppedEarly, model };
}

export interface LrFinderPoint {
  learningRate: number;
  loss: number;
}

/**
 * Learning-rate range test: raise the rate exponentially from `low` to `high` over `steps` mini-batches,
 * recording a smoothed loss. The suggested rate is a tenth of the rate at which the smoothed loss was lowest.
 */
export function learningRateFinder(
  config: ToolboxConfig,
  train: LabeledSet,
  options: {
    low: number;
    high: number;
    steps: number;
    batchSize: number;
    optimizer: ToolOptimizer;
    seed: number;
  },
): { curve: LrFinderPoint[]; suggested: number } {
  const model = new ToolboxMLP(config);
  const rng = createRng(options.seed);
  const n = train.x.rows;
  const curve: LrFinderPoint[] = [];
  let smooth = NaN;
  const ratio = (options.high / options.low) ** (1 / Math.max(options.steps - 1, 1));
  let lr = options.low;
  let bestLoss = Infinity;
  for (let s = 0; s < options.steps; s++) {
    const idx = Array.from({ length: options.batchSize }, () => Math.floor(rng.next() * n));
    model.zeroGrad();
    model.forward(subset(train.x, idx), true, rng);
    const loss = model.backward(idx.map((i) => train.y[i] as number));
    model.update(options.optimizer, lr);
    smooth = Number.isNaN(smooth) ? loss : 0.9 * smooth + 0.1 * loss;
    curve.push({ learningRate: lr, loss: smooth });
    bestLoss = Math.min(bestLoss, smooth);
    if (!Number.isFinite(smooth) || smooth > 4 * Math.max(bestLoss, 0.3) + 2) break;
    lr *= ratio;
  }
  // the loss is lowest just before the rate becomes too large: a tenth of that rate is a safe choice
  let lowest = 0;
  curve.forEach((p, i) => {
    if (p.loss < (curve[lowest] as LrFinderPoint).loss) lowest = i;
  });
  const suggested = Math.max(options.low, (curve[lowest] as LrFinderPoint).learningRate / 10);
  return { curve, suggested };
}
