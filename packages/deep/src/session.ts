import {
  type HIDDEN_ACTIVATIONS,
  defaultLoss,
  isValidLoss,
  lossValue,
  type LossType,
  Matrix,
  selectRows,
  OPTIMIZER_NAMES,
  type OptimizerName,
  createRng,
  LR_SCHEDULES,
  type LrSchedule,
} from '@sim/core';
import {
  DOMAIN,
  MAX_CLASSES,
  MIN_CLASSES,
  datasetInfo,
  generateDataset,
  splitIndices,
  subsetX,
  toTargets,
  type Dataset,
  type DatasetName,
  type Split,
} from './datasets';
import { FEATURES, extractFeatures, isFeatureId, type FeatureId } from './features';
import { NeuralNetwork, type LayerWeights, type NetworkConfig } from './network';
import {
  flattenParams,
  layerHistograms,
  weightSpaceLandscape,
  type Landscape,
  type LandscapeOptions,
  type LayerHistograms,
} from './landscape';
import { Trainer, type EpochStats, type TrainingData } from './trainer';

export type HiddenActivation = (typeof HIDDEN_ACTIVATIONS)[number];
export type RegularizationKind = 'none' | 'l1' | 'l2';

export interface HiddenLayerConfig {
  neurons: number;
  activation: HiddenActivation;
}

/** Everything the playground UI can configure. Serialisable (URL, presets, worker messages). */
export interface PlaygroundConfig {
  dataset: DatasetName;
  /** Class count for classification datasets (clamped to what the dataset supports). */
  classes: number;
  samples: number;
  /** 0..1 */
  noise: number;
  /** Percentage of samples used for training. */
  trainRatio: number;
  features: FeatureId[];
  hidden: HiddenLayerConfig[];
  skipConnections: boolean;
  /** Regression only; classification always uses cross-entropy. */
  loss: LossType;
  optimizer: OptimizerName;
  lr: number;
  schedule: LrSchedule;
  batchSize: number;
  regularization: RegularizationKind;
  regularizationRate: number;
  seed: number;
}

/** Deep copy for JSON-serialisable values (configs). Avoids depending on DOM typings. */
export const cloneJson = <T>(value: T): T => JSON.parse(JSON.stringify(value)) as T;

export const DEFAULT_CONFIG: PlaygroundConfig = {
  dataset: 'circle',
  classes: 2,
  samples: 300,
  noise: 0.15,
  trainRatio: 75,
  features: ['X1', 'X2'],
  hidden: [
    { neurons: 4, activation: 'tanh' },
    { neurons: 3, activation: 'tanh' },
  ],
  skipConnections: false,
  loss: 'mse',
  optimizer: 'adam',
  lr: 0.03,
  schedule: 'constant',
  batchSize: 20,
  regularization: 'none',
  regularizationRate: 0.001,
  seed: 42,
};

export const LIMITS = {
  samples: [20, 1000],
  noise: [0, 1],
  trainRatio: [10, 95],
  hiddenLayers: [0, 6],
  neurons: [1, 16],
  batchSize: [1, 1000],
} as const;

export function validatePlaygroundConfig(c: PlaygroundConfig): void {
  const within = (v: number, [lo, hi]: readonly [number, number], name: string) => {
    if (!Number.isFinite(v) || v < lo || v > hi)
      throw new RangeError(`${name} must be in [${lo}, ${hi}]`);
  };
  if (!(c.dataset in datasetInfo)) throw new RangeError(`Unknown dataset "${c.dataset}"`);
  within(c.samples, LIMITS.samples, 'samples');
  within(c.noise, LIMITS.noise, 'noise');
  within(c.trainRatio, LIMITS.trainRatio, 'trainRatio');
  within(c.hidden.length, LIMITS.hiddenLayers, 'hidden layers');
  c.hidden.forEach((h, i) => within(h.neurons, LIMITS.neurons, `neurons in layer ${i + 1}`));
  within(c.batchSize, LIMITS.batchSize, 'batchSize');
  if (c.features.length === 0) throw new RangeError('Select at least one input feature');
  for (const f of c.features) if (!isFeatureId(f)) throw new RangeError(`Unknown feature "${f}"`);
  if (new Set(c.features).size !== c.features.length) throw new RangeError('Duplicate features');
  if (!(OPTIMIZER_NAMES as readonly string[]).includes(c.optimizer)) {
    throw new RangeError(`Unknown optimizer "${c.optimizer}"`);
  }
  if (!(LR_SCHEDULES as readonly string[]).includes(c.schedule)) {
    throw new RangeError(`Unknown schedule "${c.schedule}"`);
  }
  if (!(c.lr > 0)) throw new RangeError('Learning rate must be positive');
  if (c.regularizationRate < 0) throw new RangeError('Regularisation rate cannot be negative');
  if (datasetInfo[c.dataset].problem === 'regression' && !isValidLoss('regression', c.loss)) {
    throw new RangeError(`Loss "${c.loss}" is not valid for regression`);
  }
}

/** A hand edit of the training points (classification only); `index` counts training samples. */
export type PointEdit =
  | { op: 'add'; x: number; y: number; label: number }
  | { op: 'move'; index: number; x: number; y: number }
  | { op: 'erase'; index: number };

export interface PointSet {
  /** Interleaved x, y. */
  xy: Float32Array;
  /** Class index (classification) or target value (regression). */
  target: Float32Array;
}

/** Static scene data (does not change while training). */
export interface SceneData {
  problem: 'classification' | 'regression';
  numClasses: number;
  domain: number;
  train: PointSet;
  test: PointSet;
  /** Min / max of regression targets, for colour scaling. */
  targetRange: [number, number];
  featureLabels: string[];
  layerSizes: number[];
  parameterCount: number;
  /** Hand edits applied to the training points since the data were generated. */
  edits: number;
}

/**
 * Everything that happens in one training step on a single sample, with the current weights:
 * forward values, the loss, the backpropagated errors (δ) and the resulting gradients.
 * Read-only: it never changes the weights or the optimiser state.
 */
export interface StepInspection {
  /** Index into the training set. */
  sample: number;
  /** Raw plane coordinates of the sample, and its target (class index or value). */
  point: { x: number; y: number; target: number };
  /** Activations per column: [0] = input features, [1..] = each layer's output. */
  activations: number[][];
  /** Pre-activations z per weight layer (column l+1). */
  preActivations: number[][];
  /** The vector each weight layer actually received (features, plus raw inputs with skip connections). */
  layerInputs: number[][];
  /** δ = ∂L/∂z per weight layer. */
  deltas: number[][];
  /** ∂L/∂w per weight layer, shaped like W ([in][out]); ∂L/∂b per layer. */
  weightGrads: number[][][];
  biasGrads: number[][];
  /** Target vector (one-hot for classification, [value] for regression). */
  targets: number[];
  /** Loss of this one sample (no regularisation). */
  loss: number;
  lr: number;
}

export interface SnapshotOptions {
  /** Grid resolution per side for the decision boundary / surface. 0 disables the grid outputs. */
  resolution?: number;
  neuronMaps?: boolean;
  /** Grid resolution for per-neuron maps (defaults to `resolution`; they can be much coarser). */
  neuronResolution?: number;
}

export interface Snapshot {
  stats: EpochStats;
  weights: LayerWeights[];
  boundary: GridField | null;
  /** Feature maps then per-layer neuron maps (at `neuronResolution`); present when requested. */
  inputMaps: Float32Array[] | null;
  neuronMaps: Float32Array[][] | null;
}

/**
 * Scalar fields sampled on a `resolution × resolution` grid of cell centres.
 * Index = row * resolution + col, with x increasing along `col` and y increasing along `row`
 * (row 0 is the bottom of the plane).
 */
export interface GridField {
  resolution: number;
  /** Values per cell: class probabilities for classification, 1 for regression. */
  channels: number;
  values: Float32Array;
}

interface Cache {
  dataKey: string;
  netKey: string;
}

const dataKeyOf = (c: PlaygroundConfig) =>
  JSON.stringify([c.dataset, c.classes, c.samples, c.noise, c.trainRatio, c.seed]);
const netKeyOf = (c: PlaygroundConfig) =>
  JSON.stringify([
    c.features,
    c.hidden,
    c.skipConnections,
    // The loss only matters for regression (classification is always cross-entropy).
    datasetInfo[c.dataset].problem === 'regression' ? c.loss : null,
    c.seed,
  ]);

/** Derives independent, reproducible streams from one user-facing seed. */
const stream = (seed: number, index: number) =>
  createRng((seed + Math.imul(index + 1, 0x9e3779b1)) >>> 0);

/**
 * One live playground: dataset + network + trainer behind a small, serialisable API.
 * The UI never touches engine internals; it talks to this (directly or through the worker).
 */
export class PlaygroundSession {
  private cfg: PlaygroundConfig;
  private cache: Cache = { dataKey: '', netKey: '' };
  private dataset!: Dataset;
  private split!: Split;
  private network!: NeuralNetwork;
  private trainer!: Trainer;
  private trainTargets!: Matrix;
  private featuresTrain!: Matrix;
  private featuresTest!: Matrix;
  private trainingData!: TrainingData;
  /** Number of hand edits applied since the data were last generated. */
  edits = 0;
  history: EpochStats[] = [];
  /** Weights after (a thinned selection of) the epochs so far, oldest first, starting from the initial ones. */
  private trajectory: Float64Array[] = [];
  private trajectoryStride = 1;

  constructor(config: PlaygroundConfig) {
    validatePlaygroundConfig(config);
    this.cfg = cloneJson(config);
    this.build(true);
  }

  get config(): Readonly<PlaygroundConfig> {
    return this.cfg;
  }

  get epoch(): number {
    return this.trainer.epoch;
  }

  /**
   * Applies a configuration change. Hyper-parameters (optimiser, LR, batch size, regularisation)
   * change live without losing progress; anything that alters data or architecture rebuilds.
   */
  update(patch: Partial<PlaygroundConfig>): 'live' | 'rebuilt' {
    const next: PlaygroundConfig = { ...this.cfg, ...cloneJson(patch) };
    validatePlaygroundConfig(next);
    const dataChanged = dataKeyOf(next) !== this.cache.dataKey;
    const netChanged = netKeyOf(next) !== this.cache.netKey;
    this.cfg = next;
    if (dataChanged || netChanged) {
      this.build(dataChanged);
      return 'rebuilt';
    }
    this.trainer.configure(this.trainerSettings());
    return 'live';
  }

  /** Re-initialise weights and history, keeping the same seed, data and settings. */
  reset(): void {
    this.build(false);
  }

  /** Trains for `epochs` epochs and returns the per-epoch statistics. */
  step(epochs = 1): EpochStats[] {
    const out: EpochStats[] = [];
    for (let i = 0; i < epochs; i++) {
      const stats = this.trainer.runEpoch();
      this.history.push(stats);
      out.push(stats);
      this.recordWeights();
    }
    return out;
  }

  /** Keeps at most ~160 weight snapshots of the training path by thinning old ones as it grows. */
  private recordWeights(): void {
    if (this.history.length % this.trajectoryStride !== 0) return;
    this.trajectory.push(flattenParams(this.network));
    if (this.trajectory.length > 160) {
      this.trajectory = this.trajectory.filter((_, i) => i % 2 === 0);
      this.trajectoryStride *= 2;
    }
  }

  /** The loss over a plane through the current weights, with the training path drawn onto it. */
  landscape(options: LandscapeOptions = {}): Landscape {
    return weightSpaceLandscape(
      this.network,
      this.featuresTrain,
      this.trainTargets,
      this.trajectory,
      options,
    );
  }

  /** Distribution of the weights and of the latest gradients of every layer. */
  histograms(bins = 24): LayerHistograms[] {
    return layerHistograms(this.network, bins);
  }

  scene(): SceneData {
    const { dataset, split } = this;
    const pack = (indices: Uint32Array): PointSet => {
      const xy = new Float32Array(indices.length * 2);
      const target = new Float32Array(indices.length);
      indices.forEach((src, i) => {
        xy[2 * i] = dataset.X.get(src, 0);
        xy[2 * i + 1] = dataset.X.get(src, 1);
        target[i] =
          dataset.problem === 'classification'
            ? (dataset.labels[src] as number)
            : (dataset.values[src] as number);
      });
      return { xy, target };
    };
    let lo = 0;
    let hi = 0;
    if (dataset.problem === 'regression') {
      lo = Math.min(...dataset.values);
      hi = Math.max(...dataset.values);
    }
    return {
      problem: dataset.problem,
      numClasses: dataset.numClasses,
      domain: DOMAIN,
      train: pack(split.trainIndices),
      test: pack(split.testIndices),
      targetRange: [lo, hi],
      featureLabels: this.cfg.features.map((id) => FEATURES.find((f) => f.id === id)?.label ?? id),
      layerSizes: this.network.config.layerSizes,
      parameterCount: this.network.parameterCount,
      edits: this.edits,
    };
  }

  snapshot(options: SnapshotOptions = {}): Snapshot {
    const resolution = options.resolution ?? 0;
    const stats = this.trainer.evaluate();
    let boundary: GridField | null = null;
    let inputMaps: Float32Array[] | null = null;
    let neuronMaps: Float32Array[][] | null = null;

    if (resolution > 0) {
      const grid = gridCoordinates(resolution);
      const out = this.network.trace(extractFeatures(grid, this.cfg.features)).at(-1) as Matrix;
      boundary = { resolution, channels: out.cols, values: Float32Array.from(out.data) };
    }
    if (options.neuronMaps) {
      const neuronRes = options.neuronResolution ?? resolution;
      if (neuronRes > 0) {
        const features = extractFeatures(gridCoordinates(neuronRes), this.cfg.features);
        inputMaps = columnsOf(features);
        neuronMaps = this.network.trace(features).map(columnsOf);
      }
    }
    return { stats, weights: this.network.getWeights(), boundary, inputMaps, neuronMaps };
  }

  /**
   * Runs one forward + backward pass on a single training sample and reports every intermediate
   * quantity. Gradients are those of that sample alone (a real step averages a mini-batch).
   */
  inspect(sample?: number | 'worst'): StepInspection {
    const n = this.featuresTrain.rows;
    const index =
      sample === 'worst' ? this.worstSample() : sample === undefined ? 0 : Math.floor(sample);
    if (!Number.isInteger(index) || index < 0 || index >= n) {
      throw new RangeError(`Sample must be in [0, ${n - 1}]`);
    }
    const x = selectRows(this.featuresTrain, [index]);
    const y = selectRows(this.trainTargets, [index]);
    const out = this.network.forward(x);
    this.network.backward(y, out);

    const layers = this.network.layers;
    const row = (m: Matrix | null) => Array.from((m as Matrix).data);
    const original = this.split.trainIndices[index] as number;
    return {
      sample: index,
      point: {
        x: this.dataset.X.get(original, 0),
        y: this.dataset.X.get(original, 1),
        target:
          this.dataset.problem === 'classification'
            ? (this.dataset.labels[original] as number)
            : (this.dataset.values[original] as number),
      },
      activations: [Array.from(x.data), ...layers.map((l) => row(l.a))],
      preActivations: layers.map((l) => row(l.z)),
      layerInputs: layers.map((l) => row(l.inputs)),
      deltas: layers.map((l) => row(l.delta)),
      weightGrads: layers.map((l) =>
        Array.from({ length: l.inputDim }, (_, i) =>
          Array.from(l.weightParam.grad.subarray(i * l.outputDim, (i + 1) * l.outputDim)),
        ),
      ),
      biasGrads: layers.map((l) => Array.from(l.biasParam.grad)),
      targets: Array.from(y.data),
      loss: this.network.evaluate(out, y).loss,
      lr: this.trainer.currentLr(),
    };
  }

  /**
   * Adds, moves or erases training points while keeping the network and its progress. The test set
   * is never touched. Coordinates are clamped to the plane.
   */
  editPoints(edits: readonly PointEdit[]): void {
    const ds = this.dataset;
    if (ds.problem !== 'classification')
      throw new Error('Points can only be edited on classification data');
    let X = ds.X;
    let labels = ds.labels;
    let train = Array.from(this.split.trainIndices);
    const clamp = (v: number) => Math.max(-DOMAIN, Math.min(DOMAIN, v));
    for (const e of edits) {
      if (e.op === 'add') {
        if (!Number.isInteger(e.label) || e.label < 0 || e.label >= ds.numClasses)
          throw new RangeError(`Class must be 0 … ${ds.numClasses - 1}`);
        const grown = new Matrix(X.rows + 1, 2);
        grown.data.set(X.data, 0);
        grown.set(X.rows, 0, clamp(e.x));
        grown.set(X.rows, 1, clamp(e.y));
        const l = new Int32Array(labels.length + 1);
        l.set(labels, 0);
        l[labels.length] = e.label;
        train.push(X.rows);
        X = grown;
        labels = l;
      } else {
        if (!Number.isInteger(e.index) || e.index < 0 || e.index >= train.length)
          throw new RangeError(`Training point must be in [0, ${train.length - 1}]`);
        if (e.op === 'erase') {
          if (train.length <= 2) throw new RangeError('Keep at least two training points');
          train = train.filter((_, i) => i !== e.index);
        } else {
          const row = train[e.index] as number;
          X = new Matrix(X.rows, 2, Float64Array.from(X.data));
          X.set(row, 0, clamp(e.x));
          X.set(row, 1, clamp(e.y));
        }
      }
    }
    this.dataset = { ...ds, X, labels };
    this.split = { ...this.split, trainIndices: Uint32Array.from(train) };
    this.edits += edits.length;
    this.refreshTrainingSet();
  }

  /** Throws the hand edits away and regenerates the original data (the network keeps its weights). */
  restoreData(): void {
    const c = this.cfg;
    const classes = Math.min(MAX_CLASSES, Math.max(MIN_CLASSES, Math.round(c.classes)));
    this.dataset = generateDataset(
      c.dataset,
      { samples: c.samples, noise: c.noise, classes },
      stream(c.seed, 0),
    );
    this.split = splitIndices(this.dataset.X.rows, c.trainRatio, stream(c.seed, 1));
    this.featuresTest = extractFeatures(subsetX(this.dataset, this.split.testIndices), c.features);
    this.trainingData.Xtest = this.featuresTest;
    this.trainingData.ytest = toTargets(this.dataset, this.split.testIndices);
    this.edits = 0;
    this.refreshTrainingSet();
  }

  private refreshTrainingSet(): void {
    const { dataset, split } = this;
    this.featuresTrain = extractFeatures(subsetX(dataset, split.trainIndices), this.cfg.features);
    this.trainTargets = toTargets(dataset, split.trainIndices);
    this.trainingData.Xtrain = this.featuresTrain;
    this.trainingData.ytrain = this.trainTargets;
  }

  /** Index of the training sample with the highest loss right now (the most instructive one). */
  worstSample(): number {
    const out = this.network.forward(this.featuresTrain, false);
    let worst = 0;
    let worstLoss = -Infinity;
    for (let i = 0; i < out.rows; i++) {
      const loss = lossValue(
        this.network.loss,
        selectRows(out, [i]),
        selectRows(this.trainTargets, [i]),
      );
      if (loss > worstLoss) {
        worstLoss = loss;
        worst = i;
      }
    }
    return worst;
  }

  /** Number of training samples (valid range for {@link inspect}). */
  get trainSize(): number {
    return this.featuresTrain.rows;
  }

  private trainerSettings() {
    const c = this.cfg;
    return {
      optimizer: c.optimizer,
      lr: c.lr,
      schedule: c.schedule,
      batchSize: c.batchSize,
      l1: c.regularization === 'l1' ? c.regularizationRate : 0,
      l2: c.regularization === 'l2' ? c.regularizationRate : 0,
    };
  }

  private build(regenerateData: boolean): void {
    const c = this.cfg;
    if (regenerateData) {
      const classes = Math.min(MAX_CLASSES, Math.max(MIN_CLASSES, Math.round(c.classes)));
      this.dataset = generateDataset(
        c.dataset,
        { samples: c.samples, noise: c.noise, classes },
        stream(c.seed, 0),
      );
      this.split = splitIndices(this.dataset.X.rows, c.trainRatio, stream(c.seed, 1));
    }
    const { dataset, split } = this;
    const problem = dataset.problem;
    const outputs = problem === 'classification' ? dataset.numClasses : 1;
    const networkConfig: NetworkConfig = {
      layerSizes: [c.features.length, ...c.hidden.map((h) => h.neurons), outputs],
      activations: [
        ...c.hidden.map((h) => h.activation),
        problem === 'classification' ? 'softmax' : 'linear',
      ],
      problem,
      loss: problem === 'classification' ? defaultLoss(problem) : c.loss,
      skipConnections: c.skipConnections,
    };
    this.network = new NeuralNetwork(networkConfig, stream(c.seed, 2));

    this.featuresTrain = extractFeatures(subsetX(dataset, split.trainIndices), c.features);
    this.featuresTest = extractFeatures(subsetX(dataset, split.testIndices), c.features);
    this.trainTargets = toTargets(dataset, split.trainIndices);
    this.trainingData = {
      Xtrain: this.featuresTrain,
      ytrain: this.trainTargets,
      Xtest: this.featuresTest,
      ytest: toTargets(dataset, split.testIndices),
    };
    this.trainer = new Trainer(
      this.network,
      this.trainingData,
      this.trainerSettings(),
      stream(c.seed, 3),
    );
    this.history = [];
    if (regenerateData) this.edits = 0;
    this.trajectory = [flattenParams(this.network)];
    this.trajectoryStride = 1;
    this.cache = { dataKey: dataKeyOf(c), netKey: netKeyOf(c) };
  }
}

/** n×2 matrix of grid-cell centres, ordered row-major (see {@link GridField}). */
export function gridCoordinates(resolution: number): Matrix {
  const grid = new Matrix(resolution * resolution, 2);
  const cell = (2 * DOMAIN) / resolution;
  for (let row = 0; row < resolution; row++) {
    for (let col = 0; col < resolution; col++) {
      const i = row * resolution + col;
      grid.set(i, 0, -DOMAIN + (col + 0.5) * cell);
      grid.set(i, 1, -DOMAIN + (row + 0.5) * cell);
    }
  }
  return grid;
}

function columnsOf(m: Matrix): Float32Array[] {
  const cols: Float32Array[] = [];
  for (let j = 0; j < m.cols; j++) {
    const col = new Float32Array(m.rows);
    for (let i = 0; i < m.rows; i++) col[i] = m.get(i, j);
    cols.push(col);
  }
  return cols;
}
