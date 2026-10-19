import { Matrix, createRng } from '@sim/core';
import {
  DEFAULT_TOOLBOX,
  generateDigits,
  trainProbe,
  trainToolbox,
  type LoadedTransformer,
  type ToolboxMLP,
} from '@sim/deep';

export type ProbedModel = 'digits' | 'story';

export interface ProbingScenario {
  model: ProbedModel;
}

export const DEFAULT_PROBING: ProbingScenario = { model: 'digits' };

export interface ProbeScore {
  name: string;
  /** Accuracy on held-out examples, per layer. */
  test: number[];
  /** What a classifier would score by always guessing the most common class. */
  chance: number;
}

export interface ProbingResult {
  scenario: ProbingScenario;
  layers: string[];
  /** Real properties read from each layer. */
  tasks: ProbeScore[];
  /** The control task: random labels. Train accuracy shows how much the probe can memorise; test accuracy should be chance. */
  controlTrain: number[];
  controlTest: number[];
  /** Widths of the layers (features available to the probe). */
  widths: number[];
}

let storyLoader: (() => Promise<LoadedTransformer>) | null = null;
/** Tells the story probe where to get the tiny GPT (the browser fetches it; tests read it from disk). */
export const setStoryModelLoader = (loader: () => Promise<LoadedTransformer>): void => {
  storyLoader = loader;
};

// ------------------------------------------------------------------------------------------------
// digits

const SIDE = 16;

/** Tercile (0, 1, 2) of each value among all values. */
const terciles = (values: number[]): number[] => {
  const sorted = [...values].sort((a, b) => a - b);
  const lo = sorted[Math.floor(values.length / 3)] as number;
  const hi = sorted[Math.floor((2 * values.length) / 3)] as number;
  return values.map((v) => (v < lo ? 0 : v < hi ? 1 : 2));
};

const inkOf = (images: Float64Array, count: number) =>
  Array.from({ length: count }, (_, i) => {
    let s = 0;
    for (let p = 0; p < SIDE * SIDE; p++) s += images[i * 256 + p] as number;
    return s;
  });

const centreXOf = (images: Float64Array, count: number) =>
  Array.from({ length: count }, (_, i) => {
    let m = 0;
    let x = 0;
    for (let p = 0; p < 256; p++) {
      const v = images[i * 256 + p] as number;
      m += v;
      x += v * (p % SIDE);
    }
    return m > 0 ? x / m : 0;
  });

let digitNet: ToolboxMLP | null = null;
function trainedDigits(): ToolboxMLP {
  if (!digitNet) {
    const make = (perClass: number, seed: number) => {
      const set = generateDigits(perClass, createRng(seed));
      return {
        x: new Matrix(set.count, 256, Float64Array.from(set.images)),
        y: Array.from(set.labels),
      };
    };
    digitNet = trainToolbox(
      { ...DEFAULT_TOOLBOX, layers: [256, 64, 32, 16, 10], seed: 3 },
      make(40, 11),
      make(15, 12),
      { epochs: 25, batchSize: 16, learningRate: 0.005, optimizer: 'adam', patience: 0, seed: 3 },
    ).model;
  }
  return digitNet;
}

const shareOfMostCommon = (labels: number[], classes: number) =>
  Math.max(
    ...Array.from(
      { length: classes },
      (_, c) => labels.filter((v) => v === c).length / labels.length,
    ),
  );

function probeDigits(): ProbingResult {
  const net = trainedDigits();
  const train = generateDigits(60, createRng(21));
  const test = generateDigits(30, createRng(22));
  const matrix = (s: typeof train) => new Matrix(s.count, 256, Float64Array.from(s.images));
  const trainLayers = [matrix(train), ...net.hiddenActivations(matrix(train))];
  const testLayers = [matrix(test), ...net.hiddenActivations(matrix(test))];
  const tasks = [
    {
      name: 'Which digit',
      train: Array.from(train.labels),
      test: Array.from(test.labels),
      classes: 10,
    },
    {
      name: 'How much ink',
      train: terciles(inkOf(train.images, train.count)),
      test: terciles(inkOf(test.images, test.count)),
      classes: 3,
    },
    {
      name: 'Where it sits',
      train: terciles(centreXOf(train.images, train.count)),
      test: terciles(centreXOf(test.images, test.count)),
      classes: 3,
    },
  ];
  const rng = createRng(31);
  const randomTrain = Array.from({ length: train.count }, () => Math.floor(rng.next() * 10));
  const randomTest = Array.from({ length: test.count }, () => Math.floor(rng.next() * 10));
  const scores: ProbeScore[] = tasks.map((t) => ({
    name: t.name,
    test: [],
    chance: shareOfMostCommon(t.test, t.classes),
  }));
  const controlTrain: number[] = [];
  const controlTest: number[] = [];
  trainLayers.forEach((layer, l) => {
    const dim = layer.cols;
    const heldOut = testLayers[l] as Matrix;
    tasks.forEach((t, k) => {
      (scores[k] as ProbeScore).test.push(
        trainProbe(layer.data, t.train, heldOut.data, t.test, dim, t.classes).testAccuracy,
      );
    });
    const c = trainProbe(layer.data, randomTrain, heldOut.data, randomTest, dim, 10);
    controlTrain.push(c.trainAccuracy);
    controlTest.push(c.testAccuracy);
  });
  return {
    scenario: { model: 'digits' },
    layers: ['pixels', 'hidden 1', 'hidden 2', 'hidden 3'],
    tasks: scores,
    controlTrain,
    controlTest,
    widths: trainLayers.map((m) => m.cols),
  };
}

// ------------------------------------------------------------------------------------------------
// story model

const FEMALE = ['Anna', 'Mia', 'Eva', 'Lily', 'Nora', 'Zoe', 'Ruby', 'Ella'];
const MALE = ['Tom', 'Leo', 'Sam', 'Max', 'Ben', 'Jack', 'Finn', 'Oscar'];
const OBJECTS = ['ball', 'book', 'apple', 'key', 'hat', 'cake', 'kite', 'lamp'];

interface Row {
  vectors: number[][];
  name: number;
  gender: number;
  object: number;
}

async function probeStory(): Promise<ProbingResult> {
  if (!storyLoader) throw new Error('No loader for the story model');
  const { model, tokenizer } = await storyLoader();
  const d = model.config.d;
  const names = [...FEMALE.map((n) => ({ n, g: 0 })), ...MALE.map((n) => ({ n, g: 1 }))];
  const atStop: Row[] = [];
  const atName: Row[] = [];
  names.forEach(({ n, g }, ni) =>
    OBJECTS.forEach((o, oi) => {
      const ids = tokenizer.encode(`${n} found a ${o}.`);
      const trace = model.forward(ids);
      const at = (t: number) => [
        Array.from(trace.embed.data.slice(t * d, (t + 1) * d)),
        ...trace.layers.map((l) => Array.from(l.residOut.data.slice(t * d, (t + 1) * d))),
      ];
      atStop.push({ vectors: at(ids.length - 1), name: ni, gender: g, object: oi });
      atName.push({ vectors: at(0), name: ni, gender: g, object: oi });
    }),
  );
  const layers = ['embeddings', 'after layer 1', 'after layer 2'];
  // train on 12 names and test on 4 that the probe has never seen (two of each gender)
  const heldOut = new Set([0, 1, 8, 9]);
  const split = (rows: Row[], l: number) => {
    const tr = rows.filter((r) => !heldOut.has(r.name));
    const te = rows.filter((r) => heldOut.has(r.name));
    return {
      trainX: Float64Array.from(tr.flatMap((r) => r.vectors[l] as number[])),
      testX: Float64Array.from(te.flatMap((r) => r.vectors[l] as number[])),
      trainGender: tr.map((r) => r.gender),
      testGender: te.map((r) => r.gender),
      trainObject: tr.map((r) => r.object),
      testObject: te.map((r) => r.object),
    };
  };
  const rng = createRng(41);
  const name: ProbeScore = { name: 'Gender of the name, read at the name', test: [], chance: 0.5 };
  const stop: ProbeScore = {
    name: 'Gender of the name, read at the full stop',
    test: [],
    chance: 0.5,
  };
  const object: ProbeScore = {
    name: 'Which object, read at the full stop',
    test: [],
    chance: 1 / OBJECTS.length,
  };
  const controlTrain: number[] = [];
  const controlTest: number[] = [];
  layers.forEach((_, l) => {
    const s = split(atStop, l);
    const n = split(atName, l);
    name.test.push(trainProbe(n.trainX, n.trainGender, n.testX, n.testGender, d, 2).testAccuracy);
    stop.test.push(trainProbe(s.trainX, s.trainGender, s.testX, s.testGender, d, 2).testAccuracy);
    object.test.push(
      trainProbe(s.trainX, s.trainObject, s.testX, s.testObject, d, OBJECTS.length).testAccuracy,
    );
    const randomTrain = s.trainGender.map(() => (rng.next() < 0.5 ? 0 : 1));
    const randomTest = s.testGender.map(() => (rng.next() < 0.5 ? 0 : 1));
    const c = trainProbe(s.trainX, randomTrain, s.testX, randomTest, d, 2);
    controlTrain.push(c.trainAccuracy);
    controlTest.push(c.testAccuracy);
  });
  return {
    scenario: { model: 'story' },
    layers,
    tasks: [name, stop, object],
    controlTrain,
    controlTest,
    widths: layers.map(() => d),
  };
}

export async function runProbing(s: ProbingScenario): Promise<ProbingResult> {
  return s.model === 'digits' ? probeDigits() : probeStory();
}
