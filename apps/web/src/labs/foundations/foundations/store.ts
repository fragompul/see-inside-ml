import { type ActivationName } from '@sim/core';
import {
  DEFAULT_DESCENT,
  Descent,
  bestLine,
  lossLandscape,
  neuronAccuracy,
  neuronData,
  type DescentConfig,
  type DescentPoint,
  type NeuronActivation,
  type NeuronDataset,
  type NeuronParams,
} from '@sim/deep';
import { create } from 'zustand';

export const SPEEDS = [1, 3, 10] as const;
/** Half-width of the weight window of the loss landscape, and its resolution. */
export const RANGE = 8;
export const LAND_GRID = 56;
const PUBLISH_EVERY_MS = 40;

interface FoundationsState {
  dataset: NeuronDataset;
  /** The neuron explored in section 01. */
  neuron: NeuronParams;
  activation: NeuronActivation;
  /** The activation function shown in section 02. */
  activationView: Exclude<ActivationName, 'softmax'>;
  /** What the "find the best line" button last reported. */
  best: { accuracy: number } | null;
  /** The data point whose arithmetic is shown. */
  inspected: number | null;

  descent: DescentConfig;
  path: DescentPoint[];
  gradientNorm: number;
  landscape: Float64Array | null;
  running: boolean;
  /** Optimiser steps per animation frame. */
  speed: number;
  /** Where the ball is put again by "Reset". */
  start: { w1: number; w2: number };
  notice: string | null;
  version: number;

  init: () => void;
  setDataset: (dataset: NeuronDataset) => void;
  setNeuron: (patch: Partial<NeuronParams>) => void;
  setActivation: (activation: NeuronActivation) => void;
  setActivationView: (name: Exclude<ActivationName, 'softmax'>) => void;
  findBest: () => void;
  inspect: (index: number | null) => void;
  patchDescent: (patch: Partial<DescentConfig>) => void;
  /** Puts the ball at a new starting position. */
  place: (w1: number, w2: number) => void;
  setRunning: (running: boolean) => void;
  toggleRunning: () => void;
  stepOnce: (steps?: number) => void;
  /** Brings the ball back to its starting position. */
  reset: () => void;
  setSpeed: (speed: number) => void;
  /** Copies the weights the ball has found into the neuron of section 01. */
  useLearned: () => void;
}

let descent: Descent | null = null;
let rafId = 0;
let lastPublish = 0;
const INITIAL_START = { w1: -5, w2: 5 };

export const useFoundationsStore = create<FoundationsState>((set, get) => {
  const ensure = (): Descent => {
    descent ??= new Descent(DEFAULT_DESCENT, INITIAL_START);
    return descent;
  };

  const landscapeFor = (dataset: NeuronDataset, bias: number) =>
    lossLandscape(neuronData(dataset), bias, RANGE, LAND_GRID);

  const publish = () => {
    const d = ensure();
    set({
      path: [...d.path],
      gradientNorm: d.gradientNorm(),
      version: get().version + 1,
    });
  };

  const stopLoop = () => {
    cancelAnimationFrame(rafId);
    rafId = 0;
  };

  const frame = () => {
    if (!get().running) return;
    const d = ensure();
    for (let i = 0; i < get().speed; i++) d.step();
    const now = performance.now();
    if (now - lastPublish > PUBLISH_EVERY_MS) {
      lastPublish = now;
      publish();
    }
    rafId = requestAnimationFrame(frame);
  };

  return {
    dataset: 'blobs',
    neuron: { w1: 1, w2: 1, b: -1 },
    activation: 'sigmoid',
    activationView: 'sigmoid',
    best: null,
    inspected: null,
    descent: DEFAULT_DESCENT,
    path: [],
    gradientNorm: 0,
    landscape: null,
    running: false,
    speed: 3,
    start: INITIAL_START,
    notice: null,
    version: 0,

    init: () => {
      if (get().landscape) return;
      const d = ensure();
      set({
        descent: { ...d.config },
        landscape: landscapeFor(d.config.dataset, d.config.bias),
      });
      publish();
    },

    setDataset: (dataset) => {
      get().patchDescent({ dataset });
      set({ dataset, best: null, inspected: null });
    },

    setNeuron: (patch) => set({ neuron: { ...get().neuron, ...patch }, best: null }),

    setActivation: (activation) => set({ activation }),

    setActivationView: (activationView) => set({ activationView }),

    findBest: () => {
      const result = bestLine(neuronData(get().dataset));
      set({ neuron: result.params, best: { accuracy: result.accuracy } });
    },

    inspect: (inspected) => set({ inspected }),

    patchDescent: (patch) => {
      const d = ensure();
      try {
        d.update(patch);
        const config = { ...d.config };
        const surface = patch.dataset !== undefined || patch.bias !== undefined;
        set({
          descent: config,
          dataset: config.dataset,
          notice: null,
          ...(surface ? { landscape: landscapeFor(config.dataset, config.bias) } : {}),
        });
        publish();
      } catch (error) {
        set({
          notice: error instanceof Error ? error.message : String(error),
          descent: { ...d.config },
        });
      }
    },

    place: (w1, w2) => {
      stopLoop();
      ensure().place(w1, w2);
      const here = ensure().position;
      set({ running: false, start: { w1: here.w1, w2: here.w2 } });
      publish();
    },

    setRunning: (running) => {
      stopLoop();
      set({ running });
      if (running) rafId = requestAnimationFrame(frame);
    },

    toggleRunning: () => get().setRunning(!get().running),

    stepOnce: (steps = 1) => {
      stopLoop();
      set({ running: false });
      for (let i = 0; i < steps; i++) ensure().step();
      publish();
    },

    reset: () => {
      stopLoop();
      const { start } = get();
      ensure().place(start.w1, start.w2);
      set({ running: false });
      publish();
    },

    setSpeed: (speed) => set({ speed }),

    useLearned: () => {
      const p = ensure().position;
      set({ neuron: { w1: p.w1, w2: p.w2, b: p.b }, activation: 'sigmoid', best: null });
    },
  };
});

/** Accuracy of the neuron of section 01 on the current dataset. */
export function currentAccuracy(dataset: NeuronDataset, neuron: NeuronParams): number {
  return neuronAccuracy(neuronData(dataset), neuron);
}

/** Test hook: drop the module-level session so each test starts clean. */
export function resetFoundationsSession(): void {
  cancelAnimationFrame(rafId);
  rafId = 0;
  descent = null;
  useFoundationsStore.setState({
    dataset: 'blobs',
    neuron: { w1: 1, w2: 1, b: -1 },
    activation: 'sigmoid',
    activationView: 'sigmoid',
    best: null,
    inspected: null,
    descent: DEFAULT_DESCENT,
    path: [],
    gradientNorm: 0,
    landscape: null,
    running: false,
    speed: 3,
    start: INITIAL_START,
    notice: null,
    version: 0,
  });
}
