import {
  CnnSession,
  DEFAULT_CNN_LAB,
  IMAGE_SIZE,
  argmax,
  type CnnEpochStats,
  type CnnLabConfig,
  type CnnTrace,
} from '@sim/deep';
import { create } from 'zustand';
import { inkStats, normalizeDrawing, stroke } from '@/shared/canvas/draw';

export const SPEEDS = [1, 2, 4, 8] as const;
const OCCLUSION_EVERY_MS = 200;

export type ImageSource = 'blank' | 'drawn' | 'sample';

export interface KernelSet {
  conv1: Float64Array[];
  conv2: Float64Array[][];
}

interface CnnState {
  config: CnnLabConfig;
  history: CnnEpochStats[];
  /** Whole epochs completed. */
  epoch: number;
  /** Fraction of the current epoch done, 0..1 (drives a smooth progress bar). */
  progress: number;
  running: boolean;
  /** Batches advanced per animation frame. */
  speed: number;

  /** What is on the pad (raw). */
  image: Float64Array;
  source: ImageSource;
  sampleLabel: number | null;
  /** Centre and scale the drawing before the network reads it. */
  center: boolean;
  showOcclusion: boolean;

  trace: CnnTrace | null;
  kernels: KernelSet | null;
  occlusion: Float64Array | null;
  confusion: number[][] | null;
  notice: string | null;
  /** Bumps whenever the network's weights or structure may have changed. */
  version: number;

  init: () => void;
  beginStroke: (x: number, y: number) => void;
  moveStroke: (x: number, y: number) => void;
  endStroke: () => void;
  clear: () => void;
  loadSample: (index: number) => void;
  randomSample: () => void;
  patch: (patch: Partial<CnnLabConfig>) => void;
  setRunning: (running: boolean) => void;
  toggleRunning: () => void;
  stepEpoch: () => void;
  reset: () => void;
  setSpeed: (speed: number) => void;
  setCenter: (center: boolean) => void;
  setShowOcclusion: (show: boolean) => void;
}

let session: CnnSession | null = null;
let rafId = 0;
let lastPoint: [number, number] | null = null;
let lastOcclusion = 0;

const blankImage = () => new Float64Array(IMAGE_SIZE * IMAGE_SIZE);

/** The live training session (created on first use): analyses such as attacks read its network. */
export function cnnSession(): CnnSession {
  session ??= new CnnSession(DEFAULT_CNN_LAB);
  return session;
}

export const effectiveImage = (image: Float64Array, center: boolean) =>
  center ? normalizeDrawing(image) : image;

const hasInk = (image: Float64Array) => inkStats(image).mass > 1;

export const useCnnStore = create<CnnState>((set, get) => {
  const ensure = (): CnnSession => {
    session ??= new CnnSession(DEFAULT_CNN_LAB);
    return session;
  };

  /** Recomputes everything that depends on (network, image). */
  const derive = (state: Pick<CnnState, 'image' | 'center' | 'showOcclusion' | 'source'>) => {
    const s = ensure();
    const input = effectiveImage(state.image, state.center);
    const ink = state.source !== 'blank' && hasInk(input);
    const trace = s.trace(input);
    const occlusion =
      ink && state.showOcclusion ? s.occlusion(input, argmax(trace.probs), 3) : null;
    return { trace, kernels: s.kernels(), occlusion };
  };

  const stopLoop = () => {
    cancelAnimationFrame(rafId);
    rafId = 0;
  };

  const frame = () => {
    const state = get();
    if (!state.running) return;
    const s = ensure();
    const completed = s.trainBatches(state.speed);
    // The occlusion map is the costly part of a frame: refresh it a few times a second at most.
    const now = performance.now();
    const occlusionDue = state.showOcclusion && now - lastOcclusion > OCCLUSION_EVERY_MS;
    if (occlusionDue) lastOcclusion = now;
    const derived = derive({ ...state, showOcclusion: occlusionDue });
    set({
      trace: derived.trace,
      kernels: derived.kernels,
      occlusion: state.showOcclusion ? (occlusionDue ? derived.occlusion : state.occlusion) : null,
      history: completed.length ? [...s.history] : state.history,
      epoch: s.epoch,
      progress: s.epochFraction,
      confusion: completed.length ? s.confusion() : state.confusion,
      version: state.version + 1,
    });
    rafId = requestAnimationFrame(frame);
  };

  const refresh = (extra: Partial<CnnState> = {}) => {
    const state = { ...get(), ...extra };
    const s = ensure();
    set({
      ...extra,
      ...derive(state),
      history: [...s.history],
      epoch: s.epoch,
      confusion: s.confusion(),
      progress: s.epochFraction,
      version: get().version + 1,
    });
  };

  return {
    config: DEFAULT_CNN_LAB,
    history: [],
    epoch: 0,
    progress: 0,
    running: false,
    speed: 1,
    image: blankImage(),
    source: 'blank',
    sampleLabel: null,
    center: true,
    showOcclusion: false,
    trace: null,
    kernels: null,
    occlusion: null,
    confusion: null,
    notice: null,
    version: 0,

    init: () => {
      const s = ensure();
      if (get().trace) return;
      const { image, label } = s.sample(0);
      refresh({ image, source: 'sample', sampleLabel: label, config: { ...s.config } });
    },

    beginStroke: (x, y) => {
      const image = Float64Array.from(get().image);
      if (get().source !== 'drawn') image.fill(0); // drawing over a sample starts a fresh picture
      lastPoint = [x, y];
      stroke(image, [x, y], [x, y]);
      refresh({ image, source: 'drawn', sampleLabel: null });
    },

    moveStroke: (x, y) => {
      if (!lastPoint) return;
      const image = Float64Array.from(get().image);
      stroke(image, lastPoint, [x, y]);
      lastPoint = [x, y];
      refresh({ image, source: 'drawn' });
    },

    endStroke: () => {
      lastPoint = null;
    },

    clear: () => {
      lastPoint = null;
      refresh({ image: blankImage(), source: 'blank', sampleLabel: null });
    },

    loadSample: (index) => {
      const s = ensure();
      const { image, label } = s.sample(index);
      lastPoint = null;
      refresh({ image, source: 'sample', sampleLabel: label });
    },

    randomSample: () => {
      const s = ensure();
      get().loadSample(Math.floor(Math.random() * s.testCount));
    },

    patch: (patch) => {
      const s = ensure();
      try {
        const mode = s.update(patch);
        if (mode === 'rebuilt') stopLoop();
        set({
          config: { ...s.config },
          notice: null,
          running: mode === 'rebuilt' ? false : get().running,
        });
        refresh();
        if (mode === 'live' && get().running && !rafId) rafId = requestAnimationFrame(frame);
      } catch (error) {
        set({
          notice: error instanceof Error ? error.message : String(error),
          config: { ...s.config },
        });
      }
    },

    setRunning: (running) => {
      stopLoop();
      set({ running });
      if (running) rafId = requestAnimationFrame(frame);
    },

    toggleRunning: () => get().setRunning(!get().running),

    stepEpoch: () => {
      stopLoop();
      set({ running: false });
      ensure().step(1);
      refresh();
    },

    reset: () => {
      stopLoop();
      ensure().reset();
      set({ running: false });
      refresh();
    },

    setSpeed: (speed) => set({ speed }),
    setCenter: (center) => {
      set({ center });
      refresh({ center });
    },
    setShowOcclusion: (showOcclusion) => {
      set({ showOcclusion });
      refresh({ showOcclusion });
    },
  };
});

/** Test hook: drop the module-level session so each test starts clean. */
export function resetCnnSession(): void {
  cancelAnimationFrame(rafId);
  rafId = 0;
  session = null;
  lastPoint = null;
  useCnnStore.setState({
    config: DEFAULT_CNN_LAB,
    history: [],
    epoch: 0,
    progress: 0,
    running: false,
    image: blankImage(),
    source: 'blank',
    sampleLabel: null,
    trace: null,
    kernels: null,
    occlusion: null,
    confusion: null,
    notice: null,
    version: 0,
  });
}
