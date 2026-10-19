import {
  DEFAULT_DIFFUSION_LAB,
  DiffusionSession,
  type DiffusionLabConfig,
  type DiffusionStats,
} from '@sim/deep';
import { create } from 'zustand';

export const SPEEDS = [1, 2, 5, 10] as const;
/** Half-width of the plot window and resolution of the arrow field. */
export const EXTENT = 3.5;
export const FIELD_GRID = 13;
/** Cheap views refresh at most this often while training; sampling (costly) less often. */
const PUBLISH_EVERY_MS = 90;
const SAMPLE_EVERY_MS = 400;

export interface Trajectory {
  /** Times visited, T down to 0. */
  times: number[];
  /** `times.length × points × 2` positions. */
  data: Float64Array;
  points: number;
}

interface DiffusionState {
  config: DiffusionLabConfig;
  history: DiffusionStats[];
  iteration: number;
  running: boolean;
  speed: number;

  /** The time shown in the forward-process and score-field views (0 … T). */
  time: number;
  reference: Float64Array | null;
  /** The reference points noised to `time`. */
  noised: Float64Array | null;
  /** Score field at `time`: arrows' origins and vectors. */
  field: { points: Float64Array; score: Float64Array } | null;
  /** Sampling from the fixed noise set with the current model. */
  trajectory: Trajectory | null;
  notice: string | null;
  version: number;

  init: () => void;
  setTime: (time: number) => void;
  patch: (patch: Partial<DiffusionLabConfig>) => void;
  setRunning: (running: boolean) => void;
  toggleRunning: () => void;
  step: (iterations?: number) => void;
  reset: () => void;
  setSpeed: (speed: number) => void;
  /** Recomputes the sampling path (e.g. after the sampler settings changed). */
  resample: () => void;
}

let session: DiffusionSession | null = null;
let rafId = 0;
let lastPublish = 0;
let lastSample = 0;

export const useDiffusionStore = create<DiffusionState>((set, get) => {
  const ensure = (): DiffusionSession => {
    session ??= new DiffusionSession(DEFAULT_DIFFUSION_LAB);
    return session;
  };

  const sampling = (): Trajectory => {
    const s = ensure();
    const result = s.fixedSample();
    return {
      times: result.times,
      data: result.trajectory ?? new Float64Array(0),
      points: result.samples.length / 2,
    };
  };

  const timeViews = (time: number) => {
    const s = ensure();
    return {
      noised: s.forward(time),
      field: s.scoreField(Math.max(1, time), FIELD_GRID, EXTENT),
    };
  };

  const stopLoop = () => {
    cancelAnimationFrame(rafId);
    rafId = 0;
  };

  const publish = (withSampling: boolean) => {
    const s = ensure();
    set({
      ...timeViews(get().time),
      ...(withSampling ? { trajectory: sampling() } : {}),
      history: [...s.history],
      iteration: s.iteration,
      version: get().version + 1,
    });
  };

  const frame = () => {
    if (!get().running) return;
    ensure().train(get().speed);
    const now = performance.now();
    if (now - lastPublish > PUBLISH_EVERY_MS) {
      lastPublish = now;
      const withSampling = now - lastSample > SAMPLE_EVERY_MS;
      if (withSampling) lastSample = now;
      publish(withSampling);
    }
    rafId = requestAnimationFrame(frame);
  };

  return {
    config: DEFAULT_DIFFUSION_LAB,
    history: [],
    iteration: 0,
    running: false,
    speed: 2,
    time: Math.round(DEFAULT_DIFFUSION_LAB.steps / 2),
    reference: null,
    noised: null,
    field: null,
    trajectory: null,
    notice: null,
    version: 0,

    init: () => {
      if (get().trajectory) return;
      const s = ensure();
      set({ config: { ...s.config }, reference: s.reference });
      publish(true);
    },

    setTime: (time) => {
      const T = ensure().config.steps;
      const t = Math.max(0, Math.min(T, Math.round(time)));
      set({ time: t, ...timeViews(t) });
    },

    patch: (patch) => {
      const s = ensure();
      try {
        const mode = s.update(patch);
        if (mode === 'rebuilt') stopLoop();
        const steps = s.config.steps;
        set({
          config: { ...s.config },
          reference: s.reference,
          notice: null,
          running: mode === 'rebuilt' ? false : get().running,
          time: Math.min(get().time, steps),
        });
        if (mode === 'rebuilt' && patch.steps !== undefined) set({ time: Math.round(steps / 2) });
        publish(true);
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

    step: (iterations = 10) => {
      stopLoop();
      set({ running: false });
      ensure().train(iterations);
      publish(true);
    },

    reset: () => {
      stopLoop();
      ensure().reset();
      set({ running: false });
      publish(true);
    },

    setSpeed: (speed) => set({ speed }),

    resample: () => set({ trajectory: sampling(), version: get().version + 1 }),
  };
});

/** Test hook: drop the module-level session so each test starts clean. */
export function resetDiffusionSession(): void {
  cancelAnimationFrame(rafId);
  rafId = 0;
  session = null;
  useDiffusionStore.setState({
    config: DEFAULT_DIFFUSION_LAB,
    history: [],
    iteration: 0,
    running: false,
    time: Math.round(DEFAULT_DIFFUSION_LAB.steps / 2),
    reference: null,
    noised: null,
    field: null,
    trajectory: null,
    notice: null,
    version: 0,
  });
}
