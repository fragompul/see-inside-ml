import { createRng } from '@sim/core';
import {
  DEFAULT_RNN_LAB,
  RnnSession,
  labelOf,
  randomSequence,
  taskInfo,
  type RecurrentTrace,
  type RnnEpochStats,
  type RnnLabConfig,
} from '@sim/deep';
import { create } from 'zustand';

export const SPEEDS = [1, 2, 4, 8] as const;

export type SequenceSource = 'sample' | 'random' | 'edited';

interface RnnState {
  config: RnnLabConfig;
  history: RnnEpochStats[];
  epoch: number;
  progress: number;
  running: boolean;
  speed: number;

  /** The sequence under the microscope (always `config.seqLen` tokens). */
  tokens: Int32Array;
  source: SequenceSource;
  /** What the task's rule says the answer is (null if an edit broke the rule, e.g. no ★). */
  label: number | null;

  trace: RecurrentTrace | null;
  /** ‖dL/dh_t‖ for the sequence above (null when it has no valid label). */
  gradientNorms: Float64Array | null;
  /** Mean ‖dL/dh_t‖ of the live network over random sequences, refreshed once per epoch. */
  profile: Float64Array | null;
  confusion: number[][] | null;
  /** Accuracy of the live network at other sequence lengths, measured on demand. */
  lengthSweep: { length: number; accuracy: number }[] | null;
  notice: string | null;
  version: number;

  init: () => void;
  cycleToken: (index: number) => void;
  newSequence: () => void;
  loadSample: (index: number) => void;
  patch: (patch: Partial<RnnLabConfig>) => void;
  setRunning: (running: boolean) => void;
  toggleRunning: () => void;
  stepEpoch: () => void;
  reset: () => void;
  setSpeed: (speed: number) => void;
  measureLengths: () => void;
}

let session: RnnSession | null = null;
let rafId = 0;
let sequenceSeed = 1;

/** Lengths probed by the generalisation sweep, trimmed to what the task allows. */
export function sweepLengths(task: RnnLabConfig['task'], trained: number): number[] {
  const { minLength, maxLength } = taskInfo[task];
  const wanted = [trained, 4, 6, 8, 12, 16, 24, 32, 40].filter(
    (n) => n >= minLength && n <= maxLength,
  );
  return [...new Set(wanted)].sort((a, b) => a - b);
}

export const useRnnStore = create<RnnState>((set, get) => {
  const ensure = (): RnnSession => {
    session ??= new RnnSession(DEFAULT_RNN_LAB);
    return session;
  };

  const labelFor = (tokens: Int32Array): number | null => {
    try {
      return labelOf(ensure().config.task, tokens);
    } catch {
      return null;
    }
  };

  /** Recomputes everything that depends on (network, sequence). */
  const derive = (tokens: Int32Array) => {
    const s = ensure();
    const label = labelFor(tokens);
    return {
      label,
      trace: s.trace(tokens),
      gradientNorms: label === null ? null : s.gradientNorms(tokens),
    };
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
    set({
      ...derive(state.tokens),
      history: completed.length ? [...s.history] : state.history,
      epoch: s.epoch,
      progress: s.epochFraction,
      confusion: completed.length ? s.confusion() : state.confusion,
      profile: completed.length ? s.profile() : state.profile,
      version: state.version + 1,
    });
    rafId = requestAnimationFrame(frame);
  };

  const refresh = (extra: Partial<RnnState> = {}) => {
    const s = ensure();
    const tokens = extra.tokens ?? get().tokens;
    set({
      ...extra,
      ...derive(tokens),
      history: [...s.history],
      epoch: s.epoch,
      confusion: s.confusion(),
      profile: s.profile(),
      progress: s.epochFraction,
      version: get().version + 1,
    });
  };

  const freshSequence = (): Int32Array => {
    const { task, seqLen } = ensure().config;
    return randomSequence(task, seqLen, createRng(sequenceSeed++ * 7919));
  };

  return {
    config: DEFAULT_RNN_LAB,
    history: [],
    epoch: 0,
    progress: 0,
    running: false,
    speed: 1,
    tokens: new Int32Array(DEFAULT_RNN_LAB.seqLen),
    source: 'sample',
    label: null,
    trace: null,
    gradientNorms: null,
    profile: null,
    confusion: null,
    lengthSweep: null,
    notice: null,
    version: 0,

    init: () => {
      const s = ensure();
      if (get().trace) return;
      refresh({ tokens: s.sample(0).tokens, source: 'sample', config: { ...s.config } });
    },

    cycleToken: (index) => {
      const s = ensure();
      const tokens = Int32Array.from(get().tokens);
      if (index < 0 || index >= tokens.length) return;
      tokens[index] = ((tokens[index] as number) + 1) % taskInfo[s.config.task].vocab;
      refresh({ tokens, source: 'edited' });
    },

    newSequence: () => refresh({ tokens: freshSequence(), source: 'random' }),

    loadSample: (index) => {
      refresh({ tokens: ensure().sample(index).tokens, source: 'sample' });
    },

    patch: (patch) => {
      const s = ensure();
      try {
        const mode = s.update(patch);
        if (mode === 'rebuilt') stopLoop();
        const extra: Partial<RnnState> = {
          config: { ...s.config },
          notice: null,
          running: mode === 'rebuilt' ? false : get().running,
        };
        if (mode === 'rebuilt') {
          extra.lengthSweep = null;
          if (get().tokens.length !== s.config.seqLen || patch.task) {
            extra.tokens = freshSequence();
            extra.source = 'random';
          }
        }
        set(extra);
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
      set({ running: false, lengthSweep: null });
      refresh();
    },

    setSpeed: (speed) => set({ speed }),

    measureLengths: () => {
      const s = ensure();
      set({
        lengthSweep: sweepLengths(s.config.task, s.config.seqLen).map((length) => ({
          length,
          accuracy: s.accuracyAtLength(length),
        })),
      });
    },
  };
});

/** Test hook: drop the module-level session so each test starts clean. */
export function resetRnnSession(): void {
  cancelAnimationFrame(rafId);
  rafId = 0;
  session = null;
  sequenceSeed = 1;
  useRnnStore.setState({
    config: DEFAULT_RNN_LAB,
    history: [],
    epoch: 0,
    progress: 0,
    running: false,
    tokens: new Int32Array(DEFAULT_RNN_LAB.seqLen),
    source: 'sample',
    label: null,
    trace: null,
    gradientNorms: null,
    profile: null,
    confusion: null,
    lengthSweep: null,
    notice: null,
    version: 0,
  });
}
