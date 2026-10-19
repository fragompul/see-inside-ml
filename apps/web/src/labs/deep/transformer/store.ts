import { createRng } from '@sim/core';
import { generateNext, type LoadedTransformer, type SamplingOptions } from '@sim/deep';
import { create } from 'zustand';
import { loadTinyGpt } from './assets';
import type { Lesson } from './lessons';

export const DEFAULT_PROMPT = 'Anna found a ball. She';
export const DEFAULT_SAMPLING: SamplingOptions = { temperature: 0.8, topK: 0, topP: 1 };
export const MAX_NEW_TOKENS = 40;
const TICK_MS = 170;

export type Status = 'idle' | 'loading' | 'ready' | 'error';

export interface TransformerState {
  status: Status;
  error: string | null;
  loaded: LoadedTransformer | null;

  /** What the user typed. Editing it discards the generated continuation. */
  prompt: string;
  /** Token ids appended by the model (or chosen by hand). */
  generated: number[];
  sampling: SamplingOptions;
  seed: number;
  maxNew: number;
  playing: boolean;
  /** Id of the guided experiment currently loaded; cleared as soon as the user edits anything. */
  activeLesson: string | null;

  /** Token position the explanations focus on (null = the last one). */
  selected: number | null;
  layer: number;
  head: number;

  load: (loader?: () => Promise<LoadedTransformer>) => Promise<void>;
  setPrompt: (prompt: string) => void;
  setSampling: (patch: Partial<SamplingOptions>) => void;
  setSeed: (seed: number) => void;
  setMaxNew: (n: number) => void;
  /** Samples one token. Returns false when nothing more can be generated. */
  generateOne: () => boolean;
  /** Appends a specific token (the user overruling the sampler). */
  append: (id: number) => void;
  togglePlay: () => void;
  stop: () => void;
  clearGenerated: () => void;
  select: (position: number | null) => void;
  setLayer: (layer: number) => void;
  setHead: (head: number) => void;
  /** Loads a complete setup: prompt, sampling and focus. */
  applyLesson: (lesson: Lesson) => void;
}

let timer: ReturnType<typeof setInterval> | undefined;

const stopTimer = () => {
  if (timer !== undefined) clearInterval(timer);
  timer = undefined;
};

/** Token ids the model sees: the encoded prompt followed by everything generated so far. */
export function currentIds(
  state: Pick<TransformerState, 'loaded' | 'prompt' | 'generated'>,
): number[] {
  if (!state.loaded) return [];
  return [...state.loaded.tokenizer.encode(state.prompt), ...state.generated];
}

/** Each next token is drawn from its own deterministic stream, so a seed replays the same story. */
export const rngFor = (seed: number, position: number) =>
  createRng((Math.imul(seed | 0, 1_000_003) + Math.imul(position, 7919)) >>> 0);

export const useTransformerStore = create<TransformerState>((set, get) => ({
  status: 'idle',
  error: null,
  loaded: null,
  prompt: DEFAULT_PROMPT,
  generated: [],
  sampling: DEFAULT_SAMPLING,
  seed: 7,
  maxNew: 24,
  playing: false,
  activeLesson: null,
  selected: null,
  layer: 0,
  head: 0,

  load: async (loader = loadTinyGpt) => {
    if (get().status === 'loading' || get().status === 'ready') return;
    set({ status: 'loading', error: null });
    try {
      const loaded = await loader();
      set({ loaded, status: 'ready' });
    } catch (error) {
      set({ status: 'error', error: error instanceof Error ? error.message : String(error) });
    }
  },

  setPrompt: (prompt) => {
    stopTimer();
    set({ prompt, generated: [], playing: false, selected: null, activeLesson: null });
  },

  setSampling: (patch) =>
    set((s) => ({ sampling: { ...s.sampling, ...patch }, activeLesson: null })),
  setSeed: (seed) => set({ seed }),
  setMaxNew: (maxNew) => set({ maxNew: Math.max(1, Math.min(MAX_NEW_TOKENS, Math.round(maxNew))) }),

  generateOne: () => {
    const state = get();
    const { loaded } = state;
    if (!loaded) return false;
    const ids = currentIds(state);
    if (ids.length === 0) return false;
    if (ids[ids.length - 1] === loaded.tokenizer.eotId) return false;
    if (state.generated.length >= state.maxNew) return false;
    const step = generateNext(loaded.model, ids, state.sampling, rngFor(state.seed, ids.length));
    const generated = [...state.generated, step.chosen];
    set({ generated, selected: null });
    return step.chosen !== loaded.tokenizer.eotId && generated.length < state.maxNew;
  },

  append: (id) => {
    const state = get();
    const { loaded } = state;
    if (!loaded || id < 0 || id >= loaded.model.config.vocab) return;
    if (currentIds(state).length === 0) return;
    stopTimer();
    set({ generated: [...state.generated, id], selected: null, playing: false });
  },

  togglePlay: () => {
    if (get().playing) {
      get().stop();
      return;
    }
    // Starting again after the story ended or hit the limit begins a fresh continuation.
    const { loaded, generated, maxNew } = get();
    if (
      loaded &&
      (generated.length >= maxNew || generated[generated.length - 1] === loaded.tokenizer.eotId)
    ) {
      set({ generated: [] });
    }
    set({ playing: true });
    stopTimer();
    timer = setInterval(() => {
      if (!get().generateOne()) get().stop();
    }, TICK_MS);
  },

  stop: () => {
    stopTimer();
    set({ playing: false });
  },

  clearGenerated: () => {
    stopTimer();
    set({ generated: [], playing: false, selected: null });
  },

  select: (selected) => set({ selected }),
  setLayer: (layer) => set({ layer }),
  setHead: (head) => set({ head }),

  applyLesson: (lesson) => {
    const { loaded } = get();
    stopTimer();
    const tokens = loaded ? loaded.tokenizer.encode(lesson.prompt).length : 0;
    const focus = lesson.focus < 0 ? tokens + lesson.focus : lesson.focus;
    set((s) => ({
      prompt: lesson.prompt,
      generated: [],
      playing: false,
      activeLesson: lesson.id,
      selected: tokens > 0 && focus >= 0 && focus < tokens ? focus : null,
      layer: lesson.layer ?? s.layer,
      head: lesson.head ?? s.head,
      sampling: { ...DEFAULT_SAMPLING, ...lesson.sampling },
    }));
  },
}));
