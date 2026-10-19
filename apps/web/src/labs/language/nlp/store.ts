import { DEFAULT_NLP_LAB, NlpSession, type NlpLabConfig, type NlpStats } from '@sim/deep';
import { create } from 'zustand';

export const SPEEDS = [1, 5, 20, 50] as const;
/** While training, the views are republished at most this often. */
const PUBLISH_EVERY_MS = 120;

export interface Neighbour {
  word: string;
  similarity: number;
}

export interface MapData {
  words: string[];
  /** `words × 2`, row-major. */
  coords: Float64Array;
  explained: number[];
}

export interface AnalogyQuery {
  a: string;
  b: string;
  c: string;
}

interface NlpState {
  config: NlpLabConfig;
  history: NlpStats[];
  step: number;
  epoch: number;
  running: boolean;
  /** Training steps per animation frame. */
  speed: number;
  diverged: boolean;

  map: MapData | null;
  /** The word whose neighbours are shown. */
  selected: string;
  neighbours: Neighbour[];
  query: AnalogyQuery;
  /** Best answers to the analogy query. */
  answers: Neighbour[];
  groups: { id: string; label: string; accuracy: number }[];
  /** Index of the corpus sentence shown in the first section. */
  sentence: number;
  notice: string | null;
  version: number;

  init: () => void;
  select: (word: string) => void;
  setQuery: (patch: Partial<AnalogyQuery>) => void;
  setSentence: (index: number) => void;
  patch: (patch: Partial<NlpLabConfig>) => void;
  setRunning: (running: boolean) => void;
  toggleRunning: () => void;
  stepOnce: (steps?: number) => void;
  reset: () => void;
  setSpeed: (speed: number) => void;
  /** Cosine similarity of two words under the current vectors. */
  similarity: (a: string, b: string) => number;
  /** The session's corpus and counts, for the "from text to pairs" view. */
  corpus: () => string[][];
  pairCount: () => number;
}

let session: NlpSession | null = null;
let rafId = 0;
let lastPublish = 0;

export const useNlpStore = create<NlpState>((set, get) => {
  const ensure = (): NlpSession => {
    session ??= new NlpSession(DEFAULT_NLP_LAB);
    return session;
  };

  const analysis = (selected: string, query: AnalogyQuery) => {
    const s = ensure();
    return {
      map: s.projection(),
      neighbours: s.nearest(selected, 8),
      answers: s.analogy(query.a, query.b, query.c, 3),
      groups: s.analogyAccuracyByGroup(),
    };
  };

  const stopLoop = () => {
    cancelAnimationFrame(rafId);
    rafId = 0;
  };

  const publish = () => {
    const s = ensure();
    const { selected, query } = get();
    set({
      ...analysis(selected, query),
      history: [...s.history],
      step: s.step,
      epoch: s.epoch,
      diverged: s.diverged,
      version: get().version + 1,
    });
  };

  const frame = () => {
    if (!get().running) return;
    const s = ensure();
    s.train(get().speed);
    if (s.diverged) {
      stopLoop();
      set({
        running: false,
        notice: 'Training blew up: the learning rate is too high. Lower it and press Reset.',
      });
      publish();
      return;
    }
    const now = performance.now();
    if (now - lastPublish > PUBLISH_EVERY_MS) {
      lastPublish = now;
      publish();
    }
    rafId = requestAnimationFrame(frame);
  };

  return {
    config: DEFAULT_NLP_LAB,
    history: [],
    step: 0,
    epoch: 0,
    running: false,
    speed: 5,
    diverged: false,
    map: null,
    selected: 'king',
    neighbours: [],
    query: { a: 'man', b: 'king', c: 'woman' },
    answers: [],
    groups: [],
    sentence: 0,
    notice: null,
    version: 0,

    init: () => {
      if (get().map) return;
      set({ config: { ...ensure().config } });
      publish();
    },

    select: (word) => {
      set({ selected: word, neighbours: ensure().nearest(word, 8) });
    },

    setQuery: (patch) => {
      const query = { ...get().query, ...patch };
      set({ query, answers: ensure().analogy(query.a, query.b, query.c, 3) });
    },

    setSentence: (index) => {
      const n = ensure().corpus.length;
      set({ sentence: ((index % n) + n) % n });
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
          sentence: mode === 'rebuilt' ? 0 : get().sentence,
        });
        publish();
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

    stepOnce: (steps = 25) => {
      stopLoop();
      set({ running: false });
      ensure().train(steps);
      if (ensure().diverged)
        set({
          notice: 'Training blew up: the learning rate is too high. Lower it and press Reset.',
        });
      publish();
    },

    reset: () => {
      stopLoop();
      ensure().reset();
      set({ running: false, notice: null });
      publish();
    },

    setSpeed: (speed) => set({ speed }),

    similarity: (a, b) => ensure().similarity(a, b),
    corpus: () => ensure().corpus,
    pairCount: () => ensure().pairCount,
  };
});

/** Test hook: drop the module-level session so each test starts clean. */
export function resetNlpSession(): void {
  cancelAnimationFrame(rafId);
  rafId = 0;
  session = null;
  useNlpStore.setState({
    config: DEFAULT_NLP_LAB,
    history: [],
    step: 0,
    epoch: 0,
    running: false,
    diverged: false,
    map: null,
    selected: 'king',
    neighbours: [],
    query: { a: 'man', b: 'king', c: 'woman' },
    answers: [],
    groups: [],
    sentence: 0,
    notice: null,
    version: 0,
  });
}
