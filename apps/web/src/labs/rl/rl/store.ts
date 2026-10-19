import {
  DEFAULT_RL_LAB,
  GRID_PRESETS,
  KIND_CHAR,
  RlSession,
  type CellKind,
  type EpisodeStats,
  type RlLabConfig,
} from '@sim/deep';
import { create } from 'zustand';

export const SPEEDS = [1, 10, 100, 1000] as const;
/** While training, the views are republished at most this often. */
const PUBLISH_EVERY_MS = 60;

export type Tool = CellKind;

export interface QualityPoint {
  episode: number;
  /** Share of reachable cells where the greedy move is an optimal one. */
  agreement: number;
  /** Mean distance between learned and optimal state values. */
  valueError: number;
}

export interface Snapshot {
  /** `states × 4` action values. */
  q: Float64Array;
  visits: Int32Array;
  /** Optimal state values from value iteration. */
  optimal: Float64Array;
  agent: number;
  path: { cells: number[]; outcome: 'goal' | 'pit' | 'loop' };
  agreement: number;
  valueError: number;
}

interface RlState {
  config: RlLabConfig;
  history: EpisodeStats[];
  episode: number;
  totalSteps: number;
  epsilon: number;
  running: boolean;
  /** Environment steps per animation frame. */
  speed: number;
  snapshot: Snapshot | null;
  /** How close the agent is to optimal, sampled as learning goes on. */
  quality: QualityPoint[];
  /** The cell currently selected for inspection (shows its four action values). */
  inspected: number | null;
  notice: string | null;
  version: number;

  init: () => void;
  patch: (patch: Partial<RlLabConfig>) => void;
  /** Paints one cell of the grid with `kind` (rebuilds the world; learning restarts). */
  paint: (cell: number, kind: Tool) => void;
  loadPreset: (id: string) => void;
  inspect: (cell: number | null) => void;
  setRunning: (running: boolean) => void;
  toggleRunning: () => void;
  /** Takes a number of environment steps and pauses. */
  stepOnce: (steps?: number) => void;
  /** Trains whole episodes (fast-forward) and pauses. */
  episodes: (count: number) => void;
  reset: () => void;
  setSpeed: (speed: number) => void;
}

let session: RlSession | null = null;
let rafId = 0;
let lastPublish = 0;

export const useRlStore = create<RlState>((set, get) => {
  const ensure = (): RlSession => {
    session ??= new RlSession(DEFAULT_RL_LAB);
    return session;
  };

  const snapshot = (): Snapshot => {
    const s = ensure();
    return {
      q: Float64Array.from(s.q),
      visits: Int32Array.from(s.visits),
      optimal: Float64Array.from(s.optimal.values),
      agent: s.agent,
      path: s.greedyPath(),
      agreement: s.agreement(),
      valueError: s.valueError(),
    };
  };

  const stopLoop = () => {
    cancelAnimationFrame(rafId);
    rafId = 0;
  };

  const publish = () => {
    const s = ensure();
    const snap = snapshot();
    const before = get().quality;
    const quality =
      s.episode === 0
        ? []
        : before.at(-1)?.episode === s.episode
          ? before
          : [
              ...before.slice(-1999),
              { episode: s.episode, agreement: snap.agreement, valueError: snap.valueError },
            ];
    set({
      snapshot: snap,
      quality,
      history: [...s.history],
      episode: s.episode,
      totalSteps: s.totalSteps,
      epsilon: s.epsilon,
      version: get().version + 1,
    });
  };

  const frame = () => {
    if (!get().running) return;
    ensure().train(get().speed);
    const now = performance.now();
    if (now - lastPublish > PUBLISH_EVERY_MS) {
      lastPublish = now;
      publish();
    }
    rafId = requestAnimationFrame(frame);
  };

  return {
    config: DEFAULT_RL_LAB,
    history: [],
    episode: 0,
    totalSteps: 0,
    epsilon: DEFAULT_RL_LAB.epsilon,
    running: false,
    speed: 10,
    snapshot: null,
    quality: [],
    inspected: null,
    notice: null,
    version: 0,

    init: () => {
      if (get().snapshot) return;
      set({ config: { ...ensure().config, grid: [...ensure().config.grid] } });
      publish();
    },

    patch: (patch) => {
      const s = ensure();
      try {
        const mode = s.update(patch);
        if (mode === 'rebuilt') stopLoop();
        set({
          config: { ...s.config, grid: [...s.config.grid] },
          notice: null,
          running: mode === 'rebuilt' ? false : get().running,
          inspected: mode === 'rebuilt' && patch.grid ? null : get().inspected,
        });
        publish();
        if (mode === 'live' && get().running && !rafId) rafId = requestAnimationFrame(frame);
      } catch (error) {
        set({
          notice: error instanceof Error ? error.message : String(error),
          config: { ...s.config, grid: [...s.config.grid] },
        });
      }
    },

    paint: (cell, kind) => {
      const { grid } = ensure().config;
      const width = grid[0]?.length ?? 0;
      const chars = grid.map((row) => [...row]);
      const y = Math.floor(cell / width);
      const x = cell % width;
      if (!chars[y] || x >= width) return;
      if (chars[y][x] === KIND_CHAR[kind]) return;
      if (kind === 'start') {
        // there is exactly one start: moving it leaves an empty cell behind
        for (const row of chars)
          for (let i = 0; i < row.length; i++) if (row[i] === 'S') row[i] = '.';
      }
      (chars[y] as string[])[x] = KIND_CHAR[kind];
      get().patch({ grid: chars.map((row) => row.join('')) });
    },

    loadPreset: (id) => {
      const preset = GRID_PRESETS[id];
      if (preset) get().patch({ grid: [...preset.rows] });
    },

    inspect: (cell) => set({ inspected: cell }),

    setRunning: (running) => {
      stopLoop();
      set({ running });
      if (running) rafId = requestAnimationFrame(frame);
    },

    toggleRunning: () => get().setRunning(!get().running),

    stepOnce: (steps = 1) => {
      stopLoop();
      set({ running: false });
      ensure().train(steps);
      publish();
    },

    episodes: (count) => {
      stopLoop();
      set({ running: false });
      ensure().trainEpisodes(count);
      publish();
    },

    reset: () => {
      stopLoop();
      ensure().reset();
      set({ running: false });
      publish();
    },

    setSpeed: (speed) => set({ speed }),
  };
});

/** Test hook: drop the module-level session so each test starts clean. */
export function resetRlSession(): void {
  cancelAnimationFrame(rafId);
  rafId = 0;
  session = null;
  useRlStore.setState({
    config: DEFAULT_RL_LAB,
    history: [],
    episode: 0,
    totalSteps: 0,
    epsilon: DEFAULT_RL_LAB.epsilon,
    running: false,
    snapshot: null,
    quality: [],
    inspected: null,
    notice: null,
    version: 0,
  });
}
