import { DEFAULT_GAN_LAB, GanSession, type GanLabConfig, type GanStats } from '@sim/deep';
import { create } from 'zustand';

export const SPEEDS = [1, 2, 5, 10] as const;
/** Half-width of the plot window and resolution of the discriminator map. */
export const EXTENT = 3.5;
export const D_GRID = 40;
export const WARP_GRID = 11;
/** While training, views are republished at most this often. */
const PUBLISH_EVERY_MS = 70;

interface GanState {
  config: GanLabConfig;
  history: GanStats[];
  iteration: number;
  running: boolean;
  /** Training iterations per animation frame. */
  speed: number;

  /** Real points (reference set) and the generator's output for a fixed noise set. */
  real: Float64Array | null;
  fake: Float64Array | null;
  /** D(x) over the plot window, `D_GRID²` values row by row from the top. */
  discriminator: Float64Array | null;
  /** Where the generator sends a regular grid of its first two noise dimensions. */
  warp: Float64Array | null;
  notice: string | null;
  version: number;

  init: () => void;
  patch: (patch: Partial<GanLabConfig>) => void;
  setRunning: (running: boolean) => void;
  toggleRunning: () => void;
  /** Runs `iterations` and pauses. */
  step: (iterations?: number) => void;
  reset: () => void;
  setSpeed: (speed: number) => void;
}

let session: GanSession | null = null;
let rafId = 0;
let lastPublish = 0;

export const useGanStore = create<GanState>((set, get) => {
  const ensure = (): GanSession => {
    session ??= new GanSession(DEFAULT_GAN_LAB);
    return session;
  };

  const views = () => {
    const s = ensure();
    return {
      real: s.reference,
      fake: s.fixedFake(),
      discriminator: s.discriminatorGrid(EXTENT, D_GRID),
      warp: s.warpedGrid(WARP_GRID),
    };
  };

  const stopLoop = () => {
    cancelAnimationFrame(rafId);
    rafId = 0;
  };

  const publish = () => {
    const s = ensure();
    set({
      ...views(),
      history: [...s.history],
      iteration: s.iteration,
      version: get().version + 1,
    });
  };

  const frame = () => {
    if (!get().running) return;
    ensure().train(get().speed);
    const now = performance.now();
    // Training runs every frame; redrawing the views is the costly part, so it is throttled.
    if (now - lastPublish > PUBLISH_EVERY_MS) {
      lastPublish = now;
      publish();
    }
    rafId = requestAnimationFrame(frame);
  };

  return {
    config: DEFAULT_GAN_LAB,
    history: [],
    iteration: 0,
    running: false,
    speed: 2,
    real: null,
    fake: null,
    discriminator: null,
    warp: null,
    notice: null,
    version: 0,

    init: () => {
      if (get().fake) return;
      set({ config: { ...ensure().config } });
      publish();
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

    step: (iterations = 10) => {
      stopLoop();
      set({ running: false });
      ensure().train(iterations);
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
export function resetGanSession(): void {
  cancelAnimationFrame(rafId);
  rafId = 0;
  session = null;
  useGanStore.setState({
    config: DEFAULT_GAN_LAB,
    history: [],
    iteration: 0,
    running: false,
    real: null,
    fake: null,
    discriminator: null,
    warp: null,
    notice: null,
    version: 0,
  });
}
