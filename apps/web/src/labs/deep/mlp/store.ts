import {
  DEFAULT_CONFIG,
  type EpochStats,
  type Landscape,
  type LandscapeOptions,
  type LayerHistograms,
  type PointEdit,
  type PlaygroundConfig,
  type SceneData,
  type Snapshot,
  type SnapshotOptions,
  type StepInspection,
} from '@sim/deep';
import { create } from 'zustand';
import {
  PlaygroundClient,
  type ProgressResponse,
  type SceneResponse,
} from '@/shared/playground/client';

/** Decision plane grid, and the (much coarser) grid of each neuron's mini-map. */
export const GRID_RESOLUTION = 72;
export const NEURON_RESOLUTION = 28;
const SNAPSHOT: SnapshotOptions = {
  resolution: GRID_RESOLUTION,
  neuronMaps: true,
  neuronResolution: NEURON_RESOLUTION,
};

export const SPEEDS = [1, 2, 5, 10, 25, 50] as const;

export type Status = 'idle' | 'booting' | 'ready' | 'error';

interface MlpState {
  status: Status;
  error: string | null;
  config: PlaygroundConfig;
  scene: SceneData | null;
  snapshot: Snapshot | null;
  history: EpochStats[];
  running: boolean;
  /** Epochs advanced per worker round-trip while running. */
  speed: number;
  /** Bumped whenever a new scene (data / architecture) arrives; lets views reset local state. */
  sceneVersion: number;
  /** Recoverable problem (e.g. a rejected setting); the previous state was restored. */
  notice: string | null;
  /** Id of the guided experiment currently loaded; cleared as soon as the user edits anything. */
  activePreset: string | null;

  /** Step inspector: freezes training and dissects one forward/backward pass on one sample. */
  inspecting: boolean;
  inspection: StepInspection | null;
  /** Index (into the training set) of the sample being inspected. */
  inspectSample: number | 'worst' | null;
  /** 0 forward · 1 loss · 2 backward · 3 update */
  inspectStage: number;

  /** Latest weight-space analyses (loss landscape, histograms) and the epoch they describe. */
  analysis: {
    landscape: Landscape | null;
    histograms: LayerHistograms[] | null;
    epoch: number;
  } | null;
  /** Asks the worker to analyse the current weights; results land in `analysis`. */
  requestAnalysis: (request: { landscape?: LandscapeOptions; histograms?: number }) => void;

  /** Add, move or erase training points by hand (classification); training carries on. */
  editPoints: (edits: PointEdit[]) => void;
  restorePoints: () => void;

  boot: (config?: PlaygroundConfig) => void;
  dispose: () => void;
  patch: (patch: Partial<PlaygroundConfig>) => void;
  /** Loads a complete configuration from scratch (fresh weights) and starts training. */
  applyPreset: (id: string, config: PlaygroundConfig, speed?: number) => void;
  setRunning: (running: boolean) => void;
  toggleRunning: () => void;
  step: () => void;
  reset: () => void;
  setSpeed: (speed: number) => void;
  startInspect: (sample?: number | 'worst') => void;
  pickSample: (sample: number) => void;
  setStage: (stage: number) => void;
  stopInspect: () => void;
}

// ───────────────────────────── controller (module scope) ─────────────────────────────
// A single sequential pump owns the worker: config changes, resets and training steps never
// overlap, slider drags coalesce into one update, and the run loop yields to the browser
// between steps so the UI always stays responsive.

let client: PlaygroundClient | null = null;
let generation = 0;
let busy = false;
let pendingPatch: Partial<PlaygroundConfig> | null = null;
let pendingReset = false;
let pendingSteps = 0;
let pendingInspect = false;
let pendingEdits: PointEdit[] = [];
let pendingRestore = false;
let pendingAnalysis: { landscape?: LandscapeOptions; histograms?: number } | null = null;

const nextFrame = () => new Promise<void>((resolve) => requestAnimationFrame(() => resolve()));

export const useMlpStore = create<MlpState>((set, get) => {
  const applyScene = (res: SceneResponse) => {
    set((s) => ({
      scene: res.scene,
      snapshot: res.snapshot,
      config: res.config,
      history: res.keepHistory ? s.history : [],
      analysis: null,
      sceneVersion: s.sceneVersion + 1,
      // A rebuilt scene means different weights and data: refresh the open inspection.
      inspectSample:
        s.inspecting && s.inspectSample !== null
          ? s.inspectSample === 'worst'
            ? 'worst'
            : Math.min(s.inspectSample, res.scene.train.target.length - 1)
          : s.inspectSample,
    }));
    if (get().inspecting) pendingInspect = true;
  };

  const applyProgress = (res: ProgressResponse) =>
    set((s) => ({
      snapshot: res.snapshot ?? s.snapshot,
      history: res.stats.length ? [...s.history, ...res.stats] : s.history,
    }));

  async function pump(gen: number) {
    if (busy || !client) return;
    busy = true;
    try {
      while (gen === generation && client) {
        if (pendingPatch) {
          const patch = pendingPatch;
          pendingPatch = null;
          try {
            const res = await client.update(patch, SNAPSHOT);
            if (gen !== generation) return;
            if (res.type === 'scene') applyScene(res);
            else applyProgress(res);
            set({ notice: null });
            if (get().inspecting) pendingInspect = true; // e.g. the learning rate changed
          } catch (error) {
            // The engine rejected the change: it kept its previous state, so re-sync from it.
            const res = await client.snapshot(SNAPSHOT);
            if (gen !== generation) return;
            applyProgress(res);
            set({
              config: res.config,
              notice: error instanceof Error ? error.message : String(error),
            });
          }
          continue;
        }
        if (pendingReset) {
          pendingReset = false;
          const res = await client.reset(SNAPSHOT);
          if (gen !== generation) return;
          applyScene(res);
          continue;
        }
        if (pendingInspect) {
          pendingInspect = false;
          if (get().inspecting) {
            try {
              const res = await client.inspect(get().inspectSample ?? 0);
              if (gen !== generation) return;
              set({ inspection: res.inspection, inspectSample: res.inspection.sample });
            } catch (error) {
              if (gen !== generation) return;
              set({
                inspecting: false,
                inspection: null,
                notice: error instanceof Error ? error.message : String(error),
              });
            }
          }
          continue;
        }
        if (pendingRestore || pendingEdits.length > 0) {
          const restore = pendingRestore;
          const edits = pendingEdits;
          pendingRestore = false;
          pendingEdits = [];
          try {
            const res = restore
              ? await client.restoreData(SNAPSHOT)
              : await client.edit(edits, SNAPSHOT);
            if (gen !== generation) return;
            applyScene(res);
          } catch (error) {
            if (gen !== generation) return;
            set({ notice: error instanceof Error ? error.message : String(error) });
          }
          continue;
        }
        if (pendingAnalysis) {
          const request = pendingAnalysis;
          pendingAnalysis = null;
          try {
            const res = await client.analyse(request);
            if (gen !== generation) return;
            set((s) => ({
              analysis: {
                landscape: res.landscape ?? s.analysis?.landscape ?? null,
                histograms: res.histograms ?? s.analysis?.histograms ?? null,
                epoch: res.epoch,
              },
            }));
          } catch {
            // an analysis is a nicety: if it fails the rest of the lab carries on
          }
          continue;
        }
        if (pendingSteps > 0 || get().running) {
          const epochs = pendingSteps > 0 ? 1 : get().speed;
          pendingSteps = Math.max(0, pendingSteps - 1);
          const res = await client.step(epochs, SNAPSHOT);
          if (gen !== generation) return;
          applyProgress(res);
          await nextFrame();
          continue;
        }
        break;
      }
    } catch (error) {
      if (gen === generation) {
        set({
          status: 'error',
          running: false,
          error: error instanceof Error ? error.message : String(error),
        });
      }
    } finally {
      busy = false;
      // A request that arrived while we were finishing must not be dropped.
      if (
        gen === generation &&
        (pendingPatch ||
          pendingReset ||
          pendingAnalysis ||
          pendingEdits.length > 0 ||
          pendingRestore ||
          pendingSteps > 0 ||
          get().running)
      ) {
        void pump(gen);
      }
    }
  }

  return {
    status: 'idle',
    error: null,
    config: DEFAULT_CONFIG,
    scene: null,
    snapshot: null,
    history: [],
    running: false,
    speed: 5,
    sceneVersion: 0,
    notice: null,
    activePreset: null,
    inspecting: false,
    inspection: null,
    inspectSample: null,
    inspectStage: 0,
    analysis: null,

    editPoints: (edits) => {
      if (get().status !== 'ready') return;
      // dragging sends a move per pointer event: keep only the latest position of each point
      const merged = [...pendingEdits];
      for (const e of edits) {
        const last = merged[merged.length - 1];
        if (e.op === 'move' && last?.op === 'move' && last.index === e.index)
          merged[merged.length - 1] = e;
        else merged.push(e);
      }
      pendingEdits = merged;
      set({ inspecting: false, inspection: null, activePreset: null });
      void pump(generation);
    },

    restorePoints: () => {
      if (get().status !== 'ready') return;
      pendingEdits = [];
      pendingRestore = true;
      void pump(generation);
    },

    requestAnalysis: (request) => {
      if (get().status !== 'ready') return;
      pendingAnalysis = { ...pendingAnalysis, ...request };
      void pump(generation);
    },

    boot: (initial) => {
      const config = initial ?? get().config;
      generation += 1;
      const gen = generation;
      client?.dispose();
      busy = false;
      pendingPatch = null;
      pendingReset = false;
      pendingSteps = 0;
      pendingInspect = false;
      pendingAnalysis = null;
      pendingEdits = [];
      pendingRestore = false;
      client = new PlaygroundClient();
      set({
        status: 'booting',
        error: null,
        running: false,
        history: [],
        config,
        inspecting: false,
        inspection: null,
      });
      client
        .init(config, SNAPSHOT)
        .then((res) => {
          if (gen !== generation) return;
          applyScene(res);
          set({ status: 'ready' });
        })
        .catch((error: unknown) => {
          if (gen !== generation) return;
          set({ status: 'error', error: error instanceof Error ? error.message : String(error) });
        });
    },

    dispose: () => {
      generation += 1;
      client?.dispose();
      client = null;
      busy = false;
      set({ status: 'idle', running: false });
    },

    patch: (patch) => {
      if (get().status !== 'ready') return;
      pendingPatch = { ...pendingPatch, ...patch };
      set((s) => ({ config: { ...s.config, ...patch }, activePreset: null })); // optimistic
      void pump(generation);
    },

    applyPreset: (id, config, speed) => {
      if (get().status !== 'ready') return;
      pendingPatch = { ...pendingPatch, ...config };
      pendingReset = true;
      set((s) => ({
        config,
        activePreset: id,
        running: true,
        inspecting: false,
        inspection: null,
        speed: speed ?? s.speed,
      }));
      void pump(generation);
    },

    setRunning: (running) => {
      if (get().status !== 'ready') return;
      // Training changes the weights, so any open inspection would be stale.
      set(running ? { running, inspecting: false, inspection: null } : { running });
      if (running) void pump(generation);
    },

    toggleRunning: () => get().setRunning(!get().running),

    step: () => {
      if (get().status !== 'ready') return;
      set({ running: false, inspecting: false, inspection: null });
      pendingSteps += 1;
      void pump(generation);
    },

    reset: () => {
      if (get().status !== 'ready') return;
      pendingReset = true;
      void pump(generation);
    },

    setSpeed: (speed) => set({ speed }),

    startInspect: (sample) => {
      const { status, scene } = get();
      if (status !== 'ready' || !scene) return;
      const index = sample ?? 'worst';
      set({
        running: false,
        inspecting: true,
        inspection: null,
        inspectSample: index,
        inspectStage: 0,
      });
      pendingInspect = true;
      void pump(generation);
    },

    pickSample: (sample) => {
      if (!get().inspecting) return;
      set({ inspectSample: sample });
      pendingInspect = true;
      void pump(generation);
    },

    setStage: (stage) => set({ inspectStage: Math.max(0, Math.min(3, stage)) }),

    stopInspect: () => set({ inspecting: false, inspection: null }),
  };
});
