import {
  PlaygroundSession,
  cloneJson,
  type PointEdit,
  type PlaygroundConfig,
  type SceneData,
  type Snapshot,
  type SnapshotOptions,
  type StepInspection,
} from './session';
import type { Landscape, LandscapeOptions, LayerHistograms } from './landscape';
import type { EpochStats } from './trainer';

/**
 * Typed message protocol between the UI thread and the Web Worker that owns the session.
 * Every request carries an `id`; the matching response echoes it.
 */
export type PlaygroundRequest =
  | { id: number; type: 'init'; config: PlaygroundConfig; snapshot?: SnapshotOptions }
  | { id: number; type: 'update'; patch: Partial<PlaygroundConfig>; snapshot?: SnapshotOptions }
  | { id: number; type: 'reset'; snapshot?: SnapshotOptions }
  | { id: number; type: 'step'; epochs: number; snapshot?: SnapshotOptions }
  | { id: number; type: 'snapshot'; snapshot: SnapshotOptions }
  /** Hand edits of the training points (classification), or `restore` to regenerate the data. */
  | { id: number; type: 'edit'; edits: PointEdit[]; restore?: boolean; snapshot?: SnapshotOptions }
  /** Anatomy of one training step on a single training sample (read-only). */
  | { id: number; type: 'inspect'; sample?: number | 'worst' }
  /** Read-only analyses of the current weights: a loss-landscape slice and/or weight histograms. */
  | { id: number; type: 'analyse'; landscape?: LandscapeOptions; histograms?: number };

export type PlaygroundResponse =
  /** The scene (data, architecture) changed: init, reset or a rebuilding update. */
  | {
      id: number;
      type: 'scene';
      scene: SceneData;
      config: PlaygroundConfig;
      snapshot: Snapshot;
      /** True when only the points changed (a hand edit): the weights and the training history stay. */
      keepHistory: boolean;
    }
  /** Training advanced, or a live hyper-parameter change was applied. */
  | {
      id: number;
      type: 'progress';
      stats: EpochStats[];
      config: PlaygroundConfig;
      snapshot: Snapshot | null;
    }
  | { id: number; type: 'inspection'; inspection: StepInspection; trainSize: number }
  | {
      id: number;
      type: 'analysis';
      landscape: Landscape | null;
      histograms: LayerHistograms[] | null;
      epoch: number;
    }
  | { id: number; type: 'error'; message: string };

/** Transport-agnostic request handler. The worker is a 3-line wrapper around this. */
export class PlaygroundHost {
  private session: PlaygroundSession | null = null;

  handle(request: PlaygroundRequest): PlaygroundResponse {
    const { id } = request;
    try {
      switch (request.type) {
        case 'init': {
          this.session = new PlaygroundSession(request.config);
          return this.scene(id, request.snapshot);
        }
        case 'update': {
          const session = this.require();
          const mode = session.update(request.patch);
          return mode === 'rebuilt'
            ? this.scene(id, request.snapshot)
            : this.progress(id, [], request.snapshot);
        }
        case 'reset': {
          this.require().reset();
          return this.scene(id, request.snapshot);
        }
        case 'step': {
          const epochs = Math.max(1, Math.min(1000, Math.floor(request.epochs)));
          const stats = this.require().step(epochs);
          return this.progress(id, stats, request.snapshot);
        }
        case 'snapshot':
          return this.progress(id, [], request.snapshot);
        case 'edit': {
          const session = this.require();
          if (request.restore) session.restoreData();
          else session.editPoints(request.edits);
          return this.scene(id, request.snapshot, true);
        }
        case 'analyse': {
          const session = this.require();
          return {
            id,
            type: 'analysis',
            landscape: request.landscape ? session.landscape(request.landscape) : null,
            histograms: request.histograms ? session.histograms(request.histograms) : null,
            epoch: session.epoch,
          };
        }
        case 'inspect': {
          const session = this.require();
          return {
            id,
            type: 'inspection',
            inspection: session.inspect(request.sample),
            trainSize: session.trainSize,
          };
        }
      }
    } catch (error) {
      return { id, type: 'error', message: error instanceof Error ? error.message : String(error) };
    }
  }

  private require(): PlaygroundSession {
    if (!this.session) throw new Error('Playground not initialised: send "init" first');
    return this.session;
  }

  private scene(id: number, options?: SnapshotOptions, keepHistory = false): PlaygroundResponse {
    const session = this.require();
    return {
      id,
      type: 'scene',
      scene: session.scene(),
      config: cloneJson(session.config),
      snapshot: session.snapshot(options),
      keepHistory,
    };
  }

  private progress(id: number, stats: EpochStats[], options?: SnapshotOptions): PlaygroundResponse {
    const session = this.require();
    return {
      id,
      type: 'progress',
      stats,
      config: cloneJson(session.config),
      snapshot: options ? session.snapshot(options) : null,
    };
  }
}
