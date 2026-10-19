import {
  type LandscapeOptions,
  type PointEdit,
  type PlaygroundConfig,
  type PlaygroundRequest,
  type PlaygroundResponse,
  type SnapshotOptions,
} from '@sim/deep';

export type SceneResponse = Extract<PlaygroundResponse, { type: 'scene' }>;
export type ProgressResponse = Extract<PlaygroundResponse, { type: 'progress' }>;
export type InspectionResponse = Extract<PlaygroundResponse, { type: 'inspection' }>;
export type AnalysisResponse = Extract<PlaygroundResponse, { type: 'analysis' }>;

/** Minimal worker-like surface, so tests can plug in an in-process transport. */
export interface Transport {
  postMessage(message: PlaygroundRequest): void;
  onmessage: ((event: { data: PlaygroundResponse }) => void) | null;
  onerror: ((event: { message?: string }) => void) | null;
  terminate(): void;
}

type DistributiveOmit<T, K extends keyof never> = T extends unknown ? Omit<T, K> : never;

interface Pending {
  resolve: (response: PlaygroundResponse) => void;
  reject: (error: Error) => void;
}

export function createWorkerTransport(): Transport {
  return new Worker(new URL('./playground.worker.ts', import.meta.url), {
    type: 'module',
  }) as unknown as Transport;
}

/**
 * Promise-based client for the playground worker. Requests are matched to responses by id;
 * worker-side failures reject the corresponding promise instead of surfacing as uncaught errors.
 */
export class PlaygroundClient {
  private nextId = 1;
  private readonly pending = new Map<number, Pending>();
  private disposed = false;

  constructor(private readonly transport: Transport = createWorkerTransport()) {
    transport.onmessage = (event) => {
      const entry = this.pending.get(event.data.id);
      if (!entry) return;
      this.pending.delete(event.data.id);
      if (event.data.type === 'error') entry.reject(new Error(event.data.message));
      else entry.resolve(event.data);
    };
    transport.onerror = (event) => this.failAll(new Error(event.message ?? 'Worker crashed'));
  }

  init(config: PlaygroundConfig, snapshot?: SnapshotOptions) {
    return this.send<SceneResponse>({ type: 'init', config, snapshot });
  }

  /** Resolves with a `scene` when the change rebuilt the model, else with `progress`. */
  update(patch: Partial<PlaygroundConfig>, snapshot?: SnapshotOptions) {
    return this.send<SceneResponse | ProgressResponse>({ type: 'update', patch, snapshot });
  }

  reset(snapshot?: SnapshotOptions) {
    return this.send<SceneResponse>({ type: 'reset', snapshot });
  }

  step(epochs: number, snapshot?: SnapshotOptions) {
    return this.send<ProgressResponse>({ type: 'step', epochs, snapshot });
  }

  snapshot(snapshot: SnapshotOptions) {
    return this.send<ProgressResponse>({ type: 'snapshot', snapshot });
  }

  /** Forward + backward on one training sample, read-only (powers the step inspector). */
  inspect(sample?: number | 'worst') {
    return this.send<InspectionResponse>({ type: 'inspect', sample });
  }

  /** Hand edits of the training points; the network and its training history are kept. */
  edit(edits: PointEdit[], snapshot?: SnapshotOptions) {
    return this.send<SceneResponse>({ type: 'edit', edits, snapshot });
  }

  /** Regenerates the original data, discarding hand edits (the weights are kept). */
  restoreData(snapshot?: SnapshotOptions) {
    return this.send<SceneResponse>({ type: 'edit', edits: [], restore: true, snapshot });
  }

  /** Loss-landscape slice and/or weight histograms of the current weights (read-only). */
  analyse(request: { landscape?: LandscapeOptions; histograms?: number }) {
    return this.send<AnalysisResponse>({ type: 'analyse', ...request });
  }

  dispose(): void {
    this.disposed = true;
    this.transport.terminate();
    this.failAll(new Error('Playground client disposed'));
  }

  private send<T extends PlaygroundResponse>(
    request: DistributiveOmit<PlaygroundRequest, 'id'>,
  ): Promise<T> {
    if (this.disposed) return Promise.reject(new Error('Playground client disposed'));
    const id = this.nextId++;
    return new Promise<T>((resolve, reject) => {
      this.pending.set(id, { resolve: resolve as Pending['resolve'], reject });
      this.transport.postMessage({ ...request, id } as PlaygroundRequest);
    });
  }

  private failAll(error: Error): void {
    for (const entry of this.pending.values()) entry.reject(error);
    this.pending.clear();
  }
}
