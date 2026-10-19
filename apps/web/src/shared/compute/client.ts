import { TASKS, type TaskInput, type TaskName, type TaskResult } from './tasks';

interface Pending {
  resolve: (value: unknown) => void;
  reject: (error: Error) => void;
}

let worker: Worker | null = null;
let nextId = 1;
const pending = new Map<number, Pending>();

function getWorker(): Worker {
  if (!worker) {
    worker = new Worker(new URL('./compute.worker.ts', import.meta.url), { type: 'module' });
    worker.onmessage = (event: MessageEvent<{ id: number; result?: unknown; error?: string }>) => {
      const entry = pending.get(event.data.id);
      if (!entry) return;
      pending.delete(event.data.id);
      if (event.data.error !== undefined) entry.reject(new Error(event.data.error));
      else entry.resolve(event.data.result);
    };
    worker.onerror = (event) => {
      for (const entry of pending.values())
        entry.reject(new Error(event.message || 'Worker crashed'));
      pending.clear();
      worker = null;
    };
  }
  return worker;
}

/**
 * Runs a task in a Web Worker and resolves with its result. Without Worker support (very old browsers, tests)
 * the task runs on the calling thread, after a macrotask so that the page can paint first.
 */
export function compute<T extends TaskName>(task: T, input: TaskInput<T>): Promise<TaskResult<T>> {
  if (typeof Worker === 'undefined') {
    return new Promise((resolve, reject) =>
      setTimeout(async () => {
        try {
          const run = (await TASKS[task]()) as (input: unknown) => unknown | Promise<unknown>;
          resolve((await run(input)) as TaskResult<T>);
        } catch (error) {
          reject(error instanceof Error ? error : new Error(String(error)));
        }
      }, 30),
    );
  }
  return new Promise((resolve, reject) => {
    const id = nextId++;
    pending.set(id, { resolve: resolve as (v: unknown) => void, reject });
    getWorker().postMessage({ id, task, input });
  });
}
