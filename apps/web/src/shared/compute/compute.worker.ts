import { TASKS, type TaskName } from './tasks';

interface Request {
  id: number;
  task: TaskName;
  input: unknown;
}

self.onmessage = async (event: MessageEvent<Request>) => {
  const { id, task, input } = event.data;
  try {
    const run = (await TASKS[task]()) as (input: unknown) => unknown | Promise<unknown>;
    self.postMessage({ id, result: await run(input) });
  } catch (error) {
    self.postMessage({ id, error: error instanceof Error ? error.message : String(error) });
  }
};
