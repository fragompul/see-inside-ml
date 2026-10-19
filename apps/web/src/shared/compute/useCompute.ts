import { useEffect, useState } from 'react';
import { compute } from './client';
import type { TaskInput, TaskName, TaskResult } from './tasks';

/**
 * The result of a heavy task for the given input, or null while it is being computed (and after the input
 * changes, until the new result arrives). Stale results are never shown.
 */
export function useCompute<T extends TaskName>(
  task: T,
  input: TaskInput<T>,
): { result: TaskResult<T> | null; error: string | null } {
  const key = JSON.stringify(input);
  const [state, setState] = useState<{
    key: string;
    result: TaskResult<T> | null;
    error: string | null;
  }>({ key: '', result: null, error: null });

  useEffect(() => {
    let cancelled = false;
    compute(task, input)
      .then((result) => {
        if (!cancelled) setState({ key, result, error: null });
      })
      .catch((error: unknown) => {
        if (!cancelled)
          setState({
            key,
            result: null,
            error: error instanceof Error ? error.message : String(error),
          });
      });
    return () => {
      cancelled = true;
    };
    // the input is identified by its serialised form
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [task, key]);

  return state.key === key
    ? { result: state.result, error: state.error }
    : { result: null, error: null };
}
