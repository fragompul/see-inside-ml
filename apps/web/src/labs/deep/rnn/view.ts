import { type RecurrentTrace } from '@sim/deep';

/**
 * Turns time-major data (`rows × hidden`, one row per step) into unit-major (`hidden × steps`)
 * so a heat-map can run time left to right with one row per neuron, like a diagram of the
 * unrolled network. `firstRow` skips leading rows (the all-zero start state).
 */
export function unitMajor(
  data: ArrayLike<number>,
  steps: number,
  hidden: number,
  firstRow = 0,
): Float64Array {
  const out = new Float64Array(hidden * steps);
  for (let t = 0; t < steps; t++)
    for (let k = 0; k < hidden; k++)
      out[k * steps + t] = data[(t + firstRow) * hidden + k] as number;
  return out;
}

/** Mean of one step's units. */
export function stepMean(data: ArrayLike<number>, step: number, hidden: number): number {
  let sum = 0;
  for (let k = 0; k < hidden; k++) sum += data[step * hidden + k] as number;
  return sum / hidden;
}

/** Decade range `[10^lo, 10^hi]` that holds every positive value, at least `minDecades` wide. */
export function logRange(values: ArrayLike<number>, minDecades = 2): { lo: number; hi: number } {
  let min = Infinity;
  let max = -Infinity;
  for (let i = 0; i < values.length; i++) {
    const v = values[i] as number;
    if (v > 0 && Number.isFinite(v)) {
      min = Math.min(min, v);
      max = Math.max(max, v);
    }
  }
  if (!Number.isFinite(min)) return { lo: -minDecades, hi: 0 };
  let lo = Math.floor(Math.log10(min));
  const hi = Math.max(Math.ceil(Math.log10(max)), lo + 1);
  if (hi - lo < minDecades) lo = hi - minDecades;
  return { lo, hi };
}

/** Position of `value` on a log axis, 0 at `10^lo` and 1 at `10^hi` (clamped). */
export function logPosition(value: number, lo: number, hi: number): number {
  if (!(value > 0)) return 0;
  return Math.max(0, Math.min(1, (Math.log10(value) - lo) / (hi - lo)));
}

/** Indices of the tokens the task's answer actually depends on (to point them out to the learner). */
export function relevantTokens(
  task: 'first' | 'marker' | 'parity',
  tokens: ArrayLike<number>,
): number[] {
  if (task === 'first') return [0];
  if (task === 'parity') return Array.from({ length: tokens.length }, (_, i) => i);
  for (let i = 0; i < tokens.length - 1; i++) if (tokens[i] === 4) return [i, i + 1];
  return [];
}

export interface StepFacts {
  token: number;
  /** Plain-language reading of the gates at this step, or null for a vanilla RNN. */
  gates: { name: string; mean: number; meaning: string }[];
  /** Mean absolute change of the hidden state at this step. */
  change: number;
}

/** What happened at one step, in numbers the explanation text can quote. */
export function stepFacts(trace: RecurrentTrace, step: number): StepFacts {
  const { hidden, h, gates } = trace;
  let change = 0;
  for (let k = 0; k < hidden; k++)
    change += Math.abs((h[(step + 1) * hidden + k] as number) - (h[step * hidden + k] as number));
  const mean = (name: string) => stepMean(gates[name] as Float64Array, step, hidden);
  const out: StepFacts['gates'] = [];
  if (gates.forget) {
    out.push(
      { name: 'forget', mean: mean('forget'), meaning: 'share of the old memory that is kept' },
      { name: 'input', mean: mean('input'), meaning: 'how much new information is written' },
      { name: 'output', mean: mean('output'), meaning: 'how much of the memory is revealed' },
    );
  } else if (gates.update) {
    out.push(
      { name: 'update', mean: mean('update'), meaning: 'share of the old state that is kept' },
      { name: 'reset', mean: mean('reset'), meaning: 'how much of the past feeds the candidate' },
    );
  }
  return { token: trace.tokens[step] as number, gates: out, change: change / hidden };
}
