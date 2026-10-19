/**
 * Declarative description of a model's controls. A lab lists its parameters once and `ParamPanel`
 * draws them, so every classical model gets the same, consistent set of knobs.
 */
export type ParamValue = number | string | boolean;
export type ParamValues = Record<string, ParamValue>;

interface Common {
  key: string;
  label: string;
  /** One-line help shown under the control. */
  hint?: string;
  /** Hide the control unless this returns true for the current values. */
  showIf?: (values: ParamValues) => boolean;
}

export type ParamSpec =
  | (Common & {
      kind: 'slider';
      min: number;
      max: number;
      step?: number;
      format?: (v: number) => string;
    })
  | (Common & {
      /** A slider that walks a list of values (for learning rates, penalties… that span decades). */
      kind: 'ladder';
      values: readonly number[];
      format?: (v: number) => string;
    })
  | (Common & { kind: 'select'; options: { value: string; label: string }[] })
  | (Common & { kind: 'segmented'; options: { value: string; label: string }[] })
  | (Common & { kind: 'switch' });

/** Index of the ladder value closest (on a log scale) to `value`. */
export function ladderIndex(values: readonly number[], value: number): number {
  let best = 0;
  let bestDistance = Infinity;
  values.forEach((v, i) => {
    const d = Math.abs(Math.log(Math.max(v, 1e-12)) - Math.log(Math.max(value, 1e-12)));
    if (d < bestDistance) {
      bestDistance = d;
      best = i;
    }
  });
  return best;
}

/** The numeric value of a parameter (specs guarantee it exists). */
export const num = (values: ParamValues, key: string): number => Number(values[key]);
export const str = (values: ParamValues, key: string): string => String(values[key]);
export const flag = (values: ParamValues, key: string): boolean => Boolean(values[key]);

/** The default value of every parameter, taken from a `defaults` object (which must cover all keys). */
export function withDefaults(specs: readonly ParamSpec[], defaults: ParamValues): ParamValues {
  const out: ParamValues = {};
  for (const s of specs) {
    if (!(s.key in defaults)) throw new Error(`Missing default for parameter "${s.key}"`);
    out[s.key] = defaults[s.key] as ParamValue;
  }
  return out;
}
