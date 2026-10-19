import { CONSTRAINED_PROBLEMS } from '@sim/core';
import type { ParamSpec } from '@/shared/lab-kit';
import { type ConstrainedResult, type ConstrainedScenario } from './constrainedCore';

export const METHOD_LABEL = {
  penalty: 'Quadratic penalty',
  augmented: 'Augmented Lagrangian',
  barrier: 'Logarithmic barrier',
} as const;

export const constrainedParams: ParamSpec[] = [
  {
    key: 'problem',
    label: 'Problem',
    kind: 'select',
    options: Object.values(CONSTRAINED_PROBLEMS).map((p) => ({ value: p.id, label: p.label })),
  },
  {
    key: 'method',
    label: 'Method',
    kind: 'select',
    options: (Object.keys(METHOD_LABEL) as (keyof typeof METHOD_LABEL)[]).map((id) => ({
      value: id,
      label: METHOD_LABEL[id],
    })),
  },
  { key: 'outer', label: 'Stages', kind: 'ladder', values: [2, 4, 6, 8], format: (v) => String(v) },
  {
    key: 'weight',
    label: 'First weight',
    kind: 'ladder',
    values: [1, 10, 100],
    format: (v) => String(v),
    hint: 'Penalty weight (penalty, augmented) or barrier weight.',
  },
  {
    key: 'factor',
    label: 'Weight change per stage',
    kind: 'ladder',
    values: [1, 2, 10],
    format: (v) => (v === 1 ? 'none' : `×${v}`),
    hint: 'Growth of the penalty or shrink of the barrier; the augmented method keeps its weight.',
  },
];

export interface ConstrainedExperiment {
  id: string;
  title: string;
  text: string;
  scenario: Partial<ConstrainedScenario>;
  claim: (run: ConstrainedResult) => boolean;
}

const last = <T>(v: T[]): T => v[v.length - 1] as T;

export const constrainedExperiments: ConstrainedExperiment[] = [
  {
    id: 'penalty',
    title: 'The penalty is only right in the limit',
    text: 'Each stage multiplies the weight by ten. The solution of every stage is outside the disc, by an amount that falls by ten times with it: the error is 0.31, 0.055, 0.006, 0.0006 and so on. To be within 1e-6 the weight has to reach a million.',
    scenario: { problem: 'disk', method: 'penalty', outer: 7, weight: 1, factor: 10 },
    claim: (r) => {
      const e = r.error;
      return (
        e.every((v, i) => i === 0 || (e[i - 1] as number) / v > 5) &&
        r.steps.every((s) => s.violation > 1e-9) &&
        (e[3] as number) < 1e-3 &&
        (last(e) as number) < 1e-6
      );
    },
  },
  {
    id: 'augmented',
    title: 'Multipliers make a modest weight enough',
    text: 'The augmented Lagrangian keeps the weight at 10 and learns the multiplier instead. The error still falls by ten times each stage, to under 1e-6 after six, without the weight ever growing, and the multiplier settles on the value the optimality conditions demand, 1.236.',
    scenario: { problem: 'disk', method: 'augmented', outer: 6, weight: 10, factor: 1 },
    claim: (r) => {
      const l = last(r.steps);
      return (
        r.error[5] !== undefined &&
        (r.error[5] as number) < 1e-6 &&
        Math.abs((l.multipliers[0] as number) - (r.solution.multipliers[0] as number)) < 1e-4 &&
        r.steps.every((s) => s.weight === 10)
      );
    },
  },
  {
    id: 'barrier',
    title: 'The barrier approaches from inside',
    text: 'The logarithmic barrier makes the boundary impossible to cross, so every stage is strictly inside the disc and the distance to the rim shrinks with the weight. It is the only one of the three that never asks for an infeasible point to be evaluated.',
    scenario: { problem: 'disk', method: 'barrier', outer: 7, weight: 1, factor: 10 },
    claim: (r) =>
      r.steps.every((s) => s.violation < 0) &&
      r.steps.every(
        (s, i) => i === 0 || -s.violation < -(r.steps[i - 1] as { violation: number }).violation,
      ) &&
      (last(r.error) as number) < 1e-4,
  },
  {
    id: 'kkt',
    title: 'Only the nearest side is active',
    text: 'In the triangle the target lies beyond one slanted side, so only that constraint matters: its multiplier converges to 0.8 and the other four to zero, which is what complementary slackness says about constraints that are not touched. The optimality conditions hold to better than 1e-4.',
    scenario: { problem: 'polygon', method: 'augmented', outer: 8, weight: 10, factor: 1 },
    claim: (r) => {
      const m = last(r.steps).multipliers;
      return (
        Math.abs((m[0] as number) - 0.8) < 1e-3 &&
        m.slice(1).every((v) => Math.abs(v) < 1e-4) &&
        r.kkt < 1e-4
      );
    },
  },
  {
    id: 'equality',
    title: 'An equality constraint has a multiplier of either sign',
    text: 'On the line x + y = 1 the solution is (½, ½). The multiplier is −1: its sign says on which side of the line the objective would have preferred to go. An inequality multiplier can only be positive, because relaxing the constraint can never hurt.',
    scenario: { problem: 'line', method: 'augmented', outer: 8, weight: 10, factor: 1 },
    claim: (r) =>
      Math.abs((last(r.steps).multipliers[0] as number) + 1) < 1e-4 &&
      (last(r.error) as number) < 1e-6,
  },
  {
    id: 'banana',
    title: 'A curved valley, held on the rim',
    text: 'Rosenbrock’s valley runs out of the unit disc, so the constrained minimum is on the rim at about (0.786, 0.618), and the objective there is still 0.046 against 0 outside. The augmented Lagrangian finds it with the weight fixed at 20.',
    scenario: { problem: 'rosenbrock', method: 'augmented', outer: 25, weight: 20, factor: 1 },
    claim: (r) => {
      const l = last(r.steps);
      return (
        Math.abs(Math.hypot(l.x[0], l.x[1]) - 1) < 1e-4 && l.f > 0.02 && l.f < 0.08 && r.kkt < 1e-4
      );
    },
  },
];
