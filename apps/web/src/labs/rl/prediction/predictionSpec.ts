import type { ParamSpec } from '@/shared/lab-kit';
import type { PredictionRun, PredictionScenario } from './predictionCore';

export const predictionParams: ParamSpec[] = [
  {
    key: 'states',
    label: 'States in the walk',
    kind: 'ladder',
    values: [5, 19, 51],
    format: (v) => String(v),
  },
  {
    key: 'alpha',
    label: 'Step size α',
    kind: 'ladder',
    values: [0.03, 0.05, 0.1, 0.3, 0.6],
    format: (v) => String(v),
  },
  {
    key: 'n',
    label: 'n of n-step TD',
    kind: 'ladder',
    values: [1, 2, 4, 8, 16],
    format: (v) => String(v),
  },
  {
    key: 'lambda',
    label: 'λ of TD(λ)',
    kind: 'ladder',
    values: [0, 0.4, 0.8, 0.9, 0.95],
    format: (v) => String(v),
    hint: '0 is TD(0). Close to 1 acts like Monte Carlo and needs a small step size.',
  },
  {
    key: 'episodes',
    label: 'Episodes',
    kind: 'ladder',
    values: [10, 40, 100, 200],
    format: (v) => String(v),
  },
];

export interface PredictionExperiment {
  id: string;
  title: string;
  text: string;
  scenario: Partial<PredictionScenario>;
  claim: (run: PredictionRun) => boolean;
}

export const predictionExperiments: PredictionExperiment[] = [
  {
    id: 'td-vs-mc',
    title: 'Bootstrapping learns faster',
    text: "Monte Carlo waits for the end of each walk and moves every visited state towards the final reward, a noisy target. TD(0) moves each state towards the next state's current estimate: biased at first, but far less noisy. Over the first 40 episodes it makes clearly less error.",
    scenario: {},
    claim: (r) => r.area.td < r.area.mc * 0.85 && r.area.mc > 0.25,
  },
  {
    id: 'in-between',
    title: 'The middle way is best',
    text: 'Looking a few steps ahead (n-step TD) or blending all look-aheads with traces (TD(λ)) combines the speed of TD with the lower bias of Monte Carlo: both make about half the error of TD(0) here.',
    scenario: {},
    claim: (r) => r.area.nstep < r.area.td * 0.65 && r.area.lambda < r.area.td * 0.65,
  },
  {
    id: 'lambda-zero',
    title: 'λ = 0 is TD(0)',
    text: 'With no trace decay credit goes only to the state just left: TD(λ) with λ = 0 is exactly TD(0), and the two curves are the same.',
    scenario: { lambda: 0 },
    claim: (r) => Math.abs(r.area.lambda - r.area.td) < 1e-9,
  },
  {
    id: 'small-step',
    title: 'A small step size favours Monte Carlo',
    text: 'With α = 0.03 learning is slow for everyone, and TD(0), whose early estimates are biased, is now the slowest: over 40 episodes it lags behind Monte Carlo.',
    scenario: { alpha: 0.03 },
    claim: (r) => r.area.td > r.area.mc,
  },
  {
    id: 'big-step',
    title: 'A big step size punishes Monte Carlo',
    text: "At α = 0.6 every update takes a large fraction of a noisy target. Monte Carlo's targets are the noisiest, so it bounces around at a high error; TD(0) copes best.",
    scenario: { alpha: 0.6 },
    claim: (r) =>
      r.area.mc > 2.5 * r.area.td && r.area.td < r.area.nstep && r.area.td < r.area.lambda,
  },
  {
    id: 'long-run',
    title: 'The noise floor of Monte Carlo',
    text: 'After 200 episodes with α = 0.05 TD(0) is within 5% of the true values, while Monte Carlo is stuck at about four times that error: its targets are so noisy that a constant step size never settles.',
    scenario: { alpha: 0.05, episodes: 200 },
    claim: (r) =>
      (r.curves.mc[200] as number) > 3 * (r.curves.td[200] as number) &&
      (r.curves.td[200] as number) < 0.06,
  },
  {
    id: 'short-walk',
    title: 'A short walk',
    text: 'With only five states the end of the episode is always close, so there is little for look-ahead to add: plain TD(0) is best.',
    scenario: { states: 5 },
    claim: (r) => r.area.td < r.area.nstep && r.area.td < r.area.lambda && r.area.td < r.area.mc,
  },
  {
    id: 'long-walk',
    title: 'A long walk',
    text: 'With 51 states, information from the end creeps back one state per episode under TD(0). Traces and multi-step targets pass it back much faster: TD(λ) has less than half the error.',
    scenario: { states: 51, episodes: 100 },
    claim: (r) => r.area.lambda < r.area.td / 2 && r.area.nstep < r.area.td / 1.8,
  },
];
