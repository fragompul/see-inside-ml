import type { ParamSpec } from '@/shared/lab-kit';
import {
  WORLDS,
  complexityCurve,
  type BiasVarianceRun,
  type BiasVarianceScenario,
  type WorldId,
} from './biasVarianceCore';

export const bvParams: ParamSpec[] = [
  {
    key: 'world',
    label: 'The truth',
    kind: 'segmented',
    options: (Object.keys(WORLDS) as WorldId[]).map((id) => ({
      value: id,
      label: WORLDS[id].label,
    })),
  },
  {
    key: 'features',
    label: 'Model flexibility (number of polynomial terms)',
    kind: 'slider',
    min: 1,
    max: 14,
    step: 1,
    hint: '1 = a flat line, 2 = a straight line, 4 = a cubic, and so on.',
  },
  {
    key: 'n',
    label: 'Training points per dataset',
    kind: 'ladder',
    values: [10, 15, 20, 30, 60, 100, 200],
    format: (v) => String(v),
  },
  {
    key: 'noise',
    label: 'Noise in the data',
    kind: 'slider',
    min: 0,
    max: 1,
    step: 0.05,
    format: (v) => v.toFixed(2),
  },
  {
    key: 'ridge',
    label: 'Regularisation (penalty on large coefficients)',
    kind: 'ladder',
    values: [0, 0.001, 0.01, 0.1, 0.3, 1],
    format: (v) => (v === 0 ? 'none' : String(v)),
  },
];

export interface BvExperiment {
  id: string;
  title: string;
  text: string;
  scenario: Partial<BiasVarianceScenario>;
  claim: (run: BiasVarianceRun) => boolean;
}

export const bvExperiments: BvExperiment[] = [
  {
    id: 'underfit',
    title: 'Too simple: bias',
    text: 'A flat line cannot follow a wave. Every fit it makes is almost the same (low variance) and every one is wrong in the same way (large bias): the error is almost all bias.',
    scenario: { features: 1 },
    claim: (r) => r.bias2 > 10 * r.variance && r.testError > 0.5,
  },
  {
    id: 'just-right',
    title: 'About right',
    text: 'A cubic (4 terms) matches the wave closely: bias nearly vanishes, variance is small, and the test error is close to the noise floor, the best any model could do.',
    scenario: { features: 4 },
    claim: (r) => r.bias2 < 0.01 && r.testError < r.noiseVariance * 1.4,
  },
  {
    id: 'overfit',
    title: 'Too flexible: variance',
    text: 'Twelve terms from only 20 points: every dataset gives a wildly different curve. The training error is tiny, yet the test error is several times the noise: the model has fitted the noise.',
    scenario: { features: 12, n: 20 },
    claim: (r) =>
      r.variance > 50 * r.bias2 && r.trainError < 0.05 && r.testError > 5 * r.noiseVariance,
  },
  {
    id: 'regularise',
    title: 'Regularisation tames variance',
    text: 'The same over-flexible model with a penalty on large coefficients: a little bias is traded for a large drop in variance, and the test error falls to a fraction of what it was.',
    scenario: { features: 12, n: 20, ridge: 0.3 },
    claim: (r) => r.variance < 0.1 && r.testError < 0.2,
  },
  {
    id: 'more-data',
    title: 'More data cures variance',
    text: 'With 200 points the same twelve-term model has enough evidence to pin its coefficients down: variance collapses and the test error approaches the noise floor.',
    scenario: { features: 12, n: 200 },
    claim: (r) => r.variance < 0.01 && r.testError < r.noiseVariance * 1.15,
  },
  {
    id: 'noise-floor',
    title: 'The noise floor',
    text: 'With noise 0.8 no model can get below its variance (0.64). Even the right model has a test error of about that, because the next observation will be noisy as well.',
    scenario: { noise: 0.8 },
    claim: (r) => r.testError > r.noiseVariance && r.bias2 < 0.02,
  },
  {
    id: 'simple-truth',
    title: 'When the truth is a line',
    text: 'If the world really is a straight line, the two-term model is perfect and extra flexibility is pure cost: nine terms add variance without removing any bias.',
    scenario: { world: 'line', features: 9 },
    claim: (r) => r.bias2 < 0.01 && r.variance > 0.03,
  },
  {
    id: 'steps',
    title: 'Sharp steps are hard for smooth curves',
    text: 'A staircase has sudden jumps that low-degree polynomials cannot make: the bias stays high until the model is very flexible and the data plentiful.',
    scenario: { world: 'steps', features: 4 },
    claim: (r) => r.bias2 > 0.15,
  },
  {
    id: 'double-descent',
    title: 'Double descent',
    text: 'Past the point where the model can pass through every training point (as many terms as points), the test error is astronomical, but it comes down again as flexibility keeps growing: among the many curves that fit the data, the smoothest one is chosen.',
    scenario: { n: 15, noise: 0.3 },
    claim: (r) => {
      const c = complexityCurve({ ...r.scenario }, 40);
      const at = (p: number) => c[p - 1]!.test;
      return at(15) > 1000 * at(4) && at(40) < at(10) && at(40) < 0.5;
    },
  },
];
