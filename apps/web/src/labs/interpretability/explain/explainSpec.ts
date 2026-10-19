import type { ParamSpec } from '@/shared/lab-kit';
import {
  FEATURES,
  MODEL_LABELS,
  limeOf,
  topK,
  type ExplainModel,
  type ExplainRun,
  type ExplainScenario,
} from './explainCore';

export const explainParams: ParamSpec[] = [
  {
    key: 'model',
    label: 'Model to explain',
    kind: 'segmented',
    options: (Object.keys(MODEL_LABELS) as ExplainModel[]).map((m) => ({
      value: m,
      label: MODEL_LABELS[m],
    })),
  },
  {
    key: 'instance',
    label: 'Patient to explain',
    kind: 'slider',
    min: 0,
    max: 141,
    step: 1,
    hint: 'One of 142 patients the model has never seen.',
  },
  {
    key: 'feature',
    label: 'Feature for the dependence plot',
    kind: 'select',
    options: FEATURES.map((f, i) => ({ value: String(i), label: f })),
  },
  {
    key: 'limeSamples',
    label: 'LIME samples',
    kind: 'ladder',
    values: [30, 120, 500, 2000],
    format: (v) => String(v),
    hint: 'How many perturbed copies of the patient LIME asks the model about.',
  },
];

export interface ExplainExperiment {
  id: string;
  title: string;
  text: string;
  scenario: Partial<ExplainScenario>;
  claim: (run: ExplainRun) => boolean;
}

const idx = (name: string) => FEATURES.indexOf(name);

export const explainExperiments: ExplainExperiment[] = [
  {
    id: 'adds-up',
    title: 'The contributions add up',
    text: 'Shapley values split the gap between this patient\'s prediction and the average prediction exactly among the ten measurements: the bars below sum to that gap, no more and no less. That "efficiency" property is what makes them an explanation rather than just a score.',
    scenario: { model: 'forest' },
    claim: (r) =>
      Math.abs(r.shap.values.reduce((a, b) => a + b, 0) - (r.shap.prediction - r.shap.base)) <
        1e-9 && Math.abs(r.shap.base - 153) < 5,
  },
  {
    id: 'linear',
    title: 'For a linear model it is just arithmetic',
    text: 'In a linear model each feature adds its weight times how far the patient is from the average, and the Shapley value is exactly that. The method is a generalisation of reading off the coefficients to models where nothing can be read off.',
    scenario: { model: 'linear' },
    claim: (r) =>
      !!r.linearContributions &&
      r.shap.values.every((v, j) => Math.abs(v - r.linearContributions![j]!) < 1e-6),
  },
  {
    id: 'correlated',
    title: 'Correlated measurements cancel in a linear model',
    text: 'Serum measures s1 and s2 are almost the same (they are both cholesterol readings). The linear model gives them huge opposite weights, so for this patient s1 pulls the prediction down by 33 and s2 pushes it up by 12: more than bmi, but only because they are fighting each other. Reading coefficients as importance would mislead.',
    scenario: { model: 'linear' },
    claim: (r) => {
      const s1 = r.shap.values[idx('s1')]!;
      const s2 = r.shap.values[idx('s2')]!;
      return s1 * s2 < 0 && Math.abs(s1) > Math.abs(r.shap.values[idx('bmi')]!);
    },
  },
  {
    id: 'forest-ignores',
    title: 'The forest shrugs them off',
    text: 'The random forest can use either cholesterol reading and does not need them: it relies on bmi and s5 (a blood-serum measure), and the contributions of s1 and s2 together are a small fraction of the biggest ones.',
    scenario: { model: 'forest' },
    claim: (r) =>
      Math.abs(r.shap.values[idx('s1')]!) + Math.abs(r.shap.values[idx('s2')]!) <
        Math.abs(r.shap.values[idx('bmi')]!) / 3 &&
      topK(r.shap.values, 2).every((j) => [idx('bmi'), idx('s5')].includes(j)),
  },
  {
    id: 'global',
    title: 'Two global views agree',
    text: 'Averaging the size of the Shapley values over many patients, and measuring how much the score drops when a column is shuffled (permutation importance), give the same two leaders for the forest: s5 and bmi.',
    scenario: { model: 'forest' },
    claim: (r) => {
      const a = topK(r.meanAbsShap, 2).sort();
      const b = topK(r.permutation, 2).sort();
      return a[0] === b[0] && a[1] === b[1];
    },
  },
  {
    id: 'kernel',
    title: 'KernelSHAP with only 40 samples',
    text: 'Computing exact Shapley values takes 2^10 evaluations per background row. KernelSHAP samples 40 coalitions and solves a weighted regression: its contributions are within a few units of the exact ones and still add up exactly.',
    scenario: { model: 'forest' },
    claim: (r) => {
      const err = Math.max(...r.shap.values.map((v, j) => Math.abs(v - r.kernel.values[j]!)));
      const sum = r.kernel.values.reduce((a, b) => a + b, 0);
      return (
        err < 0.15 * Math.max(...r.shap.values.map(Math.abs)) &&
        Math.abs(sum - (r.kernel.prediction - r.kernel.base)) < 1e-9
      );
    },
  },
  {
    id: 'lime-noisy',
    title: 'LIME is noisy with few samples',
    text: 'LIME asks the model about random perturbations of the patient. With 30 of them, six different random draws give three or more different "top three" explanations; with 2,000 the explanation does not change.',
    scenario: { model: 'forest', limeSamples: 30 },
    claim: (r) => {
      const sets = (samples: number) =>
        new Set(
          [1, 2, 3, 4, 5, 6].map((seed) =>
            topK(limeOf(r.scenario.model, r.scenario.instance, samples, seed).contributions, 3)
              .sort((a, b) => a - b)
              .join(),
          ),
        );
      return sets(30).size >= 3 && sets(2000).size === 1;
    },
  },
  {
    id: 'lime-local',
    title: 'LIME is local',
    text: 'LIME fits a straight line to the forest around this patient, where a forest is a staircase. The line catches only the slope of the step the patient happens to be standing on: its estimate of the effect of bmi is a fraction of the Shapley value, which accounts for the whole path from the average patient to this one.',
    scenario: { model: 'forest' },
    claim: (r) =>
      Math.abs(r.lime.contributions[idx('bmi')]!) < Math.abs(r.shap.values[idx('bmi')]!) / 2,
  },
  {
    id: 'dependence',
    title: 'What the forest learned about bmi',
    text: "Setting every patient's bmi to each value in turn and averaging the predictions (the partial-dependence curve) shows the forest's model of the effect: predicted progression rises by about 70 points across the range of bmi, in a staircase.",
    scenario: { model: 'forest', feature: idx('bmi') },
    claim: (r) => r.ice.average[r.ice.average.length - 1]! - r.ice.average[0]! > 40,
  },
];
