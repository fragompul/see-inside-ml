import type { ParamSpec } from '@/shared/lab-kit';
import { MODEL_LABELS, type MetricsRun, type MetricsScenario, type ModelId } from './metricsCore';

export const metricsParams: ParamSpec[] = [
  {
    key: 'positiveShare',
    label: 'Share of positives (how rare the event is)',
    kind: 'ladder',
    values: [0.01, 0.02, 0.05, 0.1, 0.3, 0.5],
    format: (v) => `${Math.round(v * 100)}%`,
  },
  {
    key: 'separation',
    label: 'How different positives and negatives are',
    kind: 'slider',
    min: 0.5,
    max: 4,
    step: 0.5,
    format: (v) => v.toFixed(1),
    hint: 'Distance between the two groups in standard deviations.',
  },
  {
    key: 'model',
    label: 'Model',
    kind: 'segmented',
    options: (Object.keys(MODEL_LABELS) as ModelId[]).map((m) => ({
      value: m,
      label: MODEL_LABELS[m],
    })),
  },
  {
    key: 'rebalance',
    label: 'Rebalance the training data',
    kind: 'segmented',
    options: [
      { value: 'none', label: 'No' },
      { value: 'oversample', label: 'Duplicate positives' },
      { value: 'smote', label: 'SMOTE' },
    ],
  },
  {
    key: 'calibrate',
    label: 'Recalibrate probabilities (Platt scaling)',
    kind: 'switch',
  },
  {
    key: 'threshold',
    label: 'Decision threshold',
    kind: 'slider',
    min: 0.01,
    max: 0.99,
    step: 0.01,
    format: (v) => v.toFixed(2),
    hint: 'Call a case positive when its probability is at least this.',
  },
  {
    key: 'costRatio',
    label: 'A missed positive costs this many false alarms',
    kind: 'ladder',
    values: [1, 2, 5, 10, 20, 50],
    format: (v) => `${v}×`,
  },
];

export interface MetricsExperiment {
  id: string;
  title: string;
  text: string;
  scenario: Partial<MetricsScenario>;
  claim: (run: MetricsRun) => boolean;
}

export const metricsExperiments: MetricsExperiment[] = [
  {
    id: 'accuracy-paradox',
    title: 'The accuracy paradox',
    text: 'Only 2% of the cases are positive. The model is 98% accurate, but so is a rule that always says “no”, and the model finds fewer than a quarter of the real positives. Accuracy hides the failure.',
    scenario: { positiveShare: 0.02 },
    claim: (r) => r.at.accuracy - r.baselineAccuracy < 0.01 && r.at.recall < 0.25 && r.rocAuc > 0.9,
  },
  {
    id: 'lower-threshold',
    title: 'Lowering the threshold',
    text: 'Calling a case positive from a probability of 0.10 instead of 0.50 catches many more of the real positives, at the price of mostly false alarms. Accuracy even falls below the do-nothing rule, yet this model may be far more useful.',
    scenario: { positiveShare: 0.02, threshold: 0.1 },
    claim: (r) => r.at.recall > 0.5 && r.at.precision < 0.4 && r.at.accuracy < r.baselineAccuracy,
  },
  {
    id: 'strict-threshold',
    title: 'Raising the threshold',
    text: 'At a threshold of 0.8 nearly everything the model flags is right (precision above 90%), but it misses most positives. Precision and recall pull in opposite directions; the threshold chooses the compromise.',
    scenario: { threshold: 0.8 },
    claim: (r) => r.at.precision > 0.9 && r.at.recall < 0.3,
  },
  {
    id: 'roc-vs-pr',
    title: 'ROC looks rosy when positives are rare',
    text: 'The ROC curve (area above 0.9) says the model ranks positives well, but the precision-recall curve (average precision under 0.55) shows that among the cases it flags, many are still false alarms: negatives are so numerous that a small false-positive rate is still many cases.',
    scenario: { positiveShare: 0.02 },
    claim: (r) => r.rocAuc > 0.9 && r.averagePrecision < 0.55,
  },
  {
    id: 'balanced',
    title: 'With balanced classes they agree',
    text: 'When half of the cases are positive the two curves tell the same story: area under ROC and average precision are both about 0.93.',
    scenario: { positiveShare: 0.5 },
    claim: (r) => Math.abs(r.rocAuc - r.averagePrecision) < 0.02 && r.rocAuc > 0.9,
  },
  {
    id: 'rebalance',
    title: 'Rebalancing buys recall and breaks calibration',
    text: 'Duplicating the rare positives before training makes the model find about three quarters of them at a threshold of 0.5, but its probabilities now think positives are as common as negatives: they are far from the truth (calibration error above 0.15).',
    scenario: { positiveShare: 0.02, rebalance: 'oversample' },
    claim: (r) => r.at.recall > 0.7 && r.ece > 0.15,
  },
  {
    id: 'recalibrate',
    title: 'Platt scaling repairs the probabilities',
    text: 'A second small model, fitted on data the first one never saw, maps the distorted scores back to honest probabilities. Calibration error falls from above 0.2 to under 0.02.',
    scenario: { positiveShare: 0.02, rebalance: 'oversample', calibrate: true },
    claim: (r) => r.ece < 0.02,
  },
  {
    id: 'costly-misses',
    title: 'When a miss is 20 times worse',
    text: 'If a missed positive costs twenty false alarms, the best threshold collapses towards zero and the total cost drops to a third of what the default 0.5 gives.',
    scenario: { costRatio: 20 },
    claim: (r) => r.best.threshold < 0.1 && r.best.cost < r.cost / 2,
  },
  {
    id: 'overconfident-tree',
    title: 'A tree is over-confident',
    text: 'A deep decision tree outputs probabilities of exactly 0 or 1 in many leaves, so its calibration error is several times that of logistic regression and its ranking (AUC) is worse.',
    scenario: { model: 'tree' },
    claim: (r) => r.ece > 0.06 && r.rocAuc < 0.8,
  },
  {
    id: 'overlap',
    title: 'Overlapping groups',
    text: 'When positives look almost like negatives (separation 1) the model rarely dares to call anything positive at 0.5: recall is a few percent and the balanced MCC score is close to zero.',
    scenario: { separation: 1 },
    claim: (r) => r.at.recall < 0.1 && r.at.mcc < 0.2,
  },
];
