import type { ParamSpec } from '@/shared/lab-kit';
import type { TrackingRun, TrackingScenario } from './trackingCore';

export const trackingParams: ParamSpec[] = [
  {
    key: 'motion',
    label: 'How the target moves',
    kind: 'segmented',
    options: [
      { value: 'cruise', label: 'Cruising' },
      { value: 'turn', label: 'Steady turn' },
    ],
  },
  {
    key: 'sensorNoise',
    label: 'Sensor noise',
    kind: 'slider',
    min: 0.5,
    max: 10,
    step: 0.5,
    format: (v) => v.toFixed(1),
  },
  {
    key: 'truePush',
    label: 'Random pushes on the target',
    kind: 'slider',
    min: 0,
    max: 0.6,
    step: 0.05,
    format: (v) => v.toFixed(2),
    hint: 'How much the target really accelerates at random.',
  },
  {
    key: 'dropout',
    label: 'Readings lost',
    kind: 'slider',
    min: 0,
    max: 0.9,
    step: 0.05,
    format: (v) => `${Math.round(v * 100)}%`,
  },
  {
    key: 'outliers',
    label: 'Wild readings',
    kind: 'slider',
    min: 0,
    max: 0.3,
    step: 0.01,
    format: (v) => `${Math.round(v * 100)}%`,
    hint: 'Readings with eight times the normal noise.',
  },
  {
    key: 'assumedPush',
    label: 'Pushes the filters expect',
    kind: 'ladder',
    values: [0.01, 0.05, 0.15, 0.4, 0.8, 1.5],
    format: (v) => String(v),
    hint: 'The filters’ trust in a straight path: small = very confident, large = ready for surprises. (The particle filter never goes below 0.3: it needs some jitter.)',
  },
  {
    key: 'particleLikelihood',
    label: 'Particle filter assumes noise is',
    kind: 'segmented',
    options: [
      { value: 'gaussian', label: 'Gaussian' },
      { value: 'robust', label: 'Heavy-tailed' },
    ],
  },
  {
    key: 'particles',
    label: 'Particles',
    kind: 'ladder',
    values: [20, 100, 500, 1500, 3000],
    format: (v) => String(v),
  },
];

export interface TrackingExperiment {
  id: string;
  title: string;
  text: string;
  scenario: Partial<TrackingScenario>;
  claim: (run: TrackingRun) => boolean;
}

export const trackingExperiments: TrackingExperiment[] = [
  {
    id: 'denoise',
    title: 'Filtering beats the sensor',
    text: 'Every reading is jittery, but the target obeys physics: a position now tells you roughly where it is next. Blending prediction and reading cuts the error of the raw sensor by about 40%.',
    scenario: {},
    claim: (r) => r.rmse.kalman < r.rmse.raw * 0.65,
  },
  {
    id: 'smoother',
    title: 'Hindsight helps',
    text: 'The smoother also uses the readings that came after each moment, so it halves the error of the live filter, at the price of not being available in real time.',
    scenario: {},
    claim: (r) => r.rmse.smoother < r.rmse.kalman * 0.6,
  },
  {
    id: 'agree',
    title: 'Particles agree with the Kalman filter',
    text: 'On straight-line motion with Gaussian noise the Kalman filter is exact. A cloud of 1500 weighted particles reaches the same accuracy without any of its formulas.',
    scenario: {},
    claim: (r) => Math.abs(r.rmse.particle - r.rmse.kalman) < 0.15,
  },
  {
    id: 'stubborn',
    title: 'A filter that is too sure',
    text: 'The target is turning but the filter is almost certain that it flies straight (tiny expected pushes). It trusts its prediction over the readings and drifts away: far worse than the raw sensor.',
    scenario: { motion: 'turn', assumedPush: 0.01 },
    claim: (r) => r.rmse.kalman > r.rmse.raw * 3,
  },
  {
    id: 'open-minded',
    title: 'Allowing for surprises',
    text: 'Telling the same filter to expect bigger pushes makes it lean on the readings more. It follows the turn and beats the raw sensor again.',
    scenario: { motion: 'turn', assumedPush: 0.8 },
    claim: (r) => r.rmse.kalman < r.rmse.raw && r.rmse.kalman < 3.5,
  },
  {
    id: 'dropouts',
    title: 'Half the readings missing',
    text: 'With no reading the filter just predicts, and its uncertainty grows until the next one arrives. Half the data is gone, yet the estimate stays far better than the raw readings that remain.',
    scenario: { dropout: 0.5 },
    claim: (r) => r.rmse.kalman < r.rmse.raw && r.rmse.smoother < r.rmse.raw * 0.5,
  },
  {
    id: 'wild',
    title: 'Wild readings',
    text: 'A Gaussian model treats a reading 25 units off as overwhelming evidence and swings towards it. A particle filter that knows readings can be wild (heavy-tailed likelihood) simply discounts them.',
    scenario: { outliers: 0.1, particleLikelihood: 'robust' },
    claim: (r) => r.rmse.kalman > 5 && r.rmse.particle < r.rmse.kalman / 2,
  },
  {
    id: 'gullible',
    title: 'A particle filter that trusts too much',
    text: 'The same wild readings destroy a particle filter whose likelihood is Gaussian: one reading gives almost every particle zero weight, the cloud collapses onto a handful, and the track is lost.',
    scenario: { outliers: 0.1, particleLikelihood: 'gaussian' },
    claim: (r) => r.rmse.particle > r.rmse.raw * 1.5,
  },
  {
    id: 'few-particles',
    title: 'Too few particles',
    text: 'With 20 particles the cloud cannot cover the plausible states; after a few updates only one or two survive and the estimate jumps around.',
    scenario: { particles: 20 },
    claim: (r) => r.rmse.particle > r.rmse.kalman * 2,
  },
];
