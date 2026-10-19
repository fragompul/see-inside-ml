import { createRng } from '@sim/core';
import {
  ATTRIBUTION_METHODS,
  CnnNetwork,
  CnnSession,
  DEFAULT_CNN_LAB,
  IMAGE_SIZE,
  argmax,
  attribute,
  deletionCurve,
  integratedGradients,
  rankPixels,
  scoreGap,
  type AttributionMethod,
} from '@sim/deep';

export const METHOD_LABEL: Record<AttributionMethod, string> = {
  saliency: 'Saliency (gradient size)',
  'gradient-input': 'Gradient × input',
  smoothgrad: 'SmoothGrad',
  integrated: 'Integrated gradients',
};

export interface AttributionScenario {
  /** Which of the example test digits is shown in detail. */
  example: number;
  method: AttributionMethod;
  /** Steps of integrated gradients, for the completeness check. */
  steps: number;
  seed: number;
}

export const DEFAULT_ATTRIBUTION: AttributionScenario = {
  example: 0,
  method: 'integrated',
  steps: 32,
  seed: 7,
};

export const EXAMPLES = 8;
export const STEPS = [1, 2, 4, 8, 32, 128];
const EVALUATED = 30;

export interface AttributionResult {
  scenario: AttributionScenario;
  accuracy: number;
  /** The example digits: pixels, true label and the network's answer. */
  images: { pixels: number[]; label: number; predicted: number }[];
  /** For the chosen example: the map of every method (signed for gradient × input and integrated gradients). */
  maps: Record<AttributionMethod, number[]>;
  /** Deletion area (average class score while pixels are removed, lower is more faithful) per method and for a random order. */
  deletion: Record<AttributionMethod, number>;
  randomDeletion: number;
  /** Completeness of integrated gradients: |sum of attributions - score gap| as a share of the gap, averaged over images, per step count. */
  completeness: number[];
  /** The sum of the attributions for the chosen example at the chosen number of steps, and its score gap. */
  sum: number;
  gap: number;
  /** Correlation between a method's map and the map after the output layer is replaced by random weights. */
  sanity: Record<AttributionMethod, number>;
}

interface Trained {
  net: CnnNetwork;
  accuracy: number;
  test: { images: Float64Array; labels: number[] };
}

const trainedCache = new Map<number, Trained>();

function trained(seed: number): Trained {
  const hit = trainedCache.get(seed);
  if (hit) return hit;
  const session = new CnnSession({ ...DEFAULT_CNN_LAB, perClass: 40, seed });
  session.step(8);
  const test: Trained['test'] = { images: new Float64Array(EVALUATED * 256), labels: [] };
  for (let i = 0; i < EVALUATED; i++) {
    const { image, label } = session.sample(i * 9);
    test.images.set(image, i * 256);
    test.labels.push(label);
  }
  const out = { net: session.network, accuracy: session.evaluate().testAccuracy, test };
  trainedCache.set(seed, out);
  return out;
}

const correlation = (a: ArrayLike<number>, b: ArrayLike<number>): number => {
  const n = a.length;
  let ma = 0;
  let mb = 0;
  for (let i = 0; i < n; i++) {
    ma += (a[i] as number) / n;
    mb += (b[i] as number) / n;
  }
  let s = 0;
  let sa = 0;
  let sb = 0;
  for (let i = 0; i < n; i++) {
    s += ((a[i] as number) - ma) * ((b[i] as number) - mb);
    sa += ((a[i] as number) - ma) ** 2;
    sb += ((b[i] as number) - mb) ** 2;
  }
  return sa === 0 || sb === 0 ? 0 : s / Math.sqrt(sa * sb);
};

interface Evaluation {
  deletion: Record<AttributionMethod, number>;
  randomDeletion: number;
  completeness: number[];
  sanity: Record<AttributionMethod, number>;
}

const evaluations = new Map<number, Evaluation>();

const zeroByMethod = () =>
  Object.fromEntries(ATTRIBUTION_METHODS.map((m) => [m, 0])) as Record<AttributionMethod, number>;

/** The slow part: faithfulness, completeness and the sanity check over 30 test images. */
function evaluate(seed: number): Evaluation {
  const hit = evaluations.get(seed);
  if (hit) return hit;
  const { net, test } = trained(seed);
  const px = IMAGE_SIZE * IMAGE_SIZE;
  const rng = createRng(5);
  // the same convolutions with an untrained output layer, for the sanity check
  const scrambled = new CnnNetwork(net.config, createRng(99));
  const keep = net.conv1.parameters.length + net.conv2.parameters.length;
  net.parameters
    .slice(0, keep)
    .forEach((p, i) => (scrambled.parameters[i] as typeof p).data.set(p.data));

  const deletion = zeroByMethod();
  const sanity = zeroByMethod();
  let randomDeletion = 0;
  const completeness = STEPS.map(() => 0);
  for (let i = 0; i < EVALUATED; i++) {
    const image = test.images.slice(i * px, (i + 1) * px);
    const cls = argmax(net.predict(image));
    for (const method of ATTRIBUTION_METHODS) {
      const { map } = attribute(method, net, image, cls);
      deletion[method] += deletionCurve(net, image, cls, rankPixels(map), 16).area / EVALUATED;
      sanity[method] += correlation(map, attribute(method, scrambled, image, cls).map) / EVALUATED;
    }
    const shuffled = Array.from({ length: px }, (_, k) => k).sort(() => rng.next() - 0.5);
    randomDeletion += deletionCurve(net, image, cls, shuffled, 16).area / EVALUATED;
    const gap = scoreGap(net, image, cls);
    STEPS.forEach((steps, k) => {
      const total = integratedGradients(net, image, cls, steps).reduce((a, b) => a + b, 0);
      completeness[k] =
        (completeness[k] as number) +
        Math.abs(total - gap) / Math.max(1e-9, Math.abs(gap)) / EVALUATED;
    });
  }
  const out = { deletion, randomDeletion, completeness, sanity };
  evaluations.set(seed, out);
  return out;
}

export function runAttribution(s: AttributionScenario): AttributionResult {
  const { net, accuracy, test } = trained(s.seed);
  const px = IMAGE_SIZE * IMAGE_SIZE;
  const evaluation = evaluate(s.seed);
  const shown = Array.from({ length: EXAMPLES }, (_, k) => {
    const pixels = test.images.slice(k * px, (k + 1) * px);
    return {
      pixels: Array.from(pixels),
      label: test.labels[k] as number,
      predicted: argmax(net.predict(pixels)),
    };
  });
  const image = test.images.slice(s.example * px, (s.example + 1) * px);
  const cls = argmax(net.predict(image));
  const maps = Object.fromEntries(
    ATTRIBUTION_METHODS.map((m) => [m, Array.from(attribute(m, net, image, cls).map)]),
  ) as Record<AttributionMethod, number[]>;
  return {
    scenario: s,
    accuracy,
    images: shown,
    maps,
    ...evaluation,
    sum: integratedGradients(net, image, cls, s.steps).reduce((a, b) => a + b, 0),
    gap: scoreGap(net, image, cls),
  };
}
