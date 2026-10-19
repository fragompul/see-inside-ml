import { createRng } from '@sim/core';
import {
  CnnSession,
  DEFAULT_CNN_LAB,
  DEFAULT_VISUALISE,
  argmax,
  generateDigits,
  totalVariation,
  visualise,
  type CnnNetwork,
  type VisualiseOptions,
} from '@sim/deep';

export type Regulariser = 'none' | 'decay' | 'blur';

export const REGULARISER_LABEL: Record<Regulariser, string> = {
  none: 'None (only keep pixels in range)',
  decay: 'Pull towards zero',
  blur: 'Blur a little at every step',
};

const SETTINGS: Record<Regulariser, Partial<VisualiseOptions>> = {
  none: {},
  decay: { decay: 0.04 },
  blur: { blur: 0.5 },
};

export interface VisScenario {
  /** What is visualised: a digit class, or one of the feature maps of the last convolution. */
  kind: 'class' | 'channel';
  regulariser: Regulariser;
  steps: number;
  seed: number;
}

export const DEFAULT_VIS: VisScenario = { kind: 'class', regulariser: 'blur', steps: 120, seed: 7 };

export const CLASSES = 10;
export const CHANNELS = 12;

export interface VisResult {
  scenario: VisScenario;
  /** One picture per class (or per feature map) and its score at the end. */
  images: number[][];
  scores: number[];
  /** Total variation of each picture (large = noisy). */
  noise: number[];
  /** For classes: the class the network assigns to each picture, and its probability. */
  predicted: number[];
  confidence: number[];
  /** Correlation of each class picture with the average real digit of that class (classes only). */
  likeness: number[];
  /** Mean over all pictures. */
  meanScore: number;
  meanNoise: number;
  meanLikeness: number;
  /** Share of class pictures that the network assigns to the class they were made for. */
  hitRate: number;
  /** For comparison: the same measures for real digits. */
  realNoise: number;
  realScore: number;
}

interface Trained {
  net: CnnNetwork;
  classMean: number[][];
  realNoise: number;
  realScore: number;
}

const trainedCache = new Map<number, Trained>();

function trained(seed: number): Trained {
  const hit = trainedCache.get(seed);
  if (hit) return hit;
  const session = new CnnSession({ ...DEFAULT_CNN_LAB, perClass: 40, seed });
  session.step(8);
  const real = generateDigits(30, createRng(seed + 100));
  const classMean = Array.from({ length: CLASSES }, () => new Array<number>(256).fill(0));
  const count = new Array<number>(CLASSES).fill(0);
  let realNoise = 0;
  let realScore = 0;
  for (let i = 0; i < real.count; i++) {
    const c = real.labels[i] as number;
    const image = real.images.slice(i * 256, (i + 1) * 256);
    count[c] = (count[c] as number) + 1;
    (classMean[c] as number[]).forEach(
      (_, p) =>
        ((classMean[c] as number[])[p] =
          ((classMean[c] as number[])[p] as number) + (image[p] as number)),
    );
    realNoise += totalVariation(image) / real.count;
    realScore += session.network.logitGradient(image, c).logit / real.count;
  }
  classMean.forEach((m, c) =>
    m.forEach((_, p) => (m[p] = (m[p] as number) / (count[c] as number))),
  );
  const out = { net: session.network, classMean, realNoise, realScore };
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

const cache = new Map<string, VisResult>();

export function runVis(s: VisScenario): VisResult {
  const key = JSON.stringify(s);
  const hit = cache.get(key);
  if (hit) return hit;
  const { net, classMean, realNoise, realScore } = trained(s.seed);
  const count = s.kind === 'class' ? CLASSES : CHANNELS;
  const options: VisualiseOptions = {
    ...DEFAULT_VISUALISE,
    ...SETTINGS[s.regulariser],
    steps: s.steps,
    seed: 1,
  };
  const images: number[][] = [];
  const scores: number[] = [];
  const noise: number[] = [];
  const predicted: number[] = [];
  const confidence: number[] = [];
  const likeness: number[] = [];
  for (let k = 0; k < count; k++) {
    const v = visualise(net, { kind: s.kind, index: k }, options);
    images.push(Array.from(v.image));
    scores.push(v.endScore);
    noise.push(totalVariation(v.image));
    if (s.kind === 'class') {
      const probs = net.predict(v.image);
      predicted.push(argmax(probs));
      confidence.push(probs[argmax(probs)] as number);
      likeness.push(correlation(v.image, classMean[k] as number[]));
    }
  }
  const mean = (v: number[]) => (v.length ? v.reduce((a, b) => a + b, 0) / v.length : 0);
  const out: VisResult = {
    scenario: s,
    images,
    scores,
    noise,
    predicted,
    confidence,
    likeness,
    meanScore: mean(scores),
    meanNoise: mean(noise),
    meanLikeness: mean(likeness),
    hitRate: s.kind === 'class' ? predicted.filter((p, k) => p === k).length / count : 0,
    realNoise,
    realScore,
  };
  cache.set(key, out);
  return out;
}
