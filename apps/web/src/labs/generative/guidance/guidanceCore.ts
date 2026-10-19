import { createRng } from '@sim/core';
import {
  BLOBS,
  DEFAULT_FIELD_TRAINING,
  blobCentre,
  nearestBlob,
  sampleBlobs,
  sampleFlowMatching,
  sampleGuided,
  sampleUnguided,
  slicedWasserstein,
  trainFlowMatching,
  trainGuidedDiffusion,
} from '@sim/deep';

export type GuidanceMode = 'guidance' | 'flow';

export interface GuidanceScenario {
  mode: GuidanceMode;
  /** Classifier-free guidance weight. */
  weight: number;
  /** Class asked for when drawing the picture. */
  asked: number;
  /** Sampling steps compared in the flow-matching view. */
  steps: number;
  seed: number;
}

export const DEFAULT_GUIDANCE: GuidanceScenario = {
  mode: 'guidance',
  weight: 0,
  asked: 0,
  steps: 4,
  seed: 1,
};

export const WEIGHTS = [0, 1, 3, 8];
export const STEP_COUNTS = [1, 2, 4, 8, 16, 32];

export interface GuidanceResult {
  scenario: GuidanceScenario;
  /** Points drawn for the asked class at the chosen weight, and the reference blobs. */
  drawn: number[];
  real: number[];
  /** For each guidance weight: share of points that land nearest the class asked for, and mean distance of the points to that class centre. */
  accuracy: number[];
  spread: number[];
  /** The spread of real points around their blob centre, for comparison. */
  realSpread: number;
  /** For each count of steps: distance of the samples from the data for flow matching and for DDIM diffusion. */
  stepCounts: number[];
  flowDistance: number[];
  diffusionDistance: number[];
  /** Points from flow matching and from diffusion at the chosen number of steps. */
  flowPoints: number[];
  diffusionPoints: number[];
}

const cache = new Map<string, GuidanceResult>();
let models: {
  guided: ReturnType<typeof trainGuidedDiffusion>;
  flow: ReturnType<typeof trainFlowMatching>;
  seed: number;
} | null = null;

function modelsFor(seed: number) {
  if (!models || models.seed !== seed) {
    const data = sampleBlobs(4000, createRng(seed + 5));
    const options = { ...DEFAULT_FIELD_TRAINING, seed };
    models = {
      guided: trainGuidedDiffusion({ ...options, dropout: 0.15 }, data),
      flow: trainFlowMatching(data.points, options),
      seed,
    };
  }
  return models;
}

interface Sweeps {
  real: number[];
  accuracy: number[];
  spread: number[];
  realSpread: number;
  flowDistance: number[];
  diffusionDistance: number[];
  reference: Float64Array;
}

const sweeps = new Map<number, Sweeps>();

/** The slow part, which depends only on the random start: sweeps over guidance weights and over step counts. */
function sweepsFor(seed: number): Sweeps {
  const hit = sweeps.get(seed);
  if (hit) return hit;
  const { guided, flow } = modelsFor(seed);
  const per = 80;
  const centreDistance = (points: Float64Array, k: number) => {
    const [cx, cy] = blobCentre(k);
    let sum = 0;
    for (let i = 0; i < points.length / 2; i++)
      sum += Math.hypot((points[2 * i] as number) - cx, (points[2 * i + 1] as number) - cy);
    return sum / (points.length / 2);
  };
  const accuracy: number[] = [];
  const spread: number[] = [];
  for (const w of WEIGHTS) {
    let right = 0;
    let dist = 0;
    for (let k = 0; k < BLOBS; k++) {
      const pts = sampleGuided(guided, k, w, per, 100 + k);
      for (let i = 0; i < per; i++)
        if (nearestBlob(pts[2 * i] as number, pts[2 * i + 1] as number) === k) right++;
      dist += centreDistance(pts, k) / BLOBS;
    }
    accuracy.push(right / (per * BLOBS));
    spread.push(dist);
  }
  const reference = sampleBlobs(400, createRng(seed + 50));
  let realSpread = 0;
  for (let i = 0; i < 400; i++) {
    const [cx, cy] = blobCentre(reference.labels[i] as number);
    realSpread +=
      Math.hypot(
        (reference.points[2 * i] as number) - cx,
        (reference.points[2 * i + 1] as number) - cy,
      ) / 400;
  }
  const out: Sweeps = {
    real: Array.from(reference.points),
    accuracy,
    spread,
    realSpread,
    flowDistance: STEP_COUNTS.map((n) =>
      slicedWasserstein(sampleFlowMatching(flow, 400, 7, n), reference.points),
    ),
    diffusionDistance: STEP_COUNTS.map((n) =>
      slicedWasserstein(sampleUnguided(guided, 400, 7, n), reference.points),
    ),
    reference: reference.points,
  };
  sweeps.set(seed, out);
  return out;
}

export function runGuidance(s: GuidanceScenario): GuidanceResult {
  const key = JSON.stringify(s);
  const hit = cache.get(key);
  if (hit) return hit;
  const { guided, flow } = modelsFor(s.seed);
  const sweep = sweepsFor(s.seed);
  const out: GuidanceResult = {
    scenario: s,
    drawn: Array.from(sampleGuided(guided, s.asked, s.weight, 200, 300)),
    real: sweep.real,
    accuracy: sweep.accuracy,
    spread: sweep.spread,
    realSpread: sweep.realSpread,
    stepCounts: STEP_COUNTS,
    flowDistance: sweep.flowDistance,
    diffusionDistance: sweep.diffusionDistance,
    flowPoints: Array.from(sampleFlowMatching(flow, 400, 8, s.steps)),
    diffusionPoints: Array.from(sampleUnguided(guided, 400, 8, s.steps)),
  };
  cache.set(key, out);
  return out;
}
