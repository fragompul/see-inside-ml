import {
  DEFAULT_DIFFUSION_LAB,
  DiffusionSession,
  makeSchedule,
  slicedWasserstein,
  type DiffusionLabConfig,
} from '@sim/deep';

/**
 * One-click experiments. `claim.check` verifies, on a session trained for `claim.iterations`
 * iterations with the lesson's configuration, the statement the text makes; `lessons.test.ts`
 * runs them all.
 */
export interface Lesson {
  id: string;
  title: string;
  config: Partial<DiffusionLabConfig>;
  scrollTo: string;
  text: string;
  claim: { iterations: number; check: (session: DiffusionSession) => boolean };
}

const ITERATIONS = 1000;

/** Distance of fixed-noise samples to the data. */
export function sampleDistance(
  s: DiffusionSession,
  method: 'ddpm' | 'ddim',
  steps: number,
): number {
  return slicedWasserstein(s.fixedSample({ method, steps }).samples, s.reference);
}

/** A sibling session with the same settings but some changes, trained for `iterations`. */
function sibling(
  s: DiffusionSession,
  patch: Partial<DiffusionLabConfig>,
  iterations = ITERATIONS,
): DiffusionSession {
  const other = new DiffusionSession({ ...DEFAULT_DIFFUSION_LAB, ...s.config, ...patch });
  other.train(iterations);
  return other;
}

export const LESSONS: readonly Lesson[] = [
  {
    id: 'one-step',
    title: 'One step is not enough',
    config: { dataset: 'gaussians8', sampler: 'ddim', sampleSteps: 1 },
    scrollTo: 'diff-sample',
    text: 'With a single denoising step the model can only jump to the average of everything it might have been: a blur in the middle. Raise the number of denoising steps in section 03 and the eight clusters appear.',
    claim: {
      iterations: ITERATIONS,
      check: (s) => sampleDistance(s, 'ddim', 1) > 8 * sampleDistance(s, 'ddim', 20),
    },
  },
  {
    id: 'few-steps',
    title: 'Skipping steps with DDIM',
    config: { sampler: 'ddim', sampleSteps: 10 },
    scrollTo: 'diff-sample',
    text: 'DDPM needs every one of the 50 steps. DDIM, which is deterministic, takes only 10 of them and gets nearly as good a result: five times fewer network evaluations for the same picture.',
    claim: {
      iterations: ITERATIONS,
      check: (s) => sampleDistance(s, 'ddim', 10) < 1.8 * sampleDistance(s, 'ddim', 50),
    },
  },
  {
    id: 'early',
    title: 'Early in training',
    config: { sampler: 'ddim', sampleSteps: 20 },
    scrollTo: 'diff-sample',
    text: 'After only a hundred iterations the network has barely learned anything, so the samples are a smeared cloud. Compare with the same model after a thousand: the clusters are crisp. Press Reset in section 04 and generate again to see the early version.',
    claim: {
      iterations: ITERATIONS,
      check: (s) =>
        sampleDistance(sibling(s, {}, 100), 'ddim', 20) > 2 * sampleDistance(s, 'ddim', 20),
    },
  },
  {
    id: 'ring',
    title: 'Learning a curve',
    config: { dataset: 'ring', sampler: 'ddpm' },
    scrollTo: 'diff-sample',
    text: 'A thin ring is a hard target for a simple method, but the denoiser only ever has to learn small corrections at each noise level. Watch the points slide onto the circle during generation.',
    claim: { iterations: ITERATIONS, check: (s) => sampleDistance(s, 'ddpm', 0) < 0.3 },
  },
  {
    id: 'cosine',
    title: 'A gentler noise schedule',
    config: { schedule: 'cosine' },
    scrollTo: 'diff-forward',
    text: 'The cosine schedule destroys the data more slowly: at the halfway point it still keeps a lot of signal, while the linear one has already lost almost all of it. Compare the curves in section 01 (switch the schedule in section 04).',
    claim: {
      iterations: 0,
      check: (s) => {
        const mid = Math.round(s.config.steps / 2);
        return (
          (makeSchedule('cosine', s.config.steps).alphaBar[mid] as number) >
          3 * (makeSchedule('linear', s.config.steps).alphaBar[mid] as number)
        );
      },
    },
  },
];
