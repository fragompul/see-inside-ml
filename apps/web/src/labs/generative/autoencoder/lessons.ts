import { createRng } from '@sim/core';
import {
  AutoencoderSession,
  DEFAULT_AE_LAB,
  corruptImage,
  type AutoencoderLabConfig,
} from '@sim/deep';

/**
 * One-click experiments. `claim.check` verifies, on a session trained for `claim.epochs` epochs
 * with the lesson's configuration, the statement the text makes; `lessons.test.ts` runs them all.
 */
export interface Lesson {
  id: string;
  title: string;
  config: Partial<AutoencoderLabConfig>;
  scrollTo: string;
  text: string;
  claim: { epochs: number; check: (session: AutoencoderSession) => boolean };
}

const EPOCHS = 30;

const lastRecon = (s: AutoencoderSession) => s.history[s.history.length - 1]?.testRecon as number;

/** A sibling session with the same settings but some changes, trained for the same time. */
function sibling(s: AutoencoderSession, patch: Partial<AutoencoderLabConfig>): AutoencoderSession {
  const other = new AutoencoderSession({ ...DEFAULT_AE_LAB, ...s.config, ...patch });
  other.step(EPOCHS);
  return other;
}

/** Mean squared distance from images decoded from prior samples to their nearest real test digit. */
export function priorRealism(s: AutoencoderSession, samples = 60): number {
  const generated = s.generate(samples, createRng(9));
  let total = 0;
  for (const g of generated) {
    let best = Infinity;
    for (let i = 0; i < s.testCount; i++) {
      const { image } = s.sample(i);
      let e = 0;
      for (let k = 0; k < image.length; k++) e += ((g[k] as number) - (image[k] as number)) ** 2;
      best = Math.min(best, e / image.length);
    }
    total += best;
  }
  return total / generated.length;
}

/** Error of the reconstruction of noisy inputs against the clean images, and of the noise itself. */
export function denoisingError(s: AutoencoderSession, level = 0.4, count = 100) {
  const rng = createRng(5);
  let noisy = 0;
  let rebuilt = 0;
  for (let i = 0; i < count; i++) {
    const { image } = s.sample(i);
    const corrupted = corruptImage(image, level, rng);
    const out = s.reconstruct(corrupted);
    for (let k = 0; k < image.length; k++) {
      noisy += ((corrupted[k] as number) - (image[k] as number)) ** 2;
      rebuilt += ((out[k] as number) - (image[k] as number)) ** 2;
    }
  }
  return { noisy: noisy / count / 256, rebuilt: rebuilt / count / 256 };
}

export const LESSONS: readonly Lesson[] = [
  {
    id: 'bottleneck',
    title: 'The bottleneck',
    config: { mode: 'ae', latent: 2, corruption: 0 },
    scrollTo: 'ae-latent',
    text: 'Two numbers are not much to describe a digit: reconstructions come out blurry and the error stays high. Raise "Numbers in the code" to 8 in the training panel and the pictures sharpen, with about a fifth less error.',
    claim: {
      epochs: EPOCHS,
      check: (s) => lastRecon(s) > 1.15 * lastRecon(sibling(s, { latent: 8 })),
    },
  },
  {
    id: 'denoise',
    title: 'Cleaning noise',
    config: { mode: 'ae', latent: 8, corruption: 0.4 },
    scrollTo: 'ae-read',
    text: 'This autoencoder is trained on damaged inputs but scored against the clean ones, so it learns what digits look like instead of copying pixels. In section 01, add noise to the input: the reconstruction comes out much cleaner than the input.',
    claim: {
      epochs: EPOCHS,
      check: (s) => {
        const { noisy, rebuilt } = denoisingError(s);
        const plain = denoisingError(sibling(s, { corruption: 0 }));
        return rebuilt < 0.5 * noisy && plain.rebuilt > 0.9 * plain.noisy;
      },
    },
  },
  {
    id: 'vae',
    title: 'A map you can sample',
    config: { mode: 'vae', latent: 2, beta: 1, corruption: 0 },
    scrollTo: 'ae-imagine',
    text: 'The KL term pulls every code towards a standard normal blob (dashed circles on the map). So decoding random codes from that blob gives digit-like pictures: in section 03, a VAE imagines far more convincing digits than a plain autoencoder does.',
    claim: {
      epochs: EPOCHS,
      check: (s) => priorRealism(s) < 0.7 * priorRealism(sibling(s, { mode: 'ae' })),
    },
  },
  {
    id: 'beta',
    title: 'Too much regularisation',
    config: { mode: 'vae', latent: 2, beta: 8, corruption: 0 },
    scrollTo: 'ae-train',
    text: 'With β = 8 the KL term dominates: the codes collapse onto the prior (KL near zero, points piled in the middle) and the pictures lose detail. Lower β live in the training panel and watch the clusters spread out again.',
    claim: {
      epochs: EPOCHS,
      check: (s) => {
        const h = s.history[s.history.length - 1];
        const reference = lastRecon(sibling(s, { beta: 1 }));
        return (h?.testKl as number) < 1 && lastRecon(s) > reference + 5;
      },
    },
  },
];
