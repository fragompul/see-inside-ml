import { GanSession, DEFAULT_GAN_LAB, type GanLabConfig, type GanStats } from '@sim/deep';

/**
 * One-click experiments. `claim.check` verifies, on a session trained for `claim.iterations`
 * iterations with the lesson's configuration, the statement the text makes; `lessons.test.ts`
 * runs them all.
 */
export interface Lesson {
  id: string;
  title: string;
  config: Partial<GanLabConfig>;
  scrollTo: string;
  text: string;
  claim: { iterations: number; check: (session: GanSession) => boolean };
}

const ITERATIONS = 1500;

const last = (s: GanSession): GanStats => s.history[s.history.length - 1] as GanStats;

/** A sibling session with the same settings but some changes, trained for the same time. */
function sibling(s: GanSession, patch: Partial<GanLabConfig>): GanSession {
  const other = new GanSession({ ...DEFAULT_GAN_LAB, ...s.config, ...patch });
  other.train(ITERATIONS);
  return other;
}

/** Mean generator signal over the first `records` records. */
const earlySignal = (s: GanSession, records = 10) =>
  s.history.slice(0, records).reduce((a, h) => a + h.signal, 0) /
  Math.min(records, s.history.length);

export const LESSONS: readonly Lesson[] = [
  {
    id: 'fair',
    title: 'A fair duel',
    config: { dataset: 'gaussians8', dSteps: 1, lrD: 0.002, lrG: 0.002, lossG: 'nonsaturating' },
    scrollTo: 'gan-duel',
    text: 'Both players learn at the same pace. The generator finds all eight clusters, and the discriminator ends up unable to tell real from fake: it says about 50% everywhere (the map fades to grey).',
    claim: {
      iterations: ITERATIONS,
      check: (s) => {
        const h = last(s);
        return h.modes === 8 && h.near > 0.8 && Math.abs(h.dReal - h.dFake) < 0.15;
      },
    },
  },
  {
    id: 'collapse',
    title: 'Mode collapse',
    config: { dataset: 'gaussians8', dSteps: 3, lrD: 0.01 },
    scrollTo: 'gan-duel',
    text: 'Now the discriminator is much stronger. The generator finds the single spot that fools it best and piles all its points there, hopping from cluster to cluster instead of covering all eight. This is mode collapse.',
    claim: {
      iterations: ITERATIONS,
      check: (s) => {
        const h = last(s);
        const fair = last(sibling(s, { dSteps: 1, lrD: 0.002 }));
        return (h.modes as number) <= 4 && h.distance > 3 * fair.distance;
      },
    },
  },
  {
    id: 'lazy',
    title: 'A lazy discriminator',
    config: { dataset: 'gaussians8', lrD: 0.0003 },
    scrollTo: 'gan-duel',
    text: 'If the discriminator learns far more slowly, it gives the generator no useful guidance: most generated points never reach the real clusters. A GAN is only as good as its critic.',
    claim: {
      iterations: ITERATIONS,
      check: (s) => {
        const fair = last(sibling(s, { lrD: 0.002 }));
        return last(s).near < 0.3 && last(s).distance > 3 * fair.distance;
      },
    },
  },
  {
    id: 'minimax',
    title: 'Fading gradients',
    config: { dataset: 'gaussians8', lossG: 'minimax', dSteps: 3, lrD: 0.01 },
    scrollTo: 'gan-charts',
    text: 'With the original minimax loss, the more the discriminator wins, the weaker the signal the generator receives. Compare the "gradient to the generator" chart with the non-saturating loss under the same strong discriminator: it is several times smaller.',
    claim: {
      iterations: ITERATIONS,
      check: (s) => earlySignal(s) < 0.5 * earlySignal(sibling(s, { lossG: 'nonsaturating' })),
    },
  },
  {
    id: 'ring',
    title: 'Learning a curve',
    config: { dataset: 'ring', dSteps: 1, lrD: 0.002, lossG: 'nonsaturating' },
    scrollTo: 'gan-duel',
    text: 'No clusters this time: the generator has to bend its Gaussian noise into a thin circle. Turn on the generator grid to see how it stretches and wraps the plane.',
    claim: { iterations: ITERATIONS, check: (s) => last(s).near > 0.9 && last(s).distance < 0.2 },
  },
];
