import {
  factorize,
  itemMeanPredictor,
  predict,
  rmseOf,
  type FactorState,
  type Rating,
} from '@sim/classic';
import { createRng } from '@sim/core';

export const GENRES = ['action', 'romance', 'sci-fi'] as const;
export const USERS = 40;
export const ITEMS = 30;

export interface RecommenderScenario {
  /** Share of the user-item table that is observed (the rest is what we want to predict). */
  observed: number;
  /** Hidden taste dimensions used by the model. */
  factors: number;
  lambda: number;
  iterations: number;
  method: 'als' | 'sgd';
  /** Noise added to the true ratings before rounding to stars. */
  noise: number;
  seed: number;
}

export const DEFAULT_RECOMMENDER: RecommenderScenario = {
  observed: 0.3,
  factors: 3,
  lambda: 1,
  iterations: 20,
  method: 'als',
  noise: 0.3,
  seed: 5,
};

export interface World {
  /** True rating before noise and rounding, on the 1..5 scale: `USERS x ITEMS`. */
  truth: number[][];
  /** Stars the users would give (1..5) for every pair, including the unobserved ones. */
  stars: number[][];
  /** Main genre of each item and its loading on each genre. */
  itemGenre: number[];
}

export function makeWorld(noise: number, seed: number): World {
  const rng = createRng(seed);
  // every user cares mostly about one genre (a tendency), every item belongs mostly to one genre
  const taste = Array.from({ length: USERS }, () => {
    const main = Math.floor(rng.next() * GENRES.length);
    return GENRES.map((_, g) => (g === main ? 1 : -0.3) + rng.normal() * 0.15);
  });
  const itemGenre = Array.from({ length: ITEMS }, (_, i) => i % GENRES.length);
  const loading = itemGenre.map((main) =>
    GENRES.map((_, g) => (g === main ? 1 : 0) + rng.normal() * 0.1),
  );
  const truth = taste.map((t) =>
    loading.map((l) =>
      Math.min(5, Math.max(1, 3 + 1.1 * t.reduce((s, x, g) => s + x * (l[g] as number), 0))),
    ),
  );
  const stars = truth.map((row) =>
    row.map((v) => Math.min(5, Math.max(1, Math.round(v + rng.normal() * noise)))),
  );
  return { truth, stars, itemGenre };
}

export interface RecommenderRun {
  scenario: RecommenderScenario;
  world: World;
  train: Rating[];
  test: Rating[];
  history: FactorState[];
  final: FactorState;
  trainRmse: number[];
  testRmse: number[];
  baselineMean: number;
  baselineItem: number;
  /** The predicted rating of every pair. */
  predicted: number[][];
}

export function runRecommender(s: RecommenderScenario): RecommenderRun {
  const world = makeWorld(s.noise, s.seed);
  const rng = createRng(s.seed + 99);
  const train: Rating[] = [];
  const test: Rating[] = [];
  for (let u = 0; u < USERS; u++)
    for (let i = 0; i < ITEMS; i++) {
      const r: Rating = { user: u, item: i, value: world.stars[u]![i]! };
      const draw = rng.next();
      if (draw < s.observed) train.push(r);
      else test.push(r);
    }
  const history = [
    ...factorize(train, {
      users: USERS,
      items: ITEMS,
      factors: s.factors,
      lambda: s.lambda,
      iterations: s.iterations,
      method: s.method,
      learningRate: 0.03,
      seed: s.seed,
    }),
  ];
  const final = history[history.length - 1] as FactorState;
  const meanOnly = { U: final.U, V: final.V, mean: final.mean };
  const item = itemMeanPredictor(train, ITEMS);
  const rmse = (f: (u: number, i: number) => number) =>
    Math.sqrt(test.reduce((a, r) => a + (r.value - f(r.user, r.item)) ** 2, 0) / test.length);
  return {
    scenario: s,
    world,
    train,
    test,
    history,
    final,
    trainRmse: history.map((h) => h.trainRmse),
    testRmse: history.map((h) => rmseOf(h, test)),
    baselineMean: rmse(() => meanOnly.mean),
    baselineItem: rmse(item),
    predicted: Array.from({ length: USERS }, (_, u) =>
      Array.from({ length: ITEMS }, (_, i) => Math.min(5, Math.max(1, predict(final, u, i)))),
    ),
  };
}
