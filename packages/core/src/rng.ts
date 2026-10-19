/**
 * Seeded pseudo-random number generator (mulberry32).
 * Every module draws randomness from here so a `seed` in the URL
 * reproduces exactly the same dataset, initial weights and training run.
 */
export interface Rng {
  /** Uniform float in [0, 1). */
  next(): number;
  /** Uniform float in [min, max). */
  uniform(min: number, max: number): number;
  /** Standard normal sample (Box–Muller). */
  normal(): number;
}

export function createRng(seed: number): Rng {
  let state = seed >>> 0;
  let spare: number | null = null;

  const next = (): number => {
    state = (state + 0x6d2b79f5) >>> 0;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };

  return {
    next,
    uniform: (min, max) => min + (max - min) * next(),
    normal: () => {
      if (spare !== null) {
        const value = spare;
        spare = null;
        return value;
      }
      let u = 0;
      while (u === 0) u = next();
      const v = next();
      const radius = Math.sqrt(-2 * Math.log(u));
      spare = radius * Math.sin(2 * Math.PI * v);
      return radius * Math.cos(2 * Math.PI * v);
    },
  };
}

/** Random permutation of 0..n-1 (Fisher–Yates). */
export function permutation(n: number, rng: Rng): Uint32Array {
  const out = new Uint32Array(n);
  for (let i = 0; i < n; i++) out[i] = i;
  for (let i = n - 1; i > 0; i--) {
    const j = Math.floor(rng.next() * (i + 1));
    const tmp = out[i] as number;
    out[i] = out[j] as number;
    out[j] = tmp;
  }
  return out;
}
