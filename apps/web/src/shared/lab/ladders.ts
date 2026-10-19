/** Learning rates and friends are best explored on a log-ish scale, so sliders pick from a ladder. */
export const LR_LADDER = [0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3] as const;
export const BATCH_LADDER = [1, 5, 10, 20, 40, 80, 150, 250] as const;
export const REG_LADDER = [0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1] as const;

export const nearestIndex = (ladder: readonly number[], value: number): number => {
  let best = 0;
  let bestDistance = Infinity;
  ladder.forEach((v, i) => {
    const distance = Math.abs(Math.log(v) - Math.log(Math.max(value, 1e-12)));
    if (distance < bestDistance) {
      bestDistance = distance;
      best = i;
    }
  });
  return best;
};
