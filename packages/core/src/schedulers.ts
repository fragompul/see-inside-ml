export const LR_SCHEDULES = ['constant', 'exponential', 'cosine'] as const;
export type LrSchedule = (typeof LR_SCHEDULES)[number];

export const COSINE_PERIOD = 500;
export const COSINE_FLOOR = 0.01;
export const EXPONENTIAL_RATE = 0.96;
export const EXPONENTIAL_EPOCHS = 100;

/** Learning rate for a given epoch. */
export function learningRateAt(schedule: LrSchedule, baseLr: number, epoch: number): number {
  switch (schedule) {
    case 'constant':
      return baseLr;
    case 'exponential':
      return baseLr * EXPONENTIAL_RATE ** (epoch / EXPONENTIAL_EPOCHS);
    case 'cosine': {
      const floor = baseLr * COSINE_FLOOR;
      return (
        floor +
        0.5 * (baseLr - floor) * (1 + Math.cos((Math.PI * (epoch % COSINE_PERIOD)) / COSINE_PERIOD))
      );
    }
  }
}

export const scheduleInfo: Record<LrSchedule, { label: string; note: string }> = {
  constant: { label: 'Constant', note: 'The learning rate never changes.' },
  exponential: { label: 'Exponential decay', note: 'Shrinks by 4% every 100 epochs.' },
  cosine: {
    label: 'Cosine annealing',
    note: 'Glides down to 1% of the base rate and restarts every 500 epochs.',
  },
};
