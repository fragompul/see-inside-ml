import { it } from 'vitest';
import { DEFAULT_CONFIG, PlaygroundSession, type PlaygroundConfig } from '../src/session';

declare const performance: { now(): number };
declare const console: { log(message: string): void };

const layers = (sizes: number[]): PlaygroundConfig['hidden'] =>
  sizes.map((neurons) => ({ neurons, activation: 'tanh' as const }));
const make = (patch: Partial<PlaygroundConfig>) =>
  new PlaygroundSession({ ...DEFAULT_CONFIG, ...patch });

/** Median milliseconds per call over `runs` runs, after a warm-up. */
function time(fn: () => void, runs = 25): number {
  for (let i = 0; i < 3; i++) fn();
  const samples: number[] = [];
  for (let i = 0; i < runs; i++) {
    const t0 = performance.now();
    fn();
    samples.push(performance.now() - t0);
  }
  samples.sort((a, b) => a - b);
  return samples[Math.floor(samples.length / 2)] as number;
}

it('prints engine timings', () => {
  const small = make({ hidden: layers([4, 3]) });
  const medium = make({
    hidden: layers([8, 8, 8]),
    features: ['X1', 'X2', 'X1^2', 'X2^2', 'X1*X2'],
  });
  const large = make({ hidden: layers([16, 16, 16, 16, 16, 16]), samples: 1000 });
  medium.step(5);

  const rows: [string, number][] = [
    ['epoch  | 300 samples, batch 20, layers [4, 3]', time(() => void small.step(1))],
    [
      'epoch  | 300 samples, batch 20, layers [8, 8, 8], 5 features',
      time(() => void medium.step(1)),
    ],
    ['epoch  | 1000 samples, batch 20, 6 × 16 neurons', time(() => void large.step(1))],
    ['snapshot | boundary 50×50', time(() => void medium.snapshot({ resolution: 50 }))],
    [
      'snapshot | boundary + all neuron maps 50×50',
      time(() => void medium.snapshot({ resolution: 50, neuronMaps: true })),
    ],
    [
      'snapshot | boundary + all neuron maps 100×100',
      time(() => void medium.snapshot({ resolution: 100, neuronMaps: true })),
    ],
  ];
  const lines = rows.map(([name, ms]) => `${name.padEnd(62)} ${ms.toFixed(2).padStart(8)} ms`);
  console.log(`\nMedian time per call\n${lines.join('\n')}\n`);
});
