import { positionMatrix, type PositionScheme } from '@sim/deep';
import type { ParamSpec } from '@/shared/lab-kit';
import { type PositionalRun, type PositionalScenario } from './positionalCore';

export const SCHEME_LABEL: Record<PositionScheme, string> = {
  sinusoidal: 'Sinusoidal (added)',
  rope: 'Rotary (RoPE)',
  alibi: 'ALiBi (distance penalty)',
  learned: 'Learned table',
};

export const positionalParams: ParamSpec[] = [
  {
    key: 'scheme',
    label: 'Encoding',
    kind: 'select',
    options: (Object.keys(SCHEME_LABEL) as PositionScheme[]).map((id) => ({
      value: id,
      label: SCHEME_LABEL[id],
    })),
  },
  {
    key: 'dim',
    label: 'Dimensions',
    kind: 'ladder',
    values: [2, 8, 32, 128],
    format: (v) => String(v),
    hint: 'Width of the position code (ignored by ALiBi).',
  },
  {
    key: 'length',
    label: 'Sequence length',
    kind: 'ladder',
    values: [16, 32, 48, 64],
    format: (v) => String(v),
  },
  {
    key: 'trained',
    label: 'Length seen in training',
    kind: 'ladder',
    values: [16, 32, 64],
    format: (v) => String(v),
  },
];

export interface PositionalExperiment {
  id: string;
  title: string;
  text: string;
  scenario: Partial<PositionalScenario>;
  claim: (run: PositionalRun) => boolean;
}

export const positionalExperiments: PositionalExperiment[] = [
  {
    id: 'offset-only',
    title: 'Only the distance matters',
    text: 'Each stripe parallel to the diagonal has a single colour: the similarity of two sinusoidal codes depends only on how far apart the positions are, never on where they are. Close positions are most alike and similarity falls with distance.',
    scenario: { scheme: 'sinusoidal', dim: 32, length: 64, trained: 64 },
    claim: (r) => r.spread < 1e-9 && r.near > r.mid && r.mid > r.far && r.near > 0.8,
  },
  {
    id: 'rope-same',
    title: 'Rotation gives the same picture',
    text: 'RoPE does not add anything to the embedding. It rotates queries and keys by an angle that grows with position, so their dot product depends on the offset alone. For identical content the result is exactly the sinusoidal picture, obtained a different way.',
    scenario: { scheme: 'rope', dim: 32, length: 64, trained: 64 },
    claim: (r) => {
      const other = positionMatrix(
        'sinusoidal',
        r.scenario.length,
        r.scenario.dim,
        r.scenario.trained,
      );
      return (
        r.spread < 1e-9 &&
        r.matrix.values.every((v, i) => Math.abs(v - (other.values[i] as number)) < 1e-9)
      );
    },
  },
  {
    id: 'few-dims',
    title: 'Too few dimensions, and far looks near',
    text: 'With only two dimensions the code is a single point going round a circle. Positions six steps apart land almost on the same spot (similarity 0.96) while positions three apart are opposites (−0.99): the encoding cannot tell far from near.',
    scenario: { scheme: 'sinusoidal', dim: 2, length: 64, trained: 64 },
    claim: (r) => (r.byOffset[6] as number) > 0.9 && (r.byOffset[3] as number) < -0.9,
  },
  {
    id: 'learned-blank',
    title: 'A learned table has no future',
    text: 'A table has one row per training position. Ask about a longer text and everything beyond the table is simply missing: only a quarter of the pairs here have any score. Before training the rows are random, so even the defined pairs show no sense of distance.',
    scenario: { scheme: 'learned', dim: 32, length: 32, trained: 16 },
    claim: (r) => Math.abs(r.definedShare - 0.25) < 1e-9 && Math.abs(r.near) < 0.1,
  },
  {
    id: 'alibi-extrapolates',
    title: 'ALiBi works at any length',
    text: 'ALiBi is a rule, not a table: subtract a fixed amount per step of distance. It is defined for every pair no matter how long the text is, here four times the training length, and 10 steps back always costs exactly 5.',
    scenario: { scheme: 'alibi', length: 64, trained: 16 },
    claim: (r) => r.definedShare === 1 && Math.abs((r.byOffset[10] as number) + 5) < 1e-9,
  },
];
