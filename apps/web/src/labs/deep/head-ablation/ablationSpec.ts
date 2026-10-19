import type { LoadedTransformer } from '@sim/deep';
import type { ParamSpec } from '@/shared/lab-kit';
import { runAblation, type AblationRun, type AblationScenario } from './ablationCore';

export const PROMPTS = [
  'Anna found a ball.',
  'Tom found a ball.',
  'Anna and Max went to the park.',
  'Eva found a lamp. She took',
];

export const ablationParams: ParamSpec[] = [
  {
    key: 'mode',
    label: 'How a head is silenced',
    kind: 'select',
    options: [
      { value: 'zero', label: 'Output set to zero' },
      { value: 'mean', label: 'Output replaced by its average' },
    ],
  },
  {
    key: 'prompt',
    label: 'Prompt to continue',
    kind: 'select',
    options: PROMPTS.map((p) => ({ value: p, label: p })),
  },
];

export interface AblationExperiment {
  id: string;
  title: string;
  text: string;
  scenario: Partial<AblationScenario>;
  claim: (run: AblationRun, model: LoadedTransformer) => boolean;
}

const base = (loaded: LoadedTransformer, r: AblationRun, change: Partial<AblationScenario>) =>
  runAblation(loaded, { ...r.scenario, ...change });
const flat = (m: number[][]) => m.flat();
const argmax = (v: number[]) => v.indexOf(Math.max(...v));
const argmin = (v: number[]) => v.indexOf(Math.min(...v));

export const ablationExperiments: AblationExperiment[] = [
  {
    id: 'uneven',
    title: 'Some heads matter far more than others',
    text: 'Silencing one head at a time, two of the eight (the first two of the first layer) account for more than half of all the damage, while one head (the last of the first layer) can be removed with no measurable change.',
    scenario: {},
    claim: (r) => {
      const d = flat(r.result.delta);
      const positive = d.filter((x) => x > 0).reduce((a, b) => a + b, 0);
      const top = [...d].sort((a, b) => b - a);
      return (
        Math.abs((top[0] as number) + (top[1] as number)) / positive > 0.55 &&
        (r.result.delta[0] as number[])[0]! > 0.2 &&
        (r.result.delta[0] as number[])[1]! > 0.15 &&
        Math.abs((r.result.delta[0] as number[])[3]!) < 0.01
      );
    },
  },
  {
    id: 'redundant',
    title: 'Half the heads can go, quietly',
    text: 'Removing the three least useful heads raises the average loss on the stories by less than 0.12 nats, from 0.69. Removing all eight more than triples it, to 2.1, and the model starts predicting the end of the text.',
    scenario: {},
    claim: (r) => {
      const c = r.result.cumulative;
      return (
        (c[3] as { loss: number }).loss - r.result.baseline < 0.12 &&
        (c[8] as { loss: number }).loss > 3 * r.result.baseline
      );
    },
  },
  {
    id: 'gender',
    title: 'The second layer knows who “She” is',
    text: 'Given “Anna found a ball.”, the model puts all its probability on “She”. Silence the four heads of the second layer and it is torn between “She” and “He”: the link from a name to its pronoun is made there.',
    scenario: { prompt: 'Anna found a ball.', silenced: ['1.0', '1.1', '1.2', '1.3'] },
    claim: (r, m) => {
      const clean = base(m, r, { silenced: [] });
      const she = (run: AblationRun, token: string) =>
        run.next.find((n) => n.token.trim() === token)?.p ?? 0;
      return she(clean, 'She') > 0.99 && she(r, 'She') < 0.6 && she(r, 'He') > 0.3;
    },
  },
  {
    id: 'mean',
    title: 'Replacing by the average hurts less than zeroing',
    text: 'Setting a head to zero hands the next layer an input it never saw in training. Replacing the output by its average over the text is gentler, and for each of the four most important heads it costs less.',
    scenario: { mode: 'mean' },
    claim: (r, m) => {
      const zero = base(m, r, { mode: 'zero' });
      const order = flat(zero.result.delta)
        .map((d, i) => ({ d, i }))
        .sort((a, b) => b.d - a.d)
        .slice(0, 4);
      return order.every(({ i }) => flat(r.result.delta)[i]! < flat(zero.result.delta)[i]!);
    },
  },
  {
    id: 'patterns',
    title: 'The most scattered head is the least needed',
    text: 'The head that spreads its attention most evenly over the earlier words (the last head of the first layer) is the one whose removal changes nothing, while the head that looks hardest at the previous word (the first head of the first layer) is the one that costs most.',
    scenario: {},
    claim: (r) => {
      const delta = flat(r.result.delta);
      const prev = flat(r.patterns.map((row) => row.map((p) => p.previous)));
      const entropy = flat(r.patterns.map((row) => row.map((p) => p.entropy)));
      return argmax(delta) === argmax(prev) && argmin(delta) === argmax(entropy);
    },
  },
];
