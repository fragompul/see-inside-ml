import type { ParamSpec } from '@/shared/lab-kit';
import {
  FLOW_DATASETS,
  runFlowLab,
  type FlowDataset,
  type FlowLabRun,
  type FlowScenario,
} from './flowCore';

export const flowParams: ParamSpec[] = [
  {
    key: 'dataset',
    label: 'Data',
    kind: 'select',
    options: (Object.keys(FLOW_DATASETS) as FlowDataset[]).map((id) => ({
      value: id,
      label: FLOW_DATASETS[id],
    })),
  },
  {
    key: 'layers',
    label: 'Coupling layers',
    kind: 'ladder',
    values: [1, 2, 4, 8],
    format: (v) => String(v),
  },
  {
    key: 'steps',
    label: 'Training steps',
    kind: 'ladder',
    values: [200, 600, 1200],
    format: (v) => String(v),
  },
];

export interface FlowExperiment {
  id: string;
  title: string;
  text: string;
  scenario: Partial<FlowScenario>;
  claim: (run: FlowLabRun) => boolean;
}

const sibling = (r: FlowLabRun, change: Partial<FlowScenario>) =>
  runFlowLab({ ...r.scenario, ...change });

export const flowExperiments: FlowExperiment[] = [
  {
    id: 'layers',
    title: 'Each layer bends the space a little more',
    text: 'One coupling layer can only shear and stretch, so on the two moons it does no better than a single Gaussian (2.60 against 2.68 nats per point). Two, four and eight layers give 2.33, 1.91 and 1.64: the same data becomes about 1 nat per point cheaper to describe.',
    scenario: { dataset: 'moons', layers: 8 },
    claim: (r) => {
      const nll = [1, 2, 4, 8].map((layers) => sibling(r, { layers }).nll);
      return (
        nll.every((v, i) => i === 0 || v < (nll[i - 1] as number)) &&
        Math.abs((nll[0] as number) - r.gaussianNll) < 0.1 &&
        r.gaussianNll - (nll[3] as number) > 0.9
      );
    },
  },
  {
    id: 'exact',
    title: 'Exactly invertible',
    text: 'Every coupling layer can be undone exactly, so any point can be pushed into the Gaussian and recovered afterwards. The largest error over 300 points is of the order of 1e-15: rounding, nothing else.',
    scenario: { dataset: 'moons', layers: 8 },
    claim: (r) => r.invertError < 1e-9,
  },
  {
    id: 'held-out',
    title: 'The likelihood is exact, and it holds on new data',
    text: 'A flow gives the exact probability density of any point. On 600 new moon points the average cost is 1.69 nats, only 0.05 more than on the 600 it was trained on, and far below the 2.68 of the best Gaussian.',
    scenario: { dataset: 'moons', layers: 8 },
    claim: (r) => r.heldOutNll - r.nll < 0.15 && r.heldOutNll < r.gaussianNll - 0.8,
  },
  {
    id: 'ring',
    title: 'A ring needs depth',
    text: 'One layer cannot turn a Gaussian into a ring: its likelihood is no better than the Gaussian’s. With four layers the cost falls by 1.3 nats per point.',
    scenario: { dataset: 'ring', layers: 4 },
    claim: (r) => {
      const shallow = sibling(r, { layers: 1 });
      return Math.abs(shallow.nll - shallow.gaussianNll) < 0.05 && r.gaussianNll - r.nll > 1;
    },
  },
  {
    id: 'spiral',
    title: 'Some shapes are hard',
    text: 'The spiral winds around twice, and even eight small layers trained for 600 steps recover only 0.1 nats of the 2.83 a Gaussian costs. More layers and more training would help; this one is a test of patience.',
    scenario: { dataset: 'spiral', layers: 8 },
    claim: (r) => r.gaussianNll - r.nll < 0.2 && r.gaussianNll - r.nll > 0,
  },
];
