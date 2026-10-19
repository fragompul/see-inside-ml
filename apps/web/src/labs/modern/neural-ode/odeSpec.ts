import type { ParamSpec } from '@/shared/lab-kit';
import {
  DATASET_LABEL,
  runOde,
  type OdeDataset,
  type OdeResult,
  type OdeScenario,
} from './odeCore';

export const odeParams: ParamSpec[] = [
  {
    key: 'dataset',
    label: 'Data',
    kind: 'select',
    options: (Object.keys(DATASET_LABEL) as OdeDataset[]).map((id) => ({
      value: id,
      label: DATASET_LABEL[id],
    })),
  },
  {
    key: 'solver',
    label: 'Solver',
    kind: 'segmented',
    options: [
      { value: 'euler', label: 'Euler' },
      { value: 'rk4', label: 'Runge-Kutta 4' },
    ],
  },
  {
    key: 'steps',
    label: 'Solver steps',
    kind: 'ladder',
    values: [1, 2, 4, 8],
    format: (v) => String(v),
  },
];

export interface OdeExperiment {
  id: string;
  title: string;
  text: string;
  scenario: Partial<OdeScenario>;
  claim: (run: OdeResult) => boolean;
}

const sibling = (r: OdeResult, change: Partial<OdeScenario>) =>
  runOde({ ...r.scenario, ...change });

export const odeExperiments: OdeExperiment[] = [
  {
    id: 'consistent',
    title: 'With a good solver the model really is an ODE',
    text: 'Trained with eight Runge-Kutta steps, the network classifies the ring and disc correctly (98% to 99%) and a solver with sixteen times as many steps gives the same final states to within 0.01 and the same accuracy. The learned velocity field means what it says.',
    scenario: { dataset: 'circles', solver: 'rk4', steps: 8 },
    claim: (r) => r.stateError < 0.05 && Math.abs(r.own - r.fine) < 0.02 && r.own > 0.95,
  },
  {
    id: 'resnet',
    title: 'With a crude solver it is only a network with skip connections',
    text: 'Trained with just two Euler steps, the model is right 98% of the time, yet the very same weights run through an accurate solver put points about 3.7 units away from where the model sent them, and the accuracy drops to 66%. Two Euler steps are two residual layers: what was learned is the discretisation, not the flow.',
    scenario: { dataset: 'circles', solver: 'euler', steps: 2 },
    claim: (r) => r.own > 0.95 && r.fine < 0.8 && r.stateError > 2,
  },
  {
    id: 'order',
    title: 'More steps, a truer flow',
    text: 'With Runge-Kutta, one step is a layer (the final states are 3 units from the accurate ones); two steps bring them to 0.3, and eight to 0.01. The error falls much faster than for Euler, whose error after eight steps is still above 1.',
    scenario: { dataset: 'circles', solver: 'rk4', steps: 2 },
    claim: (r) => {
      const one = sibling(r, { steps: 1 });
      const eight = sibling(r, { steps: 8 });
      const euler8 = sibling(r, { solver: 'euler', steps: 8 });
      return (
        one.stateError > 2 && r.stateError < 0.5 && eight.stateError < 0.05 && euler8.stateError > 1
      );
    },
  },
  {
    id: 'one-layer',
    title: 'One step can classify, but it is no longer a flow',
    text: 'A single Runge-Kutta step already separates the two classes on its own terms (98% to 99%), but evaluated as a continuous flow the same weights get barely half of them right. Accuracy on the task alone does not show whether the model is an ODE.',
    scenario: { dataset: 'circles', solver: 'rk4', steps: 1 },
    claim: (r) => r.own > 0.95 && r.fine < 0.65,
  },
  {
    id: 'moons',
    title: 'Two moons work the same way',
    text: 'On two moons, eight Runge-Kutta steps give a model that is consistent with its own accurate solution (final states within 0.05), while one Euler step does not.',
    scenario: { dataset: 'moons', solver: 'rk4', steps: 8 },
    claim: (r) => r.stateError < 0.05 && sibling(r, { solver: 'euler', steps: 1 }).stateError > 1,
  },
];
