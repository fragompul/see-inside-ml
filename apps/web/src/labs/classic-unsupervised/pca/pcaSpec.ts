import { KernelPCA, PCA, type PcaKernel } from '@sim/classic';
import {
  num,
  oneShot,
  str,
  type ParamSpec,
  type ProjectorExperiment,
  type ProjectorLabSpec,
} from '@/shared/lab-kit';

const params: ParamSpec[] = [
  {
    key: 'method',
    label: 'Method',
    kind: 'segmented',
    options: [
      { value: 'pca', label: 'PCA (linear)' },
      { value: 'kpca', label: 'Kernel PCA' },
    ],
  },
  {
    key: 'kernel',
    label: 'Kernel',
    kind: 'segmented',
    options: [
      { value: 'rbf', label: 'RBF' },
      { value: 'poly', label: 'Polynomial' },
      { value: 'cosine', label: 'Cosine' },
    ],
    showIf: (v) => v.method === 'kpca',
  },
  {
    key: 'gamma',
    label: 'Kernel width γ',
    kind: 'ladder',
    values: [0.005, 0.02, 0.05, 0.2, 1],
    format: (v) => String(v),
    hint: 'Small γ = wide kernel, close to linear PCA; large γ = very local.',
    showIf: (v) => v.method === 'kpca' && v.kernel !== 'cosine',
  },
];

export const pcaSpec: ProjectorLabSpec = {
  id: 'pca',
  create: (p) =>
    str(p, 'method') === 'kpca'
      ? oneShot((x) => {
          const m = new KernelPCA({
            components: 2,
            kernel: str(p, 'kernel') as PcaKernel,
            gamma: num(p, 'gamma'),
            degree: 2,
            coef0: 1,
          }).fit(x);
          const total = m.eigenvalues.reduce((a, b) => a + b, 0);
          return {
            embedding: m.embedding,
            info: `two kernel components, eigenvalues ${m.eigenvalues[0]?.toFixed(2)} and ${m.eigenvalues[1]?.toFixed(2)} (sum ${total.toFixed(2)})`,
          };
        })
      : oneShot((x) => {
          const m = new PCA({ components: 2 }).fit(x);
          const share =
            (m.explainedVarianceRatio[0] as number) + (m.explainedVarianceRatio[1] as number);
          return {
            embedding: m.transform(x),
            info: `the two components explain ${(share * 100).toFixed(1)}% of the variance`,
          };
        }),
  params,
  defaults: { method: 'pca', kernel: 'rbf', gamma: 0.05 },
  dataset: 'swiss',
  datasets: ['swiss', 'scurve', 'helix', 'sphere', 'clusters', 'iris', 'wine', 'digits'],
  data: { n: 150, noise: 0.1 },
  experiments: [
    {
      id: 'iris',
      title: 'Iris: two axes are enough',
      text: 'Four measurements of a flower vary mostly along two directions: the first two components keep almost all the variance and the species stay separate.',
      setup: { dataset: 'iris' },
      params: { method: 'pca' },
      claim: (run) => {
        const p = new PCA().fit(run.data.x);
        return (
          p.explainedVarianceRatio[0]! + p.explainedVarianceRatio[1]! > 0.97 &&
          run.scores.trustworthiness > 0.95
        );
      },
    },
    {
      id: 'wine',
      title: 'Wine: half the story in two axes',
      text: 'Thirteen chemical measurements: two components keep a bit over half of the variance, and it takes about eight to reach 90%.',
      setup: { dataset: 'wine' },
      params: { method: 'pca' },
      claim: (run) => {
        const p = new PCA().fit(run.data.x);
        const two = p.explainedVarianceRatio[0]! + p.explainedVarianceRatio[1]!;
        return two > 0.5 && two < 0.6 && p.componentsFor(0.9) >= 6;
      },
    },
    {
      id: 'digits',
      title: 'Digits: 256 dimensions, no short cut',
      text: 'Pictures of digits have no dominant directions: two components keep little of the variance, and it takes dozens to describe them well.',
      setup: { dataset: 'digits', n: 150 },
      params: { method: 'pca' },
      claim: (run) => {
        const p = new PCA().fit(run.data.x);
        return (
          p.explainedVarianceRatio[0]! + p.explainedVarianceRatio[1]! < 0.35 &&
          p.componentsFor(0.9) > 20
        );
      },
    },
    {
      id: 'roll',
      title: 'The swiss roll gets squashed',
      text: 'A linear projection can only look at the roll from a side: layers of the sheet that are far apart along the sheet land on top of each other, so many neighbours in the picture are false.',
      setup: { dataset: 'swiss', noise: 0.1 },
      params: { method: 'pca' },
      claim: (run) => run.scores.preservation < 0.6 && run.scores.trustworthiness < 0.95,
    },
    {
      id: 'spring',
      title: 'A spring is nearly one-dimensional',
      text: 'The helix is a single curve, so even a flat view keeps most neighbours together.',
      setup: { dataset: 'helix', noise: 0.05 },
      params: { method: 'pca' },
      claim: (run) => run.scores.preservation > 0.85,
    },
    {
      id: 'sharp-kernel',
      title: 'A kernel that is too sharp',
      text: 'With a very narrow RBF kernel every point only “sees” its immediate neighbours: kernel PCA stops describing the global shape and the picture gets worse than plain PCA.',
      setup: { dataset: 'swiss', noise: 0.1 },
      params: { method: 'kpca', kernel: 'rbf', gamma: 1 },
      claim: (run) => run.scores.trustworthiness < 0.8,
    },
  ] satisfies ProjectorExperiment[],
};
