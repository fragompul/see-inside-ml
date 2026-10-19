import {
  DecisionTree,
  GaussianNB,
  KnnClassifier,
  LogisticRegression,
  SVC,
  StackingClassifier,
  VotingClassifier,
} from '@sim/classic';
import { accuracy, type Classifier } from '@sim/core';
import {
  flag,
  str,
  type ClassifierLabSpec,
  type ClassifierRun,
  type ParamSpec,
} from '@/shared/lab-kit';

export const MEMBERS = [
  { key: 'logistic', label: 'Logistic regression', make: () => new LogisticRegression() },
  { key: 'knn', label: 'k-NN (k = 9)', make: () => new KnnClassifier({ k: 9 }) },
  {
    key: 'tree',
    label: 'Decision tree (depth 4)',
    make: () => new DecisionTree({ maxDepth: 4, minSamplesLeaf: 4 }),
  },
  { key: 'nb', label: 'Naive Bayes', make: () => new GaussianNB() },
  { key: 'svm', label: 'SVM (RBF)', make: () => new SVC({ kernel: 'rbf', C: 1, gamma: 1 }) },
] as const;

const params: ParamSpec[] = [
  {
    key: 'scheme',
    label: 'How to combine',
    kind: 'segmented',
    options: [
      { value: 'hard', label: 'Majority vote' },
      { value: 'soft', label: 'Average probabilities' },
      { value: 'stack', label: 'Stacking' },
    ],
  },
  ...MEMBERS.map((m): ParamSpec => ({ key: m.key, label: m.label, kind: 'switch' })),
];

export type Ensemble = VotingClassifier | StackingClassifier;

export const ensembleOf = (m: Classifier): Ensemble | null =>
  m instanceof VotingClassifier || m instanceof StackingClassifier ? m : null;

export const membersOf = (e: Ensemble): Classifier[] =>
  e instanceof VotingClassifier ? e.members : e.bases;

/** Accuracy of every member and of the ensemble on the held-out (or training) rows. */
export function memberAccuracies(run: ClassifierRun): { members: number[]; ensemble: number } {
  const e = ensembleOf(run.model) as Ensemble;
  const { test, train } = run.data;
  const set = test.x.rows > 0 ? test : train;
  return {
    members: membersOf(e).map((m) => accuracy(set.y, m.predict(set.x))),
    ensemble: accuracy(set.y, e.predict(set.x)),
  };
}

export const ensemblesSpec: ClassifierLabSpec = {
  id: 'ensembles',
  create: (p) => {
    const makers = MEMBERS.filter((m) => flag(p, m.key)).map((m) => m.make);
    if (makers.length < 2) throw new Error('Switch on at least two models to combine.');
    const scheme = str(p, 'scheme');
    return scheme === 'stack'
      ? new StackingClassifier(makers, { folds: 5, seed: 3 })
      : new VotingClassifier(makers, { voting: scheme === 'soft' ? 'soft' : 'hard' });
  },
  params,
  defaults: { scheme: 'soft', logistic: true, knn: true, tree: true, nb: true, svm: false },
  dataset: 'moons',
  data: { n: 240, noise: 0.4 },
  experiments: [
    {
      id: 'wisdom',
      title: 'The crowd beats the average member',
      text: 'Averaging the probabilities of four different models does better than the typical member, and rarely much worse than the best.',
      params: { scheme: 'soft' },
      claim: (run) => {
        const a = memberAccuracies(run);
        return a.ensemble >= a.members.reduce((s, v) => s + v, 0) / a.members.length;
      },
    },
    {
      id: 'majority',
      title: 'Majority vote',
      text: 'Each model casts one vote; the most popular class wins. Simple, but it throws away how confident each model was.',
      params: { scheme: 'hard' },
      claim: (run) => {
        const a = memberAccuracies(run);
        return a.ensemble >= a.members.reduce((s, v) => s + v, 0) / a.members.length - 0.01;
      },
    },
    {
      id: 'stacking',
      title: 'Stacking learns whom to trust',
      text: 'A second model is trained on the out-of-fold predictions of the first-level models, so it learns how much weight each deserves.',
      params: { scheme: 'stack' },
      claim: (run) => {
        const a = memberAccuracies(run);
        return a.ensemble >= Math.max(...a.members) - 0.04;
      },
    },
    {
      id: 'strong-crowd',
      title: 'Add a strong member',
      text: 'With an RBF SVM in the crowd the ensemble reaches the accuracy of the best individual models.',
      params: { scheme: 'soft', svm: true },
      claim: (run) => {
        const a = memberAccuracies(run);
        return a.ensemble >= Math.max(...a.members) - 0.05;
      },
    },
  ],
};
