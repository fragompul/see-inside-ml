import type { ClassificationSetup, ClassifierRun, RegressionSetup, RegressorRun } from './pipeline';
import {
  DEFAULT_DATA,
  buildClassification,
  buildRegression,
  runClassifier,
  runRegressor,
} from './pipeline';
import type { ParamSpec, ParamValues } from './params';
import { withDefaults } from './params';
import { DEFAULT_TABULAR, buildTabular, type TabularSetup } from './tabularPipeline';
import {
  DEFAULT_PROJECTION_SETUP,
  buildProjectionData,
  type ProjectionDatasetId,
  type ProjectionSetup,
} from './projectionData';
import { runProjector, type EmbedModel, type ProjectorRun } from './projectorPipeline';
import {
  DEFAULT_CLUSTER_SETUP,
  buildClusterData,
  runClusterer,
  type ClusterRun,
  type ClusterSetup,
} from './clusterPipeline';
import type { Classifier, ClassificationSet, Clusterer, RegressionSet, Regressor } from '@sim/core';

/**
 * An experiment is a setup (data + parameters) and a claim about the result. The page applies the
 * setup when the learner clicks it, and the test suite evaluates the claim on the very same run:
 * what the text says is what was measured.
 */
export interface Experiment<R, S> {
  id: string;
  title: string;
  /** What to look for once the experiment is on screen. */
  text: string;
  setup?: Partial<S>;
  params?: ParamValues;
  claim: (run: R) => boolean;
}

export type ClassifierExperiment = Experiment<ClassifierRun, ClassificationSetup>;
export type RegressorExperiment = Experiment<RegressorRun, RegressionSetup>;

export interface ClassifierLabSpec {
  id: string;
  create: (params: ParamValues) => Classifier;
  params: readonly ParamSpec[];
  defaults: ParamValues;
  dataset: ClassificationSet;
  /** Data defaults that differ from the shared ones. */
  data?: Partial<ClassificationSetup>;
  experiments: readonly ClassifierExperiment[];
}

export interface RegressorLabSpec {
  id: string;
  create: (params: ParamValues) => Regressor;
  params: readonly ParamSpec[];
  defaults: ParamValues;
  dataset: RegressionSet;
  data?: Partial<RegressionSetup>;
  experiments: readonly RegressorExperiment[];
}

export function classificationSetup(
  spec: ClassifierLabSpec,
  patch: Partial<ClassificationSetup> = {},
): ClassificationSetup {
  return { ...DEFAULT_DATA, dataset: spec.dataset, ...spec.data, ...patch };
}

export function regressionSetup(
  spec: RegressorLabSpec,
  patch: Partial<RegressionSetup> = {},
): RegressionSetup {
  return { ...DEFAULT_DATA, dataset: spec.dataset, ...spec.data, ...patch };
}

/** Runs a classifier lab's experiment exactly as the page would. */
export function runClassifierExperiment(
  spec: ClassifierLabSpec,
  exp: ClassifierExperiment,
): ClassifierRun {
  const params = { ...withDefaults(spec.params, spec.defaults), ...exp.params };
  return runClassifier(
    spec.create(params),
    buildClassification(classificationSetup(spec, exp.setup)),
  );
}

export function runRegressorExperiment(
  spec: RegressorLabSpec,
  exp: RegressorExperiment,
): RegressorRun {
  const params = { ...withDefaults(spec.params, spec.defaults), ...exp.params };
  return runRegressor(spec.create(params), buildRegression(regressionSetup(spec, exp.setup)));
}

export type TabularExperiment = Experiment<RegressorRun, TabularSetup>;

/** A multi-feature regression lab: named features, coefficients, a known truth to compare with. */
export interface TabularLabSpec {
  id: string;
  create: (params: ParamValues) => Regressor;
  params: readonly ParamSpec[];
  defaults: ParamValues;
  data?: Partial<TabularSetup>;
  experiments: readonly TabularExperiment[];
}

export const tabularSetup = (
  spec: TabularLabSpec,
  patch: Partial<TabularSetup> = {},
): TabularSetup => ({ ...DEFAULT_TABULAR, ...spec.data, ...patch });

export function runTabularExperiment(spec: TabularLabSpec, exp: TabularExperiment): RegressorRun {
  const params = { ...withDefaults(spec.params, spec.defaults), ...exp.params };
  return runRegressor(spec.create(params), buildTabular(tabularSetup(spec, exp.setup)));
}

export type ClusterExperiment = Experiment<ClusterRun, ClusterSetup>;

/** A clustering lab: the algorithm never sees the labels, the page and the tests score it against them. */
export interface ClusterLabSpec {
  id: string;
  create: (params: ParamValues) => Clusterer;
  params: readonly ParamSpec[];
  defaults: ParamValues;
  dataset: ClassificationSet;
  data?: Partial<ClusterSetup>;
  experiments: readonly ClusterExperiment[];
}

export const clusterSetup = (
  spec: ClusterLabSpec,
  patch: Partial<ClusterSetup> = {},
): ClusterSetup => ({
  ...DEFAULT_CLUSTER_SETUP,
  dataset: spec.dataset,
  ...spec.data,
  ...patch,
});

export function runClusterExperiment(spec: ClusterLabSpec, exp: ClusterExperiment): ClusterRun {
  const params = { ...withDefaults(spec.params, spec.defaults), ...exp.params };
  return runClusterer(spec.create(params), buildClusterData(clusterSetup(spec, exp.setup)));
}

export type ProjectorExperiment = Experiment<ProjectorRun, ProjectionSetup>;

/** A dimensionality-reduction lab: a method that yields 2-D layouts, judged by how well it keeps neighbours. */
export interface ProjectorLabSpec {
  id: string;
  create: (params: ParamValues) => EmbedModel;
  params: readonly ParamSpec[];
  defaults: ParamValues;
  dataset: ProjectionDatasetId;
  /** Datasets offered in the picker (default: all). */
  datasets?: readonly ProjectionDatasetId[];
  data?: Partial<ProjectionSetup>;
  experiments: readonly ProjectorExperiment[];
}

export const projectionSetup = (
  spec: ProjectorLabSpec,
  patch: Partial<ProjectionSetup> = {},
): ProjectionSetup => ({
  ...DEFAULT_PROJECTION_SETUP,
  dataset: spec.dataset,
  ...spec.data,
  ...patch,
});

export function runProjectorExperiment(
  spec: ProjectorLabSpec,
  exp: ProjectorExperiment,
): ProjectorRun {
  const params = { ...withDefaults(spec.params, spec.defaults), ...exp.params };
  return runProjector(spec.create(params), buildProjectionData(projectionSetup(spec, exp.setup)));
}
