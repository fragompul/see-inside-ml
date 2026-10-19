export {
  ClassifierLab,
  type ClassifierContext,
  type ClassifierLabProps,
  type LabText,
} from './ClassifierLab';
export { contourSegments, drawContour } from './contours';
export { ClusterLab, type ClusterContext, type ClusterLabProps } from './ClusterLab';
export {
  DEFAULT_CLUSTER_SETUP,
  buildClusterData,
  clusterRegions,
  runClusterer,
  scoreClusters,
  type ClusterRun,
  type ClusterScores,
  type ClusterSetup,
} from './clusterPipeline';
export { drawEllipse } from './ellipse';
export { Dendrogram } from './Dendrogram';
export { EmbeddingPlane } from './EmbeddingPlane';
export { ProjectorLab, type ProjectorContext, type ProjectorLabProps } from './ProjectorLab';
export {
  DEFAULT_PROJECTION_SETUP,
  PROJECTION_DATASETS,
  buildProjectionData,
  projectionDatasetInfo,
  type ProjectionData,
  type ProjectionDatasetId,
  type ProjectionSetup,
} from './projectionData';
export {
  oneShot,
  runProjector,
  scoreEmbedding,
  type EmbedFrame,
  type EmbedModel,
  type EmbeddingScores,
  type ProjectorRun,
} from './projectorPipeline';
export { CurvePlane, type Band, type Curve } from './CurvePlane';
export { DataPlane, type EditTool, type Painter } from './DataPlane';
export {
  classificationSetup,
  regressionSetup,
  runClassifierExperiment,
  runRegressorExperiment,
  runProjectorExperiment,
  projectionSetup,
  type ProjectorExperiment,
  type ProjectorLabSpec,
  runClusterExperiment,
  clusterSetup,
  type ClusterExperiment,
  type ClusterLabSpec,
  runTabularExperiment,
  tabularSetup,
  type TabularExperiment,
  type TabularLabSpec,
  type ClassifierExperiment,
  type ClassifierLabSpec,
  type Experiment,
  type RegressorExperiment,
  type RegressorLabSpec,
} from './experiments';
export { MetricsStrip, fix, pct, type Metric } from './MetricsStrip';
export { ParamPanel } from './ParamPanel';
export {
  ladderIndex,
  num,
  str,
  flag,
  withDefaults,
  type ParamSpec,
  type ParamValue,
  type ParamValues,
} from './params';
export {
  DEFAULT_DATA,
  buildClassification,
  buildRegression,
  predictCurve,
  predictRegions,
  runClassifier,
  runRegressor,
  type ClassificationSetup,
  type ClassifierRun,
  type RegressionSetup,
  type RegressorRun,
  type Regions,
  type SplitData,
} from './pipeline';
export { RegressorLab, type RegressorContext, type RegressorLabProps } from './RegressorLab';
export { LabHeader, StageNav } from './StageNav';
export { TabularLab, type TabularContext, type TabularLabProps } from './TabularLab';
export { CoefficientPlot, PredictedActual, type CoefficientRow } from './tabularCharts';
export {
  DEFAULT_TABULAR,
  SYNTHETIC_TRUTH,
  TABULAR_SOURCES,
  buildTabular,
  type TabularSetup,
  type TabularSource,
} from './tabularPipeline';
export { RegionThumb } from './RegionThumb';
export { TreeDiagram } from './TreeDiagram';
export { StepPlayer } from './StepPlayer';
export { addRow, moveRow, removeRow } from './tableEdit';
