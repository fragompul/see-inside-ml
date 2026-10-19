# Datasets

Almost everything in the app runs on **generated** data (see `packages/core/src/tabular.ts`):
two-dimensional classification sets, regression curves, Anscombe's quartet, correlated features…

A few tiny, well-known real datasets are bundled for the classical-model labs, exported with
`tools/fixtures/export_datasets.py` from scikit-learn. Each is loaded on demand
(`@sim/core/datasets`) and shown with its source and licence.

| Dataset                              | Rows × features | Source                                                    | Licence       |
| ------------------------------------ | --------------- | --------------------------------------------------------- | ------------- |
| Iris                                 | 150 × 4         | Fisher (1936); UCI Machine Learning Repository            | Public domain |
| Wine                                 | 178 × 13        | UCI Machine Learning Repository                           | CC BY 4.0     |
| Diabetes                             | 442 × 10        | Efron, Hastie, Johnstone, Tibshirani (2004); scikit-learn | BSD-3-Clause  |
| Breast Cancer Wisconsin (diagnostic) | 569 × 30        | UCI Machine Learning Repository                           | CC BY 4.0     |
