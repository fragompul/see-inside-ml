# Roadmap: what is built and what is left

Status on 2026-10-18. The strategy lives in [PLAN_V2.md](../PLAN_V2.md); this file is the working ledger. Commits are
never made by the assistant: [COMMIT_PLAN.md](../COMMIT_PLAN.md) lists, in order, the commands the owner runs.

## Definition of done for a lab

1. **Engine** in `packages/{core,classic,deep}/src`, written from scratch, with unit tests. Where an oracle exists
   (scikit-learn, statsmodels, the Python BPE reference) there is a parity fixture; stochastic methods get property or
   closed-form tests.
2. **Spec** (`<lab>Spec.ts`): parameters plus experiments whose `claim(run)` function is executed by the test suite, so
   every sentence the page shows is measured (claims are measured first, then asserted).
3. **Page** (`<Lab>Page.tsx`) built with the lab kit, light and dark themes, usable at phone width.
4. **End-to-end row** in `apps/web/e2e/classic-labs.spec.ts` (panels, a metric, an experiment, no horizontal scroll).
5. **Registry entry** (`tools/registry_insert.py <track> <file>`) and a **COMMIT_PLAN.md** entry.

## Built (89 labs)

| Track                                         | Labs                                                                                                                                                                                                                                                                                                                                   |
| --------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Foundations (1)                               | foundations (neuron, activations, gradient descent)                                                                                                                                                                                                                                                                                    |
| Classical supervised (18)                     | linear, multiple, logistic regression; k-NN; naive Bayes/LDA/QDA; polynomials and regularisation; robust regression and GLMs; decision trees; random forests; AdaBoost; gradient boosting; SVM; Gaussian processes; Bayesian regression; model showdown; voting and stacking, isotonic regression, splines and MARS, survival analysis |
| Classical unsupervised and probabilistic (20) | k-means, hierarchical, DBSCAN/OPTICS, mean shift, GMM/EM, spectral, PCA/kernel PCA, anomaly detection, SOM, ICA, NMF, manifold learning; HMM, Kalman and particle filters, MCMC, Bayesian inference, Markov chains and PageRank, time-series forecasting, association rules, recommenders                                              |
| Evaluation and optimisation (8)               | bias-variance, validation and leakage, metrics and thresholds, hyperparameter tuning, optimiser race, feature selection and importance, metaheuristics, constrained optimisation                                                                                                                                                       |
| Deep learning (9)                             | MLP playground, CNN, RNN/LSTM/GRU, Transformers, character language model, training toolbox, convolution arithmetic, data augmentation, head ablation                                                                                                                                                                                  |
| Generative (7)                                | autoencoders and VAEs, GANs, diffusion, steering a VAE (conditional and VQ-VAE), GAN recipes compared, normalising flows, guidance and flow matching                                                                                                                                                                                   |
| Language (6)                                  | embeddings, position encodings, KV cache, tokenizers compared, n-grams versus a neural LM, retrieval-augmented generation                                                                                                                                                                                                              |
| Reinforcement learning (8)                    | Q-learning/SARSA gridworld, bandits, policy gradients, Monte Carlo and TD, MCTS, Dyna, actor-critic and GAE, DQN                                                                                                                                                                                                                       |
| Interpretability (5)                          | Shapley, KernelSHAP, LIME, permutation importance, dependence plots; pixel attribution; adversarial robustness; probing; feature visualisation                                                                                                                                                                                         |
| Modern topics (7)                             | compression (quantise, prune, distil), LoRA, scaling laws, mixture of experts, graph neural networks, neural ODEs, learning from human feedback                                                                                                                                                                                        |

Supporting work done: three-package monorepo, track registry with lazy pages, lab kit, design system, theming, GitHub
Pages workflow, `docs/ARCHITECTURE.md`, datasets with attribution, parity fixtures generated by `tools/fixtures/` and
`tools/transformer/`.

## Left to build

Each item follows the definition of done above. Rough scope and the claims worth measuring are noted so that the work
can proceed without new design discussions.

### Reinforcement learning

- [x] **A2C / GAE** (`rl/actor-critic`): variance and bias of the advantage estimate against lambda.
- [x] **DQN** (`rl/dqn`): replay buffer and target network switchable, value runaway without a target.

### Deep learning

- [x] **Data augmentation** (`deep/augmentation`): shifts, rotations and noise on a small digit set.
- [x] **Head ablation** (`deep/head-ablation`): silence heads of the tiny GPT, patterns and removal curve.

### Generative

- [x] **VAE variants** (`generative/latent-control`): conditional VAE and VQ-VAE (beta-VAE already lives in the autoencoder lab).
- [x] **GAN variants** (`generative/gan-variants`): standard, WGAN-GP (exact gradient penalty) and conditional GAN on eight clusters (a digits GAN was left out: it needs convolutional generators to be convincing).
- [x] **Diffusion variants** (`generative/guidance`): classifier-free guidance and flow matching.
- [x] **Normalising flows** (`generative/flows`): RealNVP coupling layers on 2-D data, exact likelihood and inverse.

### Language and modern

- [x] **RAG toy** (`language/rag`): closed-book network against retrieval on invented facts: capacity, updates, typos.
- [x] **RLHF toy** (`modern/rlhf`): Bradley-Terry reward model, KL-regularised optimum, best-of-n and reward hacking.
- [x] **Mixture of experts** (`modern/moe`): sparse gating, load balance, specialisation.
- [x] **Graph neural networks** (`modern/gnn`): GCN on planted-partition graphs, over-smoothing with depth.
- [x] **Neural ODEs** (`modern/neural-ode`): Euler and RK4 solvers, backprop through the solver, a model that is only a ResNet when the solver is crude.

### Interpretability

- [x] **Saliency and integrated gradients** (`interpretability/attribution`) on the digit CNN, with completeness, deletion and a sanity check.
- [x] **Adversarial robustness** (`interpretability/robustness`): FGSM/PGD against an ordinary and an adversarially trained CNN.
- [x] **Probing** (`interpretability/probing`): linear probes with a control task on the digit MLP and the tiny GPT.
- [x] **Feature visualisation** (`interpretability/feature-vis`): activation maximisation for classes and feature maps, with blur and decay.

### Optional classical extras (only if time allows)

- [x] isotonic regression, GAMs, MARS (`classic-supervised/flexible-regression`).
- [x] constrained optimisation (`evaluation/constrained`); survival analysis (`classic-supervised/survival`).

### Phase 9: hardening (after the labs)

- [x] Bundle: lab pages and the 3D scene are lazy chunks; `tools/check-bundle.mjs` (part of `npm run check`) holds a gzip budget. `LazyMotion` not adopted (framer-motion is used in about ten components; the saving is small against the risk).
- [x] Accessibility audit: `apps/web/e2e/a11y.spec.ts` runs axe (WCAG 2.x A and AA) on every route in both themes (180 checks, all passing); text tokens raised to 4.5:1, heat cells pick their own ink, button names contain their visible text. Reduced motion is honoured in `index.css`.
- [ ] Visual regression screenshots; refresh `img/` and the README gallery (the existing images predate the 2026-10 labs; optional polish).
- [x] Refresh README, PLAN_V2, ARCHITECTURE, COMMIT_PLAN, memory notes (counts are 89 labs).
- [x] Final `npm run check` (green, includes the bundle budget) and the whole Playwright suite (344 passing; the two slow guidance rows were then given a longer wait and pass); the 180 axe checks pass too.

## Known caveats

- Heavy labs run their computation in a Web Worker through `apps/web/src/shared/compute` (`useCompute`, or `compute` for
  step-by-step results); results must be structured-cloneable, so closures and agents are stripped in `tasks.ts`.
  Guidance takes about 80 s per scenario even in the worker.

- `npm run check` runs about 900 web tests plus package tests (2,000+ in all); the heaviest lab tests have a 300 s timeout for loaded machines.
- Run the Playwright suite with few workers (`--workers=2`) on a loaded machine; slow labs are flagged `slow` in `classic-labs.spec.ts`.
- Never run `vitest -w` (watch mode); use `npx vitest run <folder>` from inside `apps/web` or a package.
