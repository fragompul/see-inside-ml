# Architecture

## Principles

1. **Engine ≠ UI.** the packages (`packages/core`, `packages/classic`, `packages/deep`) have no DOM or React dependency. It is pure, seeded and
   unit-tested, so numerical correctness is verified independently of the visuals.
2. **One folder per lab.** Each lab lives in `apps/web/src/labs/<track>/<lab>/` and is registered in
   `apps/web/src/app/registry.ts` (tracks, then labs with `track`, `path` and `page`). Routes, the
   Labs menu and the landing are generated from the registry. Adding a lab = one registry entry +
   one folder.
3. **Lazy by default.** Modules are loaded with `React.lazy`, so the landing stays light.
4. **Runs on the user's machine.** No backend. Heavy models (later) use pretrained weights shipped in
   the repo and run via ONNX Runtime Web / WebGPU.
5. **Colour carries meaning.** `--pos` (ember) positive, `--neg` (frost) negative, `--brand` (iris)
   interaction / backpropagation. Never use them decoratively inside visualizations.

## Layout

```
apps/web/src
├─ app/              shell: router (built from the registry), layout, Labs menu, landing, registry
├─ design-system/    tokens (index.css), theme store, primitives (Radix + Tailwind), motion presets
├─ shared/           components (charts, heat maps…), canvas helpers, playground worker client
│  └─ lab-kit/       generic ClassifierLab / RegressorLab, ParamSpec panels, experiments, StepPlayer
└─ labs/<track>/<lab>/   one folder per lab: page, spec/lessons, tests
packages/core        matrix, rng, optimizers, linalg, stats, metrics, estimator interfaces, tabular helpers, datasets
packages/classic     classical models (one engine file per model family) + fixtures/ from scikit-learn/statsmodels
packages/deep        neural engines (see below); bench/ has opt-in timings; fixtures/ the golden data
tools/               author-time Python (tiny GPT, fixture generators, dataset export)
```

### The lab kit

A classical model is one engine file in `@sim/classic` implementing `Classifier`, `Regressor`,
`Clusterer`… (from `@sim/core`), plus a small spec: a `create(params)` factory, declarative
`ParamSpec`s (slider, ladder, select, segmented, switch) and `experiments`. `ClassifierLab` /
`RegressorLab` supply the data plane, dataset picker, hand editing of points, train/test split,
metrics and the experiment bar; labs add `extras` panels, overlays and highlights.

An experiment is `{ setup, params, claim(run) }`. The page applies the setup when clicked and a unit
test evaluates the same `claim` through `runClassifierExperiment` / `runRegressorExperiment`, so what
the text promises is what was measured.

Validation: deterministic algorithms are checked against author-time scikit-learn / SciPy /
statsmodels fixtures committed as JSON; stochastic ones by property and analytic tests.

## Theming

Tokens are CSS variables in `apps/web/src/index.css`, switched by `data-theme` on `<html>`
(`light` = Parchment, `dark` = Obsidian). A blocking inline script in `index.html` resolves the
stored preference (`localStorage["nnp-theme"]`: `light | dark | system`) before first paint, so
there is no flash. `useThemeSync` follows OS changes in `system` mode. Canvas visualizations read
tokens via `getComputedStyle` and re-initialize when `useTheme().resolved` changes.

## Deep engines (`packages/deep`)

Framework-free TypeScript on `Float64Array`, no DOM. Layers:

| Layer    | Files                                                   | Role                                                                                                                   |
| -------- | ------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------- |
| Math     | `matrix.ts`, `rng.ts`                                   | row-major matrices; seeded RNG (mulberry32 + Box-Muller)                                                               |
| Network  | `activations.ts`, `losses.ts`, `dense.ts`, `network.ts` | MLP with manual backprop, 8 activations, CE / MSE / MAE / Huber, optional skip connections                             |
| Learning | `optimizers.ts`, `schedulers.ts`, `trainer.ts`          | SGD / Momentum / RMSprop / Adam, L1 / L2, constant / exponential / cosine LR, mini-batch epochs                        |
| Data     | `datasets.ts`, `features.ts`                            | 5 classification + 3 regression datasets, 12 engineered features                                                       |
| Product  | `session.ts`, `host.ts`                                 | `PlaygroundSession` (serialisable config → scene + snapshots) and `PlaygroundHost` (typed request / response protocol) |

Design notes:

- **One seed, independent streams.** Dataset, split, weight init and batch shuffling each draw from
  their own derived stream, so changing the architecture keeps the same data and vice-versa, and
  `reset()` replays the exact same run.
- **Config → what happens.** Hyper-parameters (optimizer, LR, schedule, batch size, regularisation)
  apply live; data or architecture changes rebuild. `update()` reports which one occurred.
- **Snapshots are cheap and side-effect free.** `forward(x, cache = false)` and `trace(x)` never touch
  the activations cached for backprop. A snapshot gives weights, the decision boundary / regression
  surface and every neuron's response over the input plane, which is what powers "hover a neuron to
  see what it detects".
- **Grid layout.** `GridField` is row-major, x grows along columns, y grows along rows (row 0 =
  bottom of the plane), sampled at cell centres.

### Worker protocol

`apps/web/src/shared/playground` runs a `PlaygroundHost` inside a module Web Worker; the UI talks
to it through `PlaygroundClient` (promise-based, id-matched, rejects on worker errors or disposal).
Requests: `init | update | reset | step | snapshot`. Responses: `scene` (data or architecture
changed), `progress` (stats + optional snapshot) or `error`. The host never throws; it is tested in
the engine package and the client against an in-process transport.

### Correctness

1. **Parity with the NumPy reference:** 24 golden cases (every activation, optimiser, loss, skip
   connections, L1 / L2, deep nets) with explicit data and initial weights; forward output,
   first-step gradients, loss trajectory and final weights must match to 1e-8.
2. **Gradient checking:** analytic gradients vs central finite differences for every activation,
   loss and skip configuration (independent of the reference).
3. **Behavioural tests:** learning actually happens (circles ≥ 90 % test accuracy, plane → MSE < 0.01),
   determinism, live vs rebuild semantics, input validation.

Regenerate fixtures with `python reference/python/generate_fixtures.py` (only needed while the
Python reference exists; the JSON is committed).

### Performance

Median per call on the author's machine (`npm run bench -w @sim/deep`):

| Operation                                        | Time   |
| ------------------------------------------------ | ------ |
| Epoch, 300 samples, batch 20, layers [4, 3]      | 0.7 ms |
| Epoch, 300 samples, layers [8, 8, 8], 5 features | 1.5 ms |
| Epoch, 1000 samples, 6 × 16 neurons              | 21 ms  |
| Snapshot: decision boundary 50×50                | 3.5 ms |
| Snapshot: boundary + every neuron map 50×50      | 3.9 ms |
| Snapshot: boundary + every neuron map 100×100    | 18 ms  |

A small network trains at over 1 000 epochs per second, so the UI can run at 60 fps with many
epochs per frame. In the browser, 200 epochs through the Web Worker (including message passing)
took about 80 ms.

## The MLP module (`apps/web/src/labs/<track>/mlp`)

| Piece                                            | Role                                                                                                                                                                                                                                                                                     |
| ------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `store.ts`                                       | Zustand store + a **single sequential pump** that owns the worker: config patches, resets, inspections and training steps never overlap; slider drags coalesce into one update; the run loop yields a frame between steps. Rejected settings roll back (`notice`), never crash the page. |
| `components/NetworkGraph.tsx`                    | SVG edges under HTML neuron canvases. Edge colour = sign, width = magnitude; animated flow while training. Hover node → heat-map, bias, strongest weights, formula (KaTeX, lazy). Hover edge → weight.                                                                                   |
| `components/DecisionPlane.tsx`                   | Two canvases: heavy (surface, grid, points) and light (crosshair), so hover never repaints the surface.                                                                                                                                                                                  |
| `components/InspectPanel.tsx` + `inspectView.ts` | The step inspector. `inspectView.ts` holds the pure view-model (what each edge/neuron shows per stage), unit-tested against the engine.                                                                                                                                                  |
| `presets.ts`                                     | Guided experiments. Their lessons are asserted in `presets.test.ts`.                                                                                                                                                                                                                     |
| `url.ts`                                         | Shareable setup: only fields that differ from defaults, base64url JSON, **untrusted-input safe** (unknown keys ignored, invalid → defaults).                                                                                                                                             |

Inspector semantics: forward shows each edge's contribution `w·a` and each neuron's activation;
loss highlights the target; backward shows `∂L/∂w = a·δ` (violet pulses run right → left) and each
neuron's δ; update shows `Δw = −η·∂L/∂w`. It is read-only (`session.inspect` never changes weights
or optimiser state) and is closed automatically as soon as training would change the weights.

## The CNN module

`packages/deep/src/cnn` holds a from-scratch CNN: `conv(1→f1) → ReLU → maxpool 2 → conv(f1→f2)
→ ReLU → flatten → dense(10) → softmax`, with a hand-written backward pass (`layers.ts`,
`network.ts`). Digits are drawn procedurally from stroke skeletons with jitter, tilt and pixel
noise (`digits.ts`), so there is no dataset to download and everything is seeded and
reproducible. `CnnSession` wraps network, data and optimizer; `trainBatches(n)` advances a few
mini-batches at a time so the UI can animate the weights, and `update()` distinguishes live
hyper-parameters from changes that rebuild the network or data.

Correctness: convolution is compared with a naive reference, every layer and the whole network
pass finite-difference gradient checks (3×3 and 5×5), training is deterministic for a seed, and
batch-wise training equals epoch-wise training.

The web module (`apps/web/src/labs/<track>/cnn`) keeps one module-level session behind a zustand store;
a `requestAnimationFrame` loop trains `speed` batches per frame and recomputes the trace of the
image on the pad. The costly occlusion map is refreshed at most five times a second. Drawing
helpers (`draw.ts`: brush, gap-free strokes, centre/scale normalisation) and the explainer's
kernels (`kernels.ts`) are pure and unit-tested. The learning curves reuse the MLP's `LossChart`.

## The foundations module

`packages/deep/src/foundations` holds one two-input neuron with four activations, four tiny
datasets (AND, OR, XOR and two seeded clouds) and the sigmoid + cross-entropy loss with its
analytic gradient (`dL/dz = a − y`), checked against finite differences. `bestLine` finds, without
any learning, the best accuracy a single line can reach by trying 360 directions and every cut of the
projected points: 100% on AND, OR and the clouds, 75% on XOR. `Descent` turns the two weights into
the position of a ball on the loss surface (the bias is held fixed so it is a 2-D landscape) and
rolls it with the app's own `Optimizer`; the test suite pins the behaviours the lab teaches (a
0.01 learning rate never gets below loss 0.3 in 400 steps, 0.5 does in under 40; momentum under 30
steps against more than 80 for plain SGD at the same rate; XOR's landscape has no valley, its
minimum is ln 2).

The web module draws the neuron's plane and the landscape on canvases (the landscape is a small
image scaled up, log-coloured so the valley shows), reuses the engine's `activate`/`derivative` for
the activation gallery, and links to the MLP playground, which continues where the module stops.

## The reinforcement-learning module

`packages/deep/src/rl` is a tabular agent in a gridworld. `grid.ts` parses layouts (rows of
`. # G X S`), implements the dynamics (walls, edges, pits, goal, a slippery floor with a known
transition distribution) and ships the presets (open room, cliff, maze, pit by the gap).
`planning.ts` runs **value iteration** on the exact dynamics, giving the optimal values and the set
of optimal actions per cell: the yardstick the agent is measured against (`agreement` = share of
reachable cells where its best-looking move is optimal; `valueError` = mean distance to V*).
`RlSession` implements Q-learning, SARSA and Expected SARSA with ε-greedy exploration and a
per-episode ε decay, records every episode (return, steps, outcome), and exposes the greedy route.
Tests include the Bellman optimality equation, a hand-solved corridor, convergence of all three
algorithms to the shortest route in an open room, and the textbook cliff: Q-learning learns the
13-step route along the edge while SARSA settles on a longer one two rows away, with better returns
during training.

The web store publishes snapshots (a copy of Q, the visit counts, the optimal values, the greedy
path) at most every 60 ms. Painting a cell rebuilds the world through the session's validated
`update`, so illegal layouts (no goal, two starts) are refused with a message and the previous
layout is kept. The grid view is one SVG that draws values, the policy, the four action values per
cell or the visit counts.

## The embeddings module

`packages/deep/src/nlp` trains word2vec on a procedural toy world. `corpus.ts` defines the
vocabulary by family and a handful of sentence templates chosen so the corpus holds real structure:
royal and everyday gender pairs differ only by the pronoun (he/his, she/her) and by the nouns of
their world (crown and kingdom against bag and market), and every capital appears in sentences with
its own country. `word2vec.ts` is skip-gram with negative sampling and a hand-written update (input
and output tables, noise distribution ∝ count^0.75); because the summed-gradient update is exactly
`−lr · ∇(Σ loss)`, the test compares it with finite differences. `NlpSession` builds the (centre,
context) pairs for a window, trains in mini-batches, and exposes nearest neighbours, analogies
(`b − a + c`, excluding the inputs), a 2-D PCA of the content words (reusing the transformer
module's PCA), and two evaluation numbers: analogy accuracy over 54 questions and "neighbour
purity", the share of each word's three nearest neighbours that belong to its own family. A loss
above 25 marks the run as diverged and stops it (a fresh model starts near 4).

The web map is an SVG with keyboard-accessible points; the analogy view reuses it with arrows. The
store republishes the analyses every ~120 ms while training. The measured facts behind the
experiments (6000 steps, default settings): same-family neighbours > 90% after 2000 steps against
< 60% untrained; `man : king :: woman : ?` answers _queen_; country → capital analogies ≥ 80%
correct; a window of 1 stays under 35% analogy accuracy against > 60% for a window of 4; 2
dimensions stay under 30% against > 60% for 16; and a learning rate of 0.5 diverges.

## The diffusion module

`packages/deep/src/diffusion` is a DDPM on 2-D data. `schedule.ts` builds linear or cosine noise
schedules (arrays indexed by time 0…T, with the closed-form `x_t = √ᾱ x₀ + √(1−ᾱ) ε`) and the
sinusoidal time features. `network.ts` is the denoiser: a swish MLP over `[x, y, time features]`
trained on the mean squared error of its noise prediction, with a hand-written backward pass.
`sampler.ts` is pure in the predictor: DDPM (all T steps, posterior variance) and deterministic
DDIM (any number of evenly spaced steps, with x₀ clipping), optionally recording the path of every
point. Because the sampler takes a predictor function, it is tested against the exact optimal
denoiser for Gaussian data, `ε*(x, t) = √(1−ᾱ) x / (ᾱσ² + 1−ᾱ)`: it reproduces the target spread
up to the discretisation error of large steps (a wrong sign or coefficient would blow up or
collapse). `DiffusionSession` mirrors the other sessions and also exposes forward noising at any
time, fixed-noise sampling with trajectories, and the score field `−ε̂/√(1−ᾱ)`.

The web module renders everything on one reusable `PlaneCanvas` (clouds, arrows, trails). The store
republishes cheap views every ~90 ms while training and the costly sampling pass every ~400 ms.
The generation player is keyed on the length of the sampling path so it restarts when the sampler
or step count changes. Experiments carry `check` functions run by `lessons.test.ts`; the measured
facts behind them: a single DDIM step leaves >8× the distance of 20 steps, 10 DDIM steps are within
1.8× of all 50, and the cosine schedule keeps >3× more signal at the midpoint than the linear one.

## The GAN module

`packages/deep/src/gan` implements a GAN on 2-D data from the shared `DenseLayer`s. The
discriminator is trained on a concatenated batch of real points (label 1) and generated points
(label 0) with the usual cross-entropy. The generator is trained by back-propagating either
`−log D(G(z))` (non-saturating) or `log(1 − D(G(z)))` (minimax) through the discriminator: the
dense layers already return per-sample input gradients, so the generator's gradient is the
discriminator's `dL/dx` pushed into `Generator.backward`. Finite-difference tests cover both
networks and both losses, and another checks that minimax gives a ~100× smaller gradient when the
discriminator is certain, while both coincide at D = ½.

Quality is measured, not eyeballed: the sliced Wasserstein distance (fixed random projections, so
values are comparable across runs), the share of generated points near real data, and, for the
eight-Gaussians target, how many clusters receive a fair share of the samples. These feed the
charts and the experiments. `GanSession` records statistics every 10 iterations, keeps a fixed set
of noise vectors so the displayed cloud moves smoothly, and exposes the discriminator over a grid
and the generator's warped input grid.

The web module draws the plane on one canvas (the discriminator map is a tiny image scaled
underneath). Experiments carry a `check` function run by `lessons.test.ts`; the measured results
behind them: a fair duel finds 8/8 clusters, a strong discriminator (3 steps, 5× learning rate)
collapses to ≤ 4 clusters with > 3× the distance, and a very slow one leaves < 30% of the points
on the data. Shared line charts live in `src/shared/components/MiniChart.tsx`.

## The autoencoder module

`packages/deep/src/autoencoder` builds an autoencoder out of the shared `DenseLayer`s:
`x → hidden (ReLU) → code → hidden (ReLU) → pixels (sigmoid)`. The variational model adds a
log-variance head, samples `z = μ + σ·ε` when noise is supplied and adds `β · KL(q ‖ N(0, I))`.
Sigmoid and cross-entropy fuse into `p − x` at the output; the backward pass is hand-written,
including the reparameterisation gradients. Fixing `ε` makes the whole loss deterministic, which is
what lets the tests compare every parameter's gradient with finite differences for both models.
`AutoencoderSession` mirrors the CNN session (live vs rebuild updates, batch-wise training, history
of reconstruction/KL/total losses) and adds denoising corruption, mean codes of whole sets, batch
decoding and sampling from the prior.

The web module (`apps/web/src/labs/<track>/autoencoder`) keeps one session behind a zustand store. While
training, the store republishes its many views (scatter, decoder map, interpolation strip,
imagined images) at most every 100 ms: redrawing them every frame cost ~5× the training speed. The
latent map is a single canvas; the decoder's pictures are composed once into a tiny image and drawn
scaled underneath the dots (`latent.ts` holds the pure geometry). Experiments carry a `check`
function that `lessons.test.ts` runs, so each claim (bottleneck, denoising, VAE realism, β
collapse) is measured, not asserted by hand. The drawing helpers, pixel canvas, confusion matrix
and experiment bar are shared with the other labs under `src/shared`.

## The recurrent module

`packages/deep/src/rnn` holds one single-layer recurrent classifier (`recurrent.ts`) with three
cells: a vanilla tanh RNN, an LSTM (input, forget, candidate, output gates) and a GRU (reset,
update, candidate). Tokens are integers, so the input product is a row lookup. After the last
token the final hidden state feeds a softmax layer (many-to-one). Backpropagation through time is
written by hand; besides parameter gradients it returns `‖dL/dh_t‖` for every step, which the UI
plots to show vanishing gradients. `RnnSession` mirrors the CNN session (live vs rebuild updates,
`trainBatches`, history, confusion matrix, traces) and adds global-norm gradient clipping, an
accuracy-by-length probe and `gradientProfile`, the mean learning signal per step.

Three procedural tasks (`tasks.ts`): remember the first symbol, the symbol after a marker, parity.

Correctness: finite-difference checks of every parameter for all three cells (several sequence
lengths, non-zero biases), linearity of the gradient accumulation, scale-independent gradient
norms, GRU/LSTM state equations, determinism, batch-wise == epoch-wise training, and the claims
the UI makes (the learning signal at step 1 of a 30-step sequence is >100× larger in an LSTM than
in an RNN; a plain RNN fails the marker task where GRU and LSTM succeed) are asserted by tests.

The web module (`apps/web/src/labs/<track>/rnn`) keeps one session behind a zustand store with a
`requestAnimationFrame` loop like the CNN lab. The unrolled view lays heat-maps out time ×
neuron (`view.ts` transposes the time-major traces). The confusion matrix is shared with the CNN
lab (`shared/components`).

## The Transformer module

A GPT-2-style language model small enough to open up completely: **2 layers, 4 heads, width 64,
136k parameters, vocabulary 512**, trained on synthetic mini-stories. Everything runs in the
browser on the main thread (a forward pass over ≤ 48 tokens takes a few milliseconds), so hovering
and typing feel instant.

**Authoring pipeline** (`tools/transformer`, Python + NumPy only, author-time): a grammar-based
story generator, a from-scratch byte-level BPE tokenizer, a GPT with a hand-written backward pass
(finite-difference checked), a CPU trainer and an exporter. Weights (0.5 MB) and the tokenizer are
committed in `packages/deep/models/tiny-gpt`; see `tools/transformer/README.md`.

**Engine** (`packages/deep/src/transformer`):

| File                      | Role                                                                                                                                                                                                    |
| ------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `tokenizer.ts`, `utf8.ts` | Byte-level BPE, a faithful port of the Python one, with merge-by-merge replay and a hand-written UTF-8 codec (lone bytes of multi-byte characters behave like Python's `errors="replace"`).             |
| `model.ts`                | Forward pass that returns a full `ForwardTrace`: embeddings, per-head Q/K/V, scores, causal attention probabilities, residual stream after every sub-layer, MLP pre/post GELU, final LayerNorm, logits. |
| `sampling.ts`             | Temperature → top-k → top-p → renormalise, every stage inspectable (`Candidate` carries prob, cumulative mass, kept, adjusted), plus entropy.                                                           |
| `generate.ts`             | Decoding loop (`generateNext`, `generate`) and `createTransformer` which checks tokenizer and weights belong together.                                                                                  |
| `embeddings.ts`           | PCA by power iteration and cosine neighbours for the embedding map.                                                                                                                                     |

**Correctness.** The float64 NumPy forward pass on the exact float32 weights is the oracle: every
tensor above must match to 1e-8 on three prompts, and tokenisation must equal Python's on eight
tricky strings (unseen words, digits, non-ASCII, `<|endoftext|>`, whitespace). Property tests cover
causality (the future never changes the past), attention rows summing to 1, the residual-stream
identity, sampling statistics, determinism and input validation. Behavioural tests check that the
trained model really learned the grammar of its stories.

**UI** (`apps/web/src/labs/<track>/transformer`): five stages linked by one shared focus (click a token
anywhere and every panel follows it). Tokens + live BPE demo → embeddings (per-token vectors, the
whole input matrix, and a PCA map of the vocabulary coloured by word type) → attention (all heads
as small multiples, a full matrix with hover maths for `q·k/√d`) → residual stream and MLP (GELU
curve with the token's neurons on it) → next-token distribution with live temperature / top-k /
top-p, click-to-choose and auto-generation. `store.ts` keeps generation deterministic: each new
token draws from its own stream derived from `(seed, position)`.

### The 3-D view (`labs/deep/transformer/three`)

React Three Fiber, lazy-loaded (a separate chunk, fetched shortly before the panel scrolls into view;
a WebGL check shows a 2-D fallback message when unavailable).

- `layout.ts` is **pure geometry** (no Three.js) and unit-tested: slabs for every stage of the trace,
  the guided-tour stages, cell ↔ instance index maps, head-matrix placement, output bars.
- `Scene.tsx` draws each slab as **one `InstancedMesh`** (up to 12k cells, a single draw call),
  head matrices as instanced lower triangles, bars as boxes, and eases the camera between stage poses
  (user input cancels the fly-to; narrow screens pull the camera back).
- Pitfalls solved on the way, worth remembering: three.js compiles an _opaque_ shader variant for
  `transparent={false}` materials and does **not** recompile when the flag flips, so every material
  is always transparent and only its opacity changes (stage dimming); drei's `Html` mounts one React
  root per label and floods the console with warnings in development, so text is drawn as
  canvas-texture sprites (`Label.tsx`); a second copy of `three` (pinning a different version in the
  app than drei/fiber use) breaks the types.
- Known dev-only noise: `THREE.Clock` deprecation warning from @react-three/fiber (upstream).

## Deeper looks at existing labs

**CNN: attacks and attention** (`packages/deep/src/cnn/attacks.ts`). `CnnNetwork.inputGradient`
backpropagates to the pixels without disturbing the training state. On top of it: `attack` (FGSM, PGD,
targeted or not, with an L-infinity budget and the [0, 1] pixel range enforced), `robustAccuracy`,
`gradCam` and `evidenceMap`. The network's last layer is a plain dense layer over the feature maps, so
the exact per-position evidence for a class is weight x activation and sums to the logit; Grad-CAM
averages the weights over position first, which can blur or erase the map for this head. The lab shows
both and says why.

**MLP: loss landscape, histograms, hand-edited points** (`packages/deep/src/landscape.ts`, `session.ts`).
The session keeps a thinned trajectory of weight vectors. `weightSpaceLandscape` evaluates the training
loss on a plane through the current weights spanned either by the two leading principal directions of
that trajectory (Gram-matrix PCA) or by two filter-normalised random directions, and projects the path
onto the plane. `layerHistograms` bins weights and the latest gradients per layer. Both are served by a
read-only `analyse` request of the worker protocol, so they never change training. `editPoints`
adds, moves or erases training points of classification datasets without touching the weights, the
test set or the epoch counter; `restoreData` regenerates the original points.

## Testing

- **Unit (Vitest):** engine (parity, gradient checks, datasets, session, protocol), registry integrity, theme store + toggle, worker client.
- **E2E (Playwright):** landing, theme persistence across reload, routing, footer links, no horizontal overflow on mobile.

## Known follow-ups

- Initial JS bundle is ~165 kB gzip (React, Router, Radix, Framer Motion). Revisit with
  `LazyMotion` and manual chunks during the performance pass (Phase 3).

## Expansion modules (classic, evaluation, RL, modern)

Beyond the original deep-learning labs, the atlas holds classical-ML engines in `packages/classic/src`
(regression, trees, ensembles, SVM, clustering, anomaly detection, SOM, ICA, NMF, HMM, filters, MCMC,
conjugate Bayes, Markov chains, time series, recommenders, evaluation, importance, tuning, explain),
optimisers in `packages/core/src` (`descent`, `metaheuristics`) and further deep engines in
`packages/deep/src` (`rl/{bandits,policy,prediction,mcts,dyna}`, `rnn/charlm`, `toolbox`, `compress`,
`transformer/positional`).

Every lab follows one pattern: `<lab>Core.ts` turns a scenario into a run, `<lab>Spec.ts` lists the
parameters and the experiments (each with a `claim(run)` function), `<Lab>Page.tsx` renders it with the lab
kit, and `<lab>.test.ts` executes every claim, so each sentence shown in the UI is checked by the suite.
Stochastic methods are validated by property and closed-form tests; deterministic ones by parity
fixtures generated with scikit-learn/statsmodels (`tools/fixtures/gen_classic.py`). Registry entries must be
grouped by track (`registry.test.ts`); `tools/registry_insert.py` inserts one in the right place.
