import { expect, test } from '@playwright/test';

/**
 * Every lab built on the lab kit shares one structure: a header, the experiment bar, the fit with
 * its metrics, optional extra panels and the settings. This suite checks all of them the same way;
 * labs with their own interaction also have a dedicated spec.
 */
const LABS: {
  path: string;
  heading: RegExp;
  panels: string[];
  metric: string;
  /** The metric appears only after several seconds of training in the browser. */
  slow?: boolean;
}[] = [
  {
    path: '/classic/multiple-regression',
    heading: /Multiple regression/,
    panels: ['The data and the fit', 'Diagnostics: which coefficients can be trusted?', 'Settings'],
    metric: 'train-r2',
  },
  {
    path: '/classic/logistic-regression',
    heading: /Logistic regression/,
    panels: [
      'The data and the decision regions',
      'Optimisation: Newton vs gradient descent',
      'Reading the weights',
      'Settings',
    ],
    metric: 'train-accuracy',
  },
  {
    path: '/classic/knn',
    heading: /k-nearest neighbours/,
    panels: ['Choosing k', 'The curse of dimensionality', 'Settings'],
    metric: 'train-accuracy',
  },
  {
    path: '/classic/gaussian-classifiers',
    heading: /Naive Bayes, LDA & QDA/,
    panels: ['The data and the decision regions', 'How it works', 'Settings'],
    metric: 'train-accuracy',
  },
  {
    path: '/classic/polynomial-regularization',
    heading: /Polynomials & regularisation/,
    panels: [
      'The data and the fitted curve',
      'Complexity: the U-shaped test error',
      'Coefficient paths',
      'Settings',
    ],
    metric: 'train-r2',
  },
  {
    path: '/classic/robust-glm',
    heading: /Robust regression & GLMs/,
    panels: ['The data and the fitted curve', 'Four lines, one dataset', 'Settings'],
    metric: 'train-r2',
  },
  {
    path: '/classic/flexible-regression',
    heading: /Flexible regression/,
    panels: [
      'The data and the fitted curve',
      'Two inputs: additive models and interactions',
      'Settings',
    ],
    metric: 'train-r2',
  },
  {
    path: '/classic/survival',
    heading: /Survival analysis/,
    panels: ['Survival curves', 'Medians and the naive estimate', 'Settings'],
    metric: 'censored',
  },
  {
    path: '/classic/decision-tree',
    heading: /Decision trees/,
    panels: ['The tree', 'Growing it level by level', 'Pruning', 'Which features matter?'],
    metric: 'train-accuracy',
  },
  {
    path: '/classic/random-forest',
    heading: /Random forests/,
    panels: ['Trees disagree, the forest averages', 'Out-of-bag error', 'Which features matter?'],
    metric: 'train-accuracy',
  },
  {
    path: '/classic/adaboost',
    heading: /AdaBoost/,
    panels: ['Round by round', 'Settings'],
    metric: 'train-accuracy',
  },
  {
    path: '/classic/gradient-boosting',
    heading: /Gradient boosting & XGBoost/,
    panels: ['The data and the fitted curve', 'Growing the ensemble', 'Settings'],
    metric: 'train-r2',
  },
  {
    path: '/classic/svm',
    heading: /Support vector machines/,
    panels: ['Margin and support vectors', 'The penalty C', 'The kernel width γ'],
    metric: 'train-accuracy',
  },
  {
    path: '/classic/gaussian-process',
    heading: /Gaussian processes/,
    panels: ['Hyperparameters and evidence', 'What the kernel believes', 'Settings'],
    metric: 'train-r2',
  },
  {
    path: '/classic/bayesian-regression',
    heading: /Bayesian regression/,
    panels: ['What the model learned about itself', 'Settings'],
    metric: 'train-r2',
  },
  {
    path: '/classic/showdown',
    heading: /Model showdown/,
    panels: ['The arena', 'Scoreboard', 'Data settings'],
    metric: 'winner',
  },
  {
    path: '/classic/ensembles',
    heading: /Voting & stacking/,
    panels: ['Members versus the ensemble', 'Settings'],
    metric: 'train-accuracy',
  },
  {
    path: '/classic/kmeans',
    heading: /k-means/,
    panels: ['Watch it converge', 'Choosing k', 'The initialisation lottery'],
    metric: 'clusters',
  },
  {
    path: '/classic/hierarchical',
    heading: /Hierarchical clustering/,
    panels: ['The dendrogram', 'Four linkages on the same data', 'Settings'],
    metric: 'clusters',
  },
  {
    path: '/classic/dbscan',
    heading: /DBSCAN & OPTICS/,
    panels: ['Choosing ε: the k-distance plot', 'OPTICS: every ε at once', 'Settings'],
    metric: 'clusters',
  },
  {
    path: '/classic/meanshift',
    heading: /Mean shift/,
    panels: ['Choosing the bandwidth', 'Settings'],
    metric: 'clusters',
  },
  {
    path: '/classic/gmm',
    heading: /Gaussian mixtures & EM/,
    panels: [
      'Expectation–maximisation, step by step',
      'Choosing the number of components',
      'Settings',
    ],
    metric: 'clusters',
  },
  {
    path: '/classic/spectral',
    heading: /Spectral clustering/,
    panels: ['The spectrum', 'The spectral embedding', 'Settings'],
    metric: 'clusters',
  },
  {
    path: '/classic/pca',
    heading: /PCA & kernel PCA/,
    panels: [
      'How much each direction carries',
      'What the directions are',
      'Compression and reconstruction',
    ],
    metric: 'trust',
  },
  {
    path: '/classic/anomaly',
    heading: /Anomaly detection/,
    panels: ['How good is the ranking?', 'The scores', 'Settings'],
    metric: 'flagged',
  },
  {
    path: '/classic/som',
    heading: /Self-organising map/,
    panels: ['Reading the map', 'Component planes', 'Settings'],
    metric: 'trust',
  },
  {
    path: '/classic/ica',
    heading: /Independent component analysis/,
    panels: ['Separating the signals', 'The mixture in the plane', 'Settings'],
    metric: 'recovery',
  },
  {
    path: '/classic/nmf',
    heading: /Non-negative matrix factorisation/,
    panels: ['The parts it found', 'Rebuilding a digit', 'Convergence'],
    metric: 'nmf-error',
  },
  {
    path: '/classic/manifold',
    heading: /Manifold learning/,
    panels: ['About this method', 'All the methods side by side', 'Settings'],
    metric: 'trust',
  },
  {
    path: '/classic/hmm',
    heading: /Hidden Markov models/,
    panels: ['Reading the rolls', 'Dice and switches', 'Learning the model'],
    metric: 'viterbi',
  },
  {
    path: '/classic/filters',
    heading: /Kalman & particle filters/,
    panels: ['Following the target', 'How it works', 'Settings'],
    metric: 'rmse-kalman',
  },
  {
    path: '/classic/mcmc',
    heading: /MCMC sampling/,
    panels: ['The walk', 'Diagnostics', 'Settings'],
    metric: 'acceptance',
  },
  {
    path: '/classic/bayes',
    heading: /Bayesian inference/,
    panels: ['Updating a belief', 'What happens next?', 'Settings'],
    metric: 'posterior-mean',
  },
  {
    path: '/classic/pagerank',
    heading: /Markov chains & PageRank/,
    panels: ['Three ways to the answer', 'Convergence', 'Settings'],
    metric: 'top-page',
  },
  {
    path: '/classic/forecasting',
    heading: /Time-series forecasting/,
    panels: ['The forecast', 'Against benchmarks', 'Residuals'],
    metric: 'mape',
  },
  {
    path: '/classic/association',
    heading: /Association rules/,
    panels: ['The rules', 'Support and confidence', 'What is in the baskets'],
    metric: 'rules',
  },
  {
    path: '/classic/recommender',
    heading: /Recommender systems/,
    panels: ['The tables', 'Films on a map', 'Recommendations'],
    metric: 'test-rmse',
  },
  {
    path: '/eval/bias-variance',
    heading: /Bias, variance/,
    panels: ['Many fits', 'Where the error comes from', 'More data'],
    metric: 'test-error',
  },
  {
    path: '/eval/validation',
    heading: /Validation/,
    panels: ['The experiment', 'How it works', 'Settings'],
    metric: 'truth',
  },
  {
    path: '/eval/metrics',
    heading: /Metrics & thresholds/,
    panels: ['Confusion', 'ROC and precision-recall', 'Calibration'],
    metric: 'accuracy',
  },
  {
    path: '/eval/tuning',
    heading: /Hyperparameter tuning/,
    panels: ['The search', 'Which method wins?', 'Progress'],
    metric: 'regret',
  },
  {
    path: '/eval/optimizers',
    heading: /The optimiser race/,
    panels: ['The race', 'Scoreboard', 'Value per step'],
    metric: 'best',
  },
  {
    path: '/eval/feature-selection',
    heading: /Feature selection/,
    panels: ['The ranking', 'How many to keep', 'Settings'],
    metric: 'test-r2',
  },
  {
    path: '/eval/metaheuristics',
    heading: /Metaheuristics/,
    panels: ['The search', 'Which method wins?', 'Progress'],
    metric: 'best',
  },
  {
    path: '/eval/constrained',
    heading: /Constrained optimisation/,
    panels: ['The problem', 'Convergence', 'Settings'],
    metric: 'distance',
  },
  {
    path: '/rl/bandits',
    heading: /Multi-armed bandits/,
    panels: ['Regret', 'Scoreboard', 'One run in detail'],
    metric: 'best-regret',
  },
  {
    path: '/rl/policy-gradient',
    heading: /Policy gradients/,
    panels: ['What the agent learned', 'Learning curves', 'Settings'],
    metric: 'final-return',
  },
  {
    path: '/rnn/writer',
    heading: /A network that writes/,
    panels: ['Training', 'What it writes', 'Settings'],
    metric: 'loss',
  },
  {
    path: '/deep/toolbox',
    heading: /The training toolbox/,
    panels: ['Learning curves', 'What it learned', 'Learning-rate finder'],
    metric: 'test-accuracy',
  },
  {
    path: '/interpret/explain',
    heading: /Explaining a prediction/,
    panels: ['One prediction', 'Overall', 'Settings'],
    metric: 'prediction',
  },
  {
    path: '/modern/compression',
    heading: /Making models small/,
    panels: ['The result', 'How far can it go?', 'Settings'],
    metric: 'accuracy',
  },
  {
    path: '/modern/lora',
    heading: /Fine-tuning with LoRA/,
    panels: ['Adapting', 'Cost against accuracy', 'Settings'],
    metric: 'after',
  },
  {
    path: '/rl/td-learning',
    heading: /Monte Carlo and temporal-difference learning/,
    panels: ['Error while learning', 'What was learned', 'Settings'],
    metric: 'best-error',
  },
  {
    path: '/rl/mcts',
    heading: /Monte Carlo tree search/,
    panels: ['The search', 'Playing real games', 'Settings'],
    metric: 'reliability',
  },
  {
    path: '/rl/dyna',
    heading: /Dyna: planning with a model/,
    panels: ['Learning speed', 'The learned route', 'Settings'],
    metric: 'real-steps',
  },
  {
    path: '/language/positional',
    heading: /Position encodings/,
    panels: ['Position by position', 'Score by distance', 'Settings'],
    metric: 'defined-share',
  },
  {
    path: '/language/kv-cache',
    heading: /The KV cache and the cost of attention/,
    panels: ['Work done', 'Memory kept', 'Settings'],
    metric: 'speedup',
  },
  {
    path: '/deep/conv-arithmetic',
    heading: /Convolution arithmetic/,
    panels: ['The stack', 'Settings'],
    metric: 'field',
  },
  {
    path: '/language/tokenizers',
    heading: /Tokenizers compared/,
    panels: ['The text, cut up', 'Across texts', 'Settings'],
    metric: 'vocab',
  },
  {
    path: '/language/ngram-vs-neural',
    heading: /N-grams versus a neural language model/,
    panels: ['Context length', 'Settings'],
    metric: 'ngram-ppl',
  },
  {
    path: '/modern/scaling',
    heading: /Scaling laws/,
    panels: ['Loss by size and data', 'The two laws', 'Spending a budget', 'Settings'],
    metric: 'data-exponent',
    slow: true,
  },
  {
    path: '/rl/actor-critic',
    heading: /Actor-critic and advantage estimation/,
    panels: ['The race', 'Noise against bias', 'Settings'],
    metric: 'final-a2c',
    slow: true,
  },
  {
    path: '/rl/dqn',
    heading: /Deep Q-learning/,
    panels: ['Learning curves', 'Four ways to train', 'Settings'],
    metric: 'solved',
  },
  {
    path: '/deep/augmentation',
    heading: /Data augmentation/,
    panels: ['Does it help?', 'The copies', 'Settings'],
    metric: 'clean-accuracy',
    slow: true,
  },
  {
    path: '/deep/head-ablation',
    heading: /Attention head ablation/,
    panels: ['The eight heads', 'Losing heads one by one', 'Settings'],
    metric: 'baseline-loss',
  },
  {
    path: '/generative/latent-control',
    heading: /Steering a VAE/,
    panels: ['What it draws', 'Settings'],
    metric: 'control',
    slow: true,
  },
  {
    path: '/generative/gan-variants',
    heading: /GAN recipes compared/,
    panels: ['The points', 'During training', 'Settings'],
    metric: 'modes',
    slow: true,
  },
  {
    path: '/generative/flows',
    heading: /Normalising flows/,
    panels: ['Density and samples', 'From data to Gaussian', 'Settings'],
    metric: 'nll',
  },
  {
    path: '/generative/guidance',
    heading: /Guidance and flow matching/,
    panels: ['Drawing a class', 'Across guidance weights', 'Settings'],
    metric: 'accuracy',
    slow: true,
  },
  {
    path: '/interpret/robustness',
    heading: /Adversarial robustness/,
    panels: ['Accuracy under attack', 'What the attack does', 'Settings'],
    metric: 'clean',
    slow: true,
  },
  {
    path: '/interpret/probing',
    heading: /Probing the layers/,
    panels: ['What each layer knows', 'The control task', 'Settings'],
    metric: 'first-score',
    slow: true,
  },
  {
    path: '/interpret/feature-vis',
    heading: /Feature visualisation/,
    panels: ['What each digit looks like to the network', 'Settings'],
    metric: 'score',
    slow: true,
  },
  {
    path: '/modern/moe',
    heading: /Mixture of experts/,
    panels: ['Who handles what', 'Five designs compared', 'Settings'],
    metric: 'mse',
    slow: true,
  },
  {
    path: '/modern/gnn',
    heading: /Graph neural networks/,
    panels: ['The graph', 'Depth', 'Settings'],
    metric: 'accuracy',
    slow: true,
  },
  {
    path: '/modern/neural-ode',
    heading: /Neural ODEs/,
    panels: ['The flow', 'Training', 'Settings'],
    metric: 'own',
    slow: true,
  },
  {
    path: '/language/rag',
    heading: /Retrieval-augmented generation/,
    panels: ['Two ways to know a fact', 'Some questions', 'Settings'],
    metric: 'closed-book',
    slow: true,
  },
  {
    path: '/modern/rlhf',
    heading: /Learning from human feedback/,
    panels: ['Pushing on the reward model', 'Best of n', 'Settings'],
    metric: 'accuracy',
    slow: true,
  },
  {
    path: '/interpret/attribution',
    heading: /Pixel attribution/,
    panels: ['Four ways to attribute', 'Do the maps tell the truth?', 'Settings'],
    metric: 'accuracy',
    slow: true,
  },
];

for (const lab of LABS) {
  test.describe(lab.path, () => {
    test.beforeEach(async ({ page }) => {
      page.on('pageerror', (error) => {
        throw error;
      });
      await page.goto(`/#${lab.path}`);
      await expect(page.getByRole('heading', { name: lab.heading, level: 1 })).toBeVisible();
    });

    test('shows its panels and metrics', async ({ page }) => {
      test.setTimeout(lab.slow ? 240_000 : 60_000);
      // some panels only appear once the computation has finished
      await expect(page.getByTestId(lab.metric)).toContainText(/\d/, {
        timeout: lab.slow ? 200_000 : undefined,
      });
      for (const title of lab.panels) {
        await expect(page.getByRole('heading', { name: title })).toBeVisible();
      }
    });

    test('an experiment can be applied and says what to look for', async ({ page }) => {
      test.setTimeout(lab.slow ? 240_000 : 60_000);
      const second = page.locator('button[data-experiment]').nth(1);
      await second.click();
      await expect(second).toHaveAttribute('aria-pressed', 'true');
      await expect(page.getByText('What to look for:')).toBeVisible();
      await expect(page.getByTestId(lab.metric)).toContainText(/\d/, {
        timeout: lab.slow ? 200_000 : undefined,
      });
    });

    test('fits a phone screen without horizontal page scroll', async ({ page }) => {
      await page.setViewportSize({ width: 375, height: 800 });
      await page.reload();
      await expect(page.getByRole('heading', { name: lab.heading, level: 1 })).toBeVisible();
      const overflow = await page.evaluate(
        () => document.documentElement.scrollWidth - document.documentElement.clientWidth,
      );
      expect(overflow).toBeLessThanOrEqual(1);
    });
  });
}
