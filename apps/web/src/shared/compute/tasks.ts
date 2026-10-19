/**
 * Heavy computations that run off the main thread. Each task is a pure function from a plain input to a plain
 * (structured-cloneable) result, loaded on demand so that a page only pulls in the engine it needs.
 */
export const TASKS = {
  'latent-control': async () =>
    (await import('@/labs/generative/latent-control/controlCore')).runControl,
  // closures and agents do not cross the worker boundary: send the numbers the pages draw from
  scaling: async () => {
    const { runScaling } = await import('@/labs/modern/scaling/scalingCore');
    const plain = ({ predict: _predict, ...law }: ReturnType<typeof runScaling>['dataLaw']) => law;
    return (input: Parameters<typeof runScaling>[0]) => {
      const run = runScaling(input);
      return { ...run, dataLaw: plain(run.dataLaw), featureLaw: plain(run.featureLaw) };
    };
  },
  'actor-critic': async () => {
    const { runAc } = await import('@/labs/rl/actor-critic/acCore');
    return (input: Parameters<typeof runAc>[0]) => {
      const { scenario, optimal, curves, final, speed, variance, bias } = runAc(input);
      return { scenario, optimal, curves, final, speed, variance, bias };
    };
  },
  'ngram-lm': async () => (await import('@/labs/language/ngram-vs-neural/lmCore')).runLm,
  'dqn-config': async () => {
    const core = await import('@/labs/rl/dqn/dqnCore');
    return (input: { scenario: Parameters<typeof core.runConfig>[0]; id: string }) =>
      core.runConfig(
        input.scenario,
        core.CONFIGS.find((c) => c.id === input.id) as (typeof core.CONFIGS)[number],
      );
  },
  augment: async () => (await import('@/labs/deep/augmentation/augmentCore')).runAugment,
  'gan-variants': async () => (await import('@/labs/generative/gan-variants/ganCore')).runGanLab,
  flows: async () => (await import('@/labs/generative/flows/flowCore')).runFlowLab,
  guidance: async () => (await import('@/labs/generative/guidance/guidanceCore')).runGuidance,
  attribution: async () =>
    (await import('@/labs/interpretability/attribution/attributionCore')).runAttribution,
  robustness: async () =>
    (await import('@/labs/interpretability/robustness/robustCore')).runRobustness,
  probing: async () => {
    const [core, assets] = await Promise.all([
      import('@/labs/interpretability/probing/probingCore'),
      import('@/labs/deep/transformer/assets'),
    ]);
    core.setStoryModelLoader(() => assets.loadTinyGpt());
    return core.runProbing;
  },
  'feature-vis': async () => (await import('@/labs/interpretability/feature-vis/visCore')).runVis,
  moe: async () => (await import('@/labs/modern/moe/moeCore')).runMoeLab,
  gnn: async () => (await import('@/labs/modern/gnn/gnnCore')).runGnnLab,
  'neural-ode': async () => (await import('@/labs/modern/neural-ode/odeCore')).runOde,
  rag: async () => (await import('@/labs/language/rag/ragCore')).runRag,
  rlhf: async () => (await import('@/labs/modern/rlhf/rlhfCore')).runRlhf,
} as const;

export type TaskName = keyof typeof TASKS;
export type TaskInput<T extends TaskName> = Parameters<Awaited<ReturnType<(typeof TASKS)[T]>>>[0];
export type TaskResult<T extends TaskName> = Awaited<
  ReturnType<Awaited<ReturnType<(typeof TASKS)[T]>>>
>;
