import { useMemo, useState } from 'react';
import { Badge, Panel } from '@/design-system';
import { usePalette } from '@/shared/canvas/colors';
import { useCompute } from '@/shared/compute/useCompute';
import { ExperimentBar } from '@/shared/components/ExperimentBar';
import { MiniChart } from '@/shared/components/MiniChart';
import {
  LabHeader,
  MetricsStrip,
  ParamPanel,
  StageNav,
  fix,
  type Metric,
  type ParamValue,
  type ParamValues,
} from '@/shared/lab-kit';
import { BEST_OF, DEFAULT_RLHF, type RlhfResult, type RlhfScenario } from './rlhfCore';
import { rlhfExperiments, rlhfParams, type RlhfExperiment } from './rlhfSpec';

const toScenario = (p: ParamValues): RlhfScenario => ({
  pairs: Number(p.pairs),
  noise: Number(p.noise),
});
const fromScenario = (s: RlhfScenario): ParamValues => ({ ...s });

const FEATURE_NAMES = [
  'quality view 1',
  'quality view 2',
  'quality view 3',
  'length',
  'pure noise',
];

function Results({ run }: { run: RlhfResult }) {
  const palette = usePalette();
  const metrics: Metric[] = [
    {
      label: 'Reward model agrees with people',
      value: `${fix(run.accuracy * 100, 0)}%`,
      testId: 'accuracy',
      hint: 'on new pairs',
    },
    {
      label: 'Best true reward on the way',
      value: fix(run.peakTruth, 2),
      hint: `original policy: ${fix(run.baseline, 2)}`,
    },
    {
      label: 'Lost afterwards',
      value: fix(run.fall, 2),
      hint: 'peak minus the sharpest policy',
    },
  ];
  const wmax = Math.max(...run.weights.map(Math.abs), 1e-9);
  return (
    <>
      <Panel
        id="lab-optimise"
        className="scroll-mt-24"
        title="Pushing on the reward model"
        eyebrow="01"
      >
        <div className="space-y-6">
          <MetricsStrip metrics={metrics} className="lg:grid-cols-3" />
          <MiniChart
            title="Reward as the policy moves further from the original (KL divergence)"
            x={run.kl}
            xLabel="distance from the original policy (KL, nats)"
            series={[
              { name: 'reward model score', color: palette.css.brand, values: run.proxy },
              { name: 'true reward', color: palette.css.pos, values: run.truth },
              {
                name: 'original policy',
                color: palette.css.muted,
                values: run.kl.map(() => run.baseline),
                dashed: true,
              },
            ]}
            height={260}
          />
          <p className="text-sm text-muted">
            Each point is the best policy for a given strength of penalty on moving away from the
            original: the weaker the penalty, the further right. The reward model&rsquo;s own score
            can only rise; the true reward is what people would have said.
          </p>
        </div>
      </Panel>
      <Panel id="lab-bestofn" className="scroll-mt-24" title="Best of n" eyebrow="02">
        <div className="space-y-4">
          <MiniChart
            title="True reward of the response the reward model likes best among n draws"
            x={BEST_OF}
            xLabel="draws n (log scale)"
            log
            series={[
              { name: 'true reward', color: palette.css.pos, values: run.bestOfTruth },
              { name: 'reward model score', color: palette.css.brand, values: run.bestOfProxy },
            ]}
            height={220}
          />
        </div>
      </Panel>
      <Panel
        id="lab-model"
        className="scroll-mt-24"
        title="What the reward model learned"
        eyebrow="03"
      >
        <div className="space-y-2">
          {run.weights.map((w, k) => (
            <div key={k} className="space-y-1">
              <div className="flex justify-between font-mono text-xs text-muted">
                <span>{FEATURE_NAMES[k]}</span>
                <span>{fix(w, 2)}</span>
              </div>
              <div className="h-2 rounded-full bg-line">
                <div
                  className="h-2 rounded-full"
                  style={{
                    width: `${(Math.abs(w) / wmax) * 100}%`,
                    background: w >= 0 ? palette.css.pos : palette.css.neg,
                  }}
                />
              </div>
            </div>
          ))}
          <p className="pt-2 text-sm text-muted">
            The three quality views are the signal and the noise feature should get weight zero; the
            more preference data, the closer the weights get to that.
          </p>
        </div>
      </Panel>
    </>
  );
}

export default function RlhfPage() {
  const [params, setParams] = useState<ParamValues>(() => fromScenario(DEFAULT_RLHF));
  const [activeId, setActiveId] = useState<string | null>(null);
  const scenario = useMemo(() => toScenario(params), [params]);
  const { result: run, error } = useCompute('rlhf', scenario);
  const setParam = (key: string, value: ParamValue) => setParams((p) => ({ ...p, [key]: value }));

  const apply = (exp: RlhfExperiment) => {
    setParams(fromScenario({ ...DEFAULT_RLHF, ...exp.scenario }));
    setActiveId(exp.id);
  };
  const stillActive = (() => {
    const exp = rlhfExperiments.find((e) => e.id === activeId);
    return (
      !!exp &&
      JSON.stringify(fromScenario({ ...DEFAULT_RLHF, ...exp.scenario })) === JSON.stringify(params)
    );
  })();

  const stages: [string, string][] = [
    ['lab-optimise', 'Pushing on the reward model'],
    ['lab-bestofn', 'Best of n'],
    ['lab-model', 'What the reward model learned'],
    ['lab-how', 'How it works'],
    ['lab-controls', 'Settings'],
  ];

  return (
    <div className="mx-auto max-w-6xl px-4 pt-8 pb-20 sm:px-6 lg:px-8">
      <LabHeader
        eyebrow="Modern topics"
        title="Learning from human feedback"
        accent="and the trouble with optimising a proxy"
        blurb="Chat models are tuned to please a reward model that was itself fitted to human preferences. The reward model is only an estimate, and a policy that chases it hard enough finds responses the estimate likes and people would not. Fit a reward model from preferences, push a policy against it, and watch the true reward rise and then fall."
        badges={
          <>
            <Badge>{scenario.pairs} preference pairs</Badge>
            <Badge>3,000 candidate responses</Badge>
          </>
        }
      />
      <StageNav stages={stages} />
      <ExperimentBar
        lessons={rlhfExperiments}
        activeId={stillActive ? activeId : null}
        onApply={apply}
      />

      <div className="flex flex-col gap-5">
        {run ? (
          <Results run={run} />
        ) : (
          <Panel
            id="lab-optimise"
            className="scroll-mt-24"
            title="Pushing on the reward model"
            eyebrow="01"
          >
            <p className="text-sm text-muted" role="status">
              {error ? `Something went wrong: ${error}` : 'Fitting reward models to human choices…'}
            </p>
          </Panel>
        )}
        <Panel id="lab-how" className="scroll-mt-24" title="How it works" eyebrow="04">
          <div className="space-y-3 text-sm text-muted">
            <p>
              There are 3,000 candidate responses. Each has a hidden true quality (three factors,
              and a penalty for being too long) and a length that is, in the data, correlated with
              the first quality factor. The reward model sees only noisy views of the three factors,
              the length and a pure-noise feature, and is fitted so that the response people
              preferred in each pair gets the higher score (the Bradley-Terry model). The people
              choose by true reward plus noise.
            </p>
            <p>
              The best policy for a given penalty has a closed form: the original probabilities
              multiplied by exp(score / β) and renormalised, where β sets how strongly moving away
              from the original is penalised. The tests check this against nearby distributions,
              that a weaker penalty always raises the score and the distance, and that the reward
              model recovers a known reward from enough data.
            </p>
          </div>
        </Panel>
        <Panel id="lab-controls" className="scroll-mt-24" title="Settings" eyebrow="05">
          <ParamPanel specs={rlhfParams} values={params} onChange={setParam} columns={2} />
        </Panel>
      </div>
    </div>
  );
}
