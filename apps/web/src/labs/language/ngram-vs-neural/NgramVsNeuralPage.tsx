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
import { DEFAULT_LM, ORDERS, type LmScenario } from './lmCore';
import { lmExperiments, lmParams, type LmExperiment } from './lmSpec';

const toScenario = (p: ParamValues): LmScenario => ({
  trainSentences: Number(p.trainSentences),
  smoothing: p.smoothing as LmScenario['smoothing'],
  neuralSteps: Number(p.neuralSteps),
});

const fromScenario = (s: LmScenario): ParamValues => ({ ...s });

const ppl = (nats: number) =>
  Number.isNaN(nats) ? '–' : nats === Infinity ? '∞' : fix(Math.exp(nats), 2);

export default function NgramVsNeuralPage() {
  const palette = usePalette();
  const [params, setParams] = useState<ParamValues>(() => fromScenario(DEFAULT_LM));
  const [activeId, setActiveId] = useState<string | null>(null);
  const scenario = useMemo(() => toScenario(params), [params]);
  const { result } = useCompute('ngram-lm', scenario);
  const setParam = (key: string, value: ParamValue) => setParams((p) => ({ ...p, [key]: value }));

  const run = result;
  const busy = !run || JSON.stringify(run.scenario) !== JSON.stringify(scenario);

  const apply = (exp: LmExperiment) => {
    setParams(fromScenario({ ...DEFAULT_LM, ...exp.scenario }));
    setActiveId(exp.id);
  };
  const stillActive = (() => {
    const exp = lmExperiments.find((e) => e.id === activeId);
    return (
      !!exp &&
      JSON.stringify(fromScenario({ ...DEFAULT_LM, ...exp.scenario })) === JSON.stringify(params)
    );
  })();

  const metrics: Metric[] = run
    ? [
        {
          label: 'Best n-gram, new text',
          value: ppl(Math.min(...run.test)),
          testId: 'ngram-ppl',
          hint: `perplexity, order ${ORDERS[run.test.indexOf(Math.min(...run.test))]}`,
        },
        {
          label: 'Neural network, new text',
          value: ppl(run.neuralTest),
          hint: run.scenario.neuralSteps
            ? `after ${run.scenario.neuralSteps} steps`
            : 'not trained',
        },
        { label: 'Best possible', value: ppl(run.floor), hint: 'known from the grammar' },
        { label: 'Pure guessing', value: ppl(run.uniform), hint: `${run.alphabet} characters` },
      ]
    : [];

  // a model that gives something probability zero would be off the chart: draw it at the top
  const cap = (v: number) =>
    Number.isFinite(v) ? Math.min(v, run?.uniform ?? 3) : (run?.uniform ?? 3);

  const stages: [string, string][] = [
    ['lab-curves', 'Context length'],
    ['lab-how', 'How it works'],
    ['lab-controls', 'Settings'],
  ];

  return (
    <div className="mx-auto max-w-6xl px-4 pt-8 pb-20 sm:px-6 lg:px-8">
      <LabHeader
        eyebrow="Language models"
        title="N-grams versus a neural language model"
        accent="counting versus learning"
        blurb="The oldest language models just count which characters follow which. A recurrent network learns the same job by gradient descent. Measure both by perplexity on text neither has seen, on a toy language whose best possible score is known exactly."
        badges={
          <>
            <Badge>{scenario.trainSentences} training sentences</Badge>
            <Badge>
              {scenario.neuralSteps ? `${scenario.neuralSteps} training steps` : 'n-grams only'}
            </Badge>
          </>
        }
      />
      <StageNav stages={stages} />
      <ExperimentBar
        lessons={lmExperiments}
        activeId={stillActive ? activeId : null}
        onApply={apply}
      />

      <div className="flex flex-col gap-5">
        <Panel id="lab-curves" className="scroll-mt-24" title="Context length" eyebrow="01">
          {run ? (
            <div className="space-y-6" aria-busy={busy}>
              <MetricsStrip metrics={metrics} className="lg:grid-cols-4" />
              <MiniChart
                title="Loss on new text by n-gram order (nats per character, log scale)"
                x={ORDERS}
                xLabel="characters of context + 1 (order)"
                log
                series={[
                  {
                    name: 'n-gram, new text',
                    color: palette.css.brand,
                    values: run.test.map(cap),
                  },
                  {
                    name: 'n-gram, training text',
                    color: palette.css.muted,
                    values: run.train.map(cap),
                    dashed: true,
                  },
                  ...(Number.isNaN(run.neuralTest)
                    ? []
                    : [
                        {
                          name: 'neural network, new text',
                          color: palette.css.neg,
                          values: ORDERS.map(() => run.neuralTest),
                        },
                      ]),
                  {
                    name: 'best possible',
                    color: palette.css.pos,
                    values: ORDERS.map(() => run.floor),
                    dashed: true,
                  },
                ]}
                height={260}
              />
              <p className="text-sm text-muted">
                Perplexity is e raised to the loss: roughly how many characters the model is
                choosing between at each step. A model that gives probability zero to something that
                happens has infinite loss; those points are drawn at the top of the chart. The
                network does not use an order, so its result is a flat line.
              </p>
            </div>
          ) : (
            <p className="text-sm text-muted" role="status">
              Training the network…
            </p>
          )}
        </Panel>
        <Panel id="lab-how" className="scroll-mt-24" title="How it works" eyebrow="02">
          <div className="space-y-3 text-sm text-muted">
            <p>
              An n-gram model keeps a table of how often each character followed each context of the
              previous <em>n − 1</em> characters. It is fast and exact on text it has seen, and
              helpless on text it has not, so practical versions smooth the counts. The recurrent
              network reads one character at a time, keeps a running summary, and shares what it
              learns across contexts.
            </p>
            <p>
              The language is made of sentences like &ldquo;the cat sat on the mat.&rdquo;, 27
              equally likely variants, so the true entropy is known and nothing can score below it.
              The new text has 200 sentences from a different random stream. The tests run every
              experiment, including the claim about the best possible score.
            </p>
          </div>
        </Panel>
        <Panel id="lab-controls" className="scroll-mt-24" title="Settings" eyebrow="03">
          <ParamPanel specs={lmParams} values={params} onChange={setParam} columns={2} />
        </Panel>
      </div>
    </div>
  );
}
