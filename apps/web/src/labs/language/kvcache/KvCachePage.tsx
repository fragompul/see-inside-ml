import { useMemo, useState } from 'react';
import { Badge, Panel } from '@/design-system';
import { usePalette } from '@/shared/canvas/colors';
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
import { CONTEXTS, DEFAULT_KV, MODELS, gigabytes, runKv, type KvScenario } from './kvcacheCore';
import { kvExperiments, kvParams, type KvExperiment } from './kvcacheSpec';

const toScenario = (p: ParamValues): KvScenario => ({
  model: p.model as KvScenario['model'],
  sharing: p.sharing as KvScenario['sharing'],
  prompt: Number(p.prompt),
  steps: Number(p.steps),
});

const fromScenario = (s: KvScenario): ParamValues => ({ ...s });

const tera = (macs: number) => macs / 1e12;
const size = (gb: number) => (gb >= 10 ? `${fix(gb, 0)} GB` : `${fix(gb, 2)} GB`);
const tokens = (c: number) => (c >= 1024 ? `${c / 1024}k` : String(c));

export default function KvCachePage() {
  const palette = usePalette();
  const [params, setParams] = useState<ParamValues>(() => fromScenario(DEFAULT_KV));
  const [activeId, setActiveId] = useState<string | null>(null);
  const scenario = useMemo(() => toScenario(params), [params]);
  const run = useMemo(() => runKv(scenario), [scenario]);
  const setParam = (key: string, value: ParamValue) => setParams((p) => ({ ...p, [key]: value }));

  const apply = (exp: KvExperiment) => {
    setParams(fromScenario({ ...DEFAULT_KV, ...exp.scenario }));
    setActiveId(exp.id);
  };
  const stillActive = (() => {
    const exp = kvExperiments.find((e) => e.id === activeId);
    return (
      !!exp &&
      JSON.stringify(fromScenario({ ...DEFAULT_KV, ...exp.scenario })) === JSON.stringify(params)
    );
  })();

  const metrics: Metric[] = [
    {
      label: 'Work saved by the cache',
      value: `${fix(run.speedup, 0)}×`,
      testId: 'speedup',
      hint: 'multiply-adds, without ÷ with',
    },
    { label: 'Work with a cache', value: `${fix(tera(run.withCache), 2)} T` },
    { label: 'Work without', value: `${fix(tera(run.without), 1)} T` },
    {
      label: 'Cache at the end of the text',
      value: size(gigabytes(run.cacheEnd)),
      hint: `${scenario.prompt + scenario.steps} tokens`,
    },
  ];

  const stages: [string, string][] = [
    ['lab-work', 'Work done'],
    ['lab-memory', 'Memory kept'],
    ['lab-how', 'How it works'],
    ['lab-controls', 'Settings'],
  ];

  return (
    <div className="mx-auto max-w-6xl px-4 pt-8 pb-20 sm:px-6 lg:px-8">
      <LabHeader
        eyebrow="Language models"
        title="The KV cache and the cost of attention"
        accent="trading memory for time"
        blurb="A language model writes one word at a time, and each word needs the keys and values of every word before it. Remember them and the work per word stays small, but the memory grows with the text. See what the cache saves, what it costs, and how sharing keys and values between heads shrinks it."
        badges={
          <>
            <Badge>{MODELS[scenario.model].label}</Badge>
            <Badge>
              {scenario.prompt} + {scenario.steps} tokens
            </Badge>
          </>
        }
      />
      <StageNav stages={stages} />
      <ExperimentBar
        lessons={kvExperiments}
        activeId={stillActive ? activeId : null}
        onApply={apply}
      />

      <div className="flex flex-col gap-5">
        <Panel id="lab-work" className="scroll-mt-24" title="Work done" eyebrow="01">
          <div className="space-y-6">
            <MetricsStrip metrics={metrics} className="lg:grid-cols-4" />
            <MiniChart
              title="Total work so far (trillions of multiply-adds, log scale)"
              x={run.grid}
              xLabel="words written"
              log
              series={[
                {
                  name: 'with a cache',
                  color: palette.css.brand,
                  values: run.curveWith.map(tera),
                },
                {
                  name: 'without',
                  color: palette.css.neg,
                  values: run.curveWithout.map(tera),
                  dashed: true,
                },
              ]}
              height={240}
            />
            <p className="text-sm text-muted">
              Counted for the attention layers and the vocabulary scores. The first word has to read
              the whole prompt either way, which is why the cached curve starts well above zero.
            </p>
          </div>
        </Panel>
        <Panel id="lab-memory" className="scroll-mt-24" title="Memory kept" eyebrow="02">
          <div className="space-y-4">
            <MiniChart
              title="Cache size by length of text (GB, log scale)"
              x={CONTEXTS}
              xLabel="tokens in the text"
              log
              series={[
                {
                  name: 'every head separate',
                  color: palette.css.neg,
                  values: run.memory.mha.map(gigabytes),
                },
                {
                  name: 'grouped (÷4)',
                  color: palette.css.brand,
                  values: run.memory.gqa.map(gigabytes),
                },
                {
                  name: 'multi-query',
                  color: palette.css.pos,
                  values: run.memory.mqa.map(gigabytes),
                },
              ]}
              height={240}
            />
            <p className="text-sm text-muted">
              Share of one new word&rsquo;s work spent in attention rather than in the projections:{' '}
              {CONTEXTS.map(
                (c, i) => `${tokens(c)} tokens ${fix((run.attentionShare[i] as number) * 100, 0)}%`,
              ).join(', ')}
              .
            </p>
          </div>
        </Panel>
        <Panel id="lab-how" className="scroll-mt-24" title="How it works" eyebrow="03">
          <div className="space-y-3 text-sm text-muted">
            <p>
              To write the next word, attention compares one query (the last word) with the key of
              every word so far and then mixes their values. The keys and values of earlier words do
              not change as the text grows, so they can be kept instead of recomputed: this is the
              KV cache. Without it, every step recomputes them for the whole text, so the work to
              write <em>n</em> words grows with <em>n</em> squared; with it, each word costs a fixed
              amount plus a part proportional to the length so far.
            </p>
            <p>
              The price is memory: two vectors per layer per token. Grouped-query and multi-query
              attention shrink it by making several heads read the same keys and values. This lab
              runs a real miniature decoder with and without a cache to check that the text is
              identical and that the multiply-adds it counts match the formulas used for the large
              models. The formulas leave out the feed-forward blocks, which a cache also saves, so
              the real saving is larger still.
            </p>
          </div>
        </Panel>
        <Panel id="lab-controls" className="scroll-mt-24" title="Settings" eyebrow="04">
          <ParamPanel specs={kvParams} values={params} onChange={setParam} columns={2} />
        </Panel>
      </div>
    </div>
  );
}
