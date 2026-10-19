import { useEffect, useMemo, useState } from 'react';
import type { LoadedTransformer } from '@sim/deep';
import { Badge, Panel, cn } from '@/design-system';
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
import { loadTinyGpt } from '../transformer/assets';
import {
  DEFAULT_ABLATION,
  runAblation,
  type AblationMode,
  type AblationRun,
  type AblationScenario,
} from './ablationCore';
import { ablationExperiments, ablationParams, type AblationExperiment } from './ablationSpec';

const toScenario = (p: ParamValues, silenced: string[]): AblationScenario => ({
  mode: p.mode as AblationMode,
  prompt: String(p.prompt),
  silenced,
});

export default function HeadAblationPage() {
  const palette = usePalette();
  const [loaded, setLoaded] = useState<LoadedTransformer | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [params, setParams] = useState<ParamValues>({
    mode: DEFAULT_ABLATION.mode,
    prompt: DEFAULT_ABLATION.prompt,
  });
  const [silenced, setSilenced] = useState<string[]>([]);
  const [activeId, setActiveId] = useState<string | null>(null);
  const setParam = (key: string, value: ParamValue) => setParams((p) => ({ ...p, [key]: value }));

  useEffect(() => {
    let cancelled = false;
    loadTinyGpt()
      .then((m) => !cancelled && setLoaded(m))
      .catch((e: unknown) => !cancelled && setError(e instanceof Error ? e.message : String(e)));
    return () => {
      cancelled = true;
    };
  }, []);

  const scenario = useMemo(() => toScenario(params, silenced), [params, silenced]);
  const run: AblationRun | null = useMemo(
    () => (loaded ? runAblation(loaded, scenario) : null),
    [loaded, scenario],
  );

  const toggle = (key: string) =>
    setSilenced((s) => (s.includes(key) ? s.filter((k) => k !== key) : [...s, key]));

  const apply = (exp: AblationExperiment) => {
    const next = { ...DEFAULT_ABLATION, ...exp.scenario };
    setParams({ mode: next.mode, prompt: next.prompt });
    setSilenced(next.silenced);
    setActiveId(exp.id);
  };
  const stillActive = (() => {
    const exp = ablationExperiments.find((e) => e.id === activeId);
    if (!exp) return false;
    const want = { ...DEFAULT_ABLATION, ...exp.scenario };
    return (
      want.mode === scenario.mode &&
      want.prompt === scenario.prompt &&
      [...want.silenced].sort().join() === [...silenced].sort().join()
    );
  })();

  const metrics: Metric[] = run
    ? [
        {
          label: 'Loss on the stories',
          value: fix(run.result.baseline, 2),
          testId: 'baseline-loss',
          hint: 'nats per token, nothing silenced',
        },
        {
          label: 'With the chosen heads silenced',
          value: fix(run.chosenLoss, 2),
          hint: silenced.length ? `${silenced.length} silenced` : 'none chosen',
        },
        {
          label: 'Most important head',
          value: (() => {
            const flat = run.result.delta.flat();
            const i = flat.indexOf(Math.max(...flat));
            const heads = run.result.delta[0]?.length ?? 1;
            return `L${Math.floor(i / heads) + 1} · H${(i % heads) + 1}`;
          })(),
          hint: `+${fix(Math.max(...run.result.delta.flat()), 2)} nats if removed`,
        },
        {
          label: 'Next word after the prompt',
          value: `${run.next[0]?.token.trim() ?? ''} ${fix((run.next[0]?.p ?? 0) * 100, 0)}%`,
          hint: run.next
            .slice(1, 3)
            .map((n) => `${n.token.trim() || '·'} ${fix(n.p * 100, 0)}%`)
            .join(', '),
        },
      ]
    : [];

  const maxDelta = run ? Math.max(0.01, ...run.result.delta.flat()) : 1;
  const stages: [string, string][] = [
    ['lab-heads', 'The eight heads'],
    ['lab-curve', 'Losing heads one by one'],
    ['lab-how', 'How it works'],
    ['lab-controls', 'Settings'],
  ];

  return (
    <div className="mx-auto max-w-6xl px-4 pt-8 pb-20 sm:px-6 lg:px-8">
      <LabHeader
        eyebrow="Transformers"
        title="Attention head ablation"
        accent="what each head is for"
        blurb="The tiny story model has eight attention heads. Switch them off, one at a time or in groups, and see how much worse it predicts the next word. Some turn out to do the work; some do next to nothing; one layer carries the link from a name to its pronoun."
        badges={
          <>
            <Badge>2 layers × 4 heads</Badge>
            <Badge>{silenced.length} silenced</Badge>
          </>
        }
      />
      <StageNav stages={stages} />
      <ExperimentBar
        lessons={ablationExperiments}
        activeId={stillActive ? activeId : null}
        onApply={apply}
      />

      <div className="flex flex-col gap-5">
        <Panel id="lab-heads" className="scroll-mt-24" title="The eight heads" eyebrow="01">
          {error ? (
            <p className="text-sm text-neg" role="alert">
              Could not load the model: {error}
            </p>
          ) : !run ? (
            <p className="text-sm text-muted" role="status">
              Loading the model…
            </p>
          ) : (
            <div className="space-y-6">
              <MetricsStrip metrics={metrics} className="lg:grid-cols-4" />
              <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
                {run.result.delta.flatMap((row, l) =>
                  row.map((d, h) => {
                    const key = `${l}.${h}`;
                    const off = silenced.includes(key);
                    const p = run.patterns[l]?.[h];
                    return (
                      <button
                        key={key}
                        type="button"
                        aria-pressed={off}
                        onClick={() => toggle(key)}
                        className={cn(
                          'rounded-xl border p-3 text-left transition-colors',
                          off
                            ? 'border-neg bg-neg/10'
                            : 'border-line bg-surface hover:border-brand',
                        )}
                      >
                        <div className="flex items-baseline justify-between">
                          <span className="font-mono text-sm text-fg">
                            Layer {l + 1} · Head {h + 1}
                          </span>
                          <span className="text-xs text-muted">{off ? 'silenced' : 'on'}</span>
                        </div>
                        <div
                          className="mt-2 h-2 rounded-full bg-line"
                          role="img"
                          aria-label={`Loss rises by ${fix(d, 3)} if removed`}
                        >
                          <div
                            className="h-2 rounded-full"
                            style={{
                              width: `${Math.max(0, d / maxDelta) * 100}%`,
                              background: palette.css.brand,
                            }}
                          />
                        </div>
                        <div className="mt-1 font-mono text-xs text-muted">
                          +{fix(d, 3)} nats if removed
                        </div>
                        {p && (
                          <div className="mt-2 grid grid-cols-2 gap-x-2 font-mono text-[11px] text-muted">
                            <span>previous {fix(p.previous, 2)}</span>
                            <span>first {fix(p.first, 2)}</span>
                            <span>itself {fix(p.self, 2)}</span>
                            <span>spread {fix(p.entropy, 2)}</span>
                          </div>
                        )}
                      </button>
                    );
                  }),
                )}
              </div>
              <p className="text-sm text-muted">
                Click a head to silence it and see the loss and the next word change. The bar shows
                how much worse the average prediction gets when that head alone is removed. The four
                numbers are what the head attends to on average: the share of attention on the
                previous word, on the first word and on itself, and how widely it spreads the rest
                (higher is more scattered).
              </p>
            </div>
          )}
        </Panel>
        <Panel id="lab-curve" className="scroll-mt-24" title="Losing heads one by one" eyebrow="02">
          {run ? (
            <div className="space-y-3">
              <MiniChart
                title="Loss on the stories as heads are removed, least useful first"
                x={run.result.cumulative.map((_, i) => i)}
                xLabel="heads removed"
                series={[
                  {
                    name: 'loss',
                    color: palette.css.neg,
                    values: run.result.cumulative.map((c) => c.loss),
                  },
                ]}
                height={220}
              />
              <p className="text-sm text-muted">
                The first few heads can go almost for free; then the loss climbs steeply.
              </p>
            </div>
          ) : (
            <p className="text-sm text-muted">Loading the model…</p>
          )}
        </Panel>
        <Panel id="lab-how" className="scroll-mt-24" title="How it works" eyebrow="03">
          <div className="space-y-3 text-sm text-muted">
            <p>
              Each attention head reads the earlier words and writes a vector that is added to the
              running summary of the text. To silence a head its vector is replaced by zeros, or by
              its average over the text, and the rest of the model runs as usual. The loss is the
              average of minus the log of the probability the model gave the word that actually came
              next, over 24 short stories from the same grammar as the model&rsquo;s training data
              but none that it saw.
            </p>
            <p>
              The tests check that silencing nothing changes nothing, that a silenced head leaves
              exactly zeros in its own columns and other heads untouched, that with every head
              silenced the attention output is the same at every position, and every sentence on
              this page.
            </p>
          </div>
        </Panel>
        <Panel id="lab-controls" className="scroll-mt-24" title="Settings" eyebrow="04">
          <ParamPanel specs={ablationParams} values={params} onChange={setParam} columns={2} />
        </Panel>
      </div>
    </div>
  );
}
