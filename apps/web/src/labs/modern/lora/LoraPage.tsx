import { Play } from 'lucide-react';
import { useEffect, useMemo, useState } from 'react';
import { Badge, Button, Panel } from '@/design-system';
import { usePalette } from '@/shared/canvas/colors';
import { ExperimentBar } from '@/shared/components/ExperimentBar';
import { MiniChart } from '@/shared/components/MiniChart';
import {
  LabHeader,
  MetricsStrip,
  ParamPanel,
  StageNav,
  pct,
  type Metric,
  type ParamValue,
  type ParamValues,
} from '@/shared/lab-kit';
import { prepareTeacher } from '../compression/compressionCore';
import { DEFAULT_LORA, METHOD_LABELS, runLora, type LoraRun, type LoraScenario } from './loraCore';
import { loraExperiments, loraParams, type LoraExperiment } from './loraSpec';

const toScenario = (p: ParamValues): LoraScenario => ({
  ...DEFAULT_LORA,
  method: p.method as LoraScenario['method'],
  rank: Number(p.rank),
  perClass: Number(p.perClass),
  learningRate: Number(p.learningRate),
  epochs: Number(p.epochs),
});

const fromScenario = (s: LoraScenario): ParamValues => ({
  method: s.method,
  rank: s.rank,
  perClass: s.perClass,
  learningRate: s.learningRate,
  epochs: s.epochs,
});

interface Point {
  label: string;
  trainable: number;
  accuracy: number;
}

export default function LoraPage() {
  const palette = usePalette();
  const [params, setParams] = useState<ParamValues>(() => fromScenario(DEFAULT_LORA));
  const [activeId, setActiveId] = useState<string | null>(null);
  const [run, setRun] = useState<LoraRun | null>(null);
  const [points, setPoints] = useState<Point[] | null>(null);
  const [busy, setBusy] = useState(false);
  const scenario = useMemo(() => toScenario(params), [params]);

  useEffect(() => {
    let cancelled = false;
    let timer = 0;
    void prepareTeacher().then(() => {
      if (cancelled) return;
      timer = window.setTimeout(() => !cancelled && setRun(runLora(scenario)), 30);
    });
    return () => {
      cancelled = true;
      window.clearTimeout(timer);
    };
  }, [scenario]);

  const compare = async () => {
    setBusy(true);
    const configs: { label: string; s: LoraScenario }[] = [
      { label: 'last layer', s: { ...scenario, method: 'head' } },
      ...[1, 2, 4, 8, 16].map((rank) => ({
        label: `LoRA ${rank}`,
        s: { ...scenario, method: 'lora' as const, rank },
      })),
      { label: 'full', s: { ...scenario, method: 'full' } },
    ];
    const out: Point[] = [];
    for (const c of configs) {
      await new Promise((r) => setTimeout(r, 20));
      const r = runLora(c.s);
      out.push({ label: c.label, trainable: r.trainable, accuracy: r.shiftedAfter });
      setPoints([...out]);
    }
    setBusy(false);
  };

  const setParam = (key: string, value: ParamValue) => {
    setParams((p) => ({ ...p, [key]: value }));
    setPoints(null);
  };
  const apply = (exp: LoraExperiment) => {
    setParams(fromScenario({ ...DEFAULT_LORA, ...exp.scenario }));
    setPoints(null);
    setActiveId(exp.id);
  };
  const stillActive = (() => {
    const exp = loraExperiments.find((e) => e.id === activeId);
    return (
      !!exp &&
      JSON.stringify(fromScenario({ ...DEFAULT_LORA, ...exp.scenario })) === JSON.stringify(params)
    );
  })();

  const metrics: Metric[] = run
    ? [
        { label: 'Shifted digits: before', value: pct(run.shiftedBefore) },
        { label: 'Shifted digits: after', value: pct(run.shiftedAfter), testId: 'after' },
        {
          label: 'Original digits: after',
          value: pct(run.originalAfter),
          tone: run.originalAfter < 0.9 ? 'bad' : 'good',
          hint: 'accuracy on the situation the network was trained for: forgetting shows up here',
        },
        {
          label: 'Trainable numbers',
          value: `${run.trainable.toLocaleString()} of ${run.total.toLocaleString()}`,
        },
      ]
    : [];

  const stages: [string, string][] = [
    ['lab-result', 'Adapting'],
    ['lab-compare', 'Cost against accuracy'],
    ['lab-how', 'How it works'],
    ['lab-controls', 'Settings'],
  ];

  return (
    <div className="mx-auto max-w-6xl px-4 pt-8 pb-20 sm:px-6 lg:px-8">
      <LabHeader
        eyebrow="Modern topics"
        title="Fine-tuning with LoRA"
        accent="teaching an old network a new trick"
        blurb="A digit reader is trained for upright digits and then the camera moves. Adapt it from a handful of examples: re-train everything, only the last layer, or freeze the network and learn a small low-rank correction to every layer."
        badges={
          <>
            <Badge>{METHOD_LABELS[scenario.method]}</Badge>
            {scenario.method === 'lora' && <Badge>rank {scenario.rank}</Badge>}
            <Badge>{scenario.perClass * 10} labelled examples</Badge>
          </>
        }
      />
      <StageNav stages={stages} />
      <ExperimentBar
        lessons={loraExperiments}
        activeId={stillActive ? activeId : null}
        onApply={apply}
      />

      <div className="flex flex-col gap-5">
        <Panel id="lab-result" className="scroll-mt-24" title="Adapting" eyebrow="01">
          {run ? (
            <div className="space-y-5">
              <MetricsStrip metrics={metrics} className="lg:grid-cols-4" />
              {run.trainLoss.length > 1 && (
                <MiniChart
                  title="Loss on the few training examples"
                  x={run.trainLoss.map((_, i) => i + 1)}
                  xLabel="epoch"
                  series={[{ name: 'loss', color: palette.css.brand, values: run.trainLoss }]}
                  height={170}
                />
              )}
            </div>
          ) : (
            <p className="text-sm text-muted">Preparing the pre-trained network…</p>
          )}
        </Panel>
        <Panel id="lab-compare" className="scroll-mt-24" title="Cost against accuracy" eyebrow="02">
          <div className="space-y-4">
            <Button size="sm" variant="secondary" onClick={() => void compare()} disabled={busy}>
              <Play className="size-4" />{' '}
              {busy ? 'Comparing…' : 'Compare all methods with these settings'}
            </Button>
            {points && (
              <div className="space-y-1.5" data-testid="comparison">
                {points.map((p) => (
                  <div
                    key={p.label}
                    className="grid grid-cols-[6rem_minmax(0,1fr)_9rem] items-center gap-3 text-[0.8rem]"
                  >
                    <span className="text-muted">{p.label}</span>
                    <div className="h-2.5 rounded-full bg-raised">
                      <div
                        className="h-2.5 rounded-full"
                        style={{
                          width: `${p.accuracy * 100}%`,
                          background: palette.css.brand,
                          opacity: 0.85,
                        }}
                      />
                    </div>
                    <span className="text-right font-mono text-[0.74rem] text-faint">
                      {pct(p.accuracy)} · {p.trainable.toLocaleString()} weights
                    </span>
                  </div>
                ))}
              </div>
            )}
            <p className="text-sm text-muted">
              Accuracy on shifted digits against the number of weights that were trained. LoRA sits
              between re-training the last layer and re-training everything, and the rank dials the
              trade-off.
            </p>
          </div>
        </Panel>
        <Panel id="lab-how" className="scroll-mt-24" title="How it works" eyebrow="03">
          <div className="space-y-3 text-sm text-muted">
            <p>
              A layer computes <em>y = xW</em>. LoRA keeps W frozen and learns two thin matrices A
              (inputs × r) and B (r × outputs), so the layer becomes <em>y = x(W + AB)</em>. B
              starts at zero, so training begins from exactly the pre-trained network, and only
              r·(inputs + outputs) numbers per layer are learned instead of inputs·outputs. The
              update AB has rank at most r: it can only move the weights in r directions, which is
              often enough because fine-tuning changes a network far less than training it did.
            </p>
            <p>
              After training the product can be added into W (the tests check that predictions do
              not change) or kept separate, so one base model can serve many tasks by swapping small
              adapters. The gradients of A and B are checked against finite differences.
            </p>
          </div>
        </Panel>
        <Panel id="lab-controls" className="scroll-mt-24" title="Settings" eyebrow="04">
          <ParamPanel specs={loraParams} values={params} onChange={setParam} columns={2} />
        </Panel>
      </div>
    </div>
  );
}
