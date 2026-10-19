import { quantizeValues } from '@sim/deep';
import { useEffect, useMemo, useState } from 'react';
import { Badge, Panel } from '@/design-system';
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
import {
  DEFAULT_COMPRESSION,
  base,
  prepareTeacher,
  runCompression,
  type CompressionRun,
  type CompressionScenario,
} from './compressionCore';
import {
  compressionExperiments,
  compressionParams,
  type CompressionExperiment,
} from './compressionSpec';

const toScenario = (p: ParamValues): CompressionScenario => ({
  ...DEFAULT_COMPRESSION,
  mode: p.mode as CompressionScenario['mode'],
  bits: Number(p.bits),
  sparsity: Number(p.sparsity),
  fineTune: Boolean(p.fineTune),
  labelled: Number(p.labelled),
  temperature: Number(p.temperature),
});

const fromScenario = (s: CompressionScenario): ParamValues => ({
  mode: s.mode,
  bits: s.bits,
  sparsity: s.sparsity,
  fineTune: s.fineTune,
  labelled: s.labelled,
  temperature: s.temperature,
});

function Histogram({ bits }: { bits: number }) {
  const palette = usePalette();
  const w = useMemo(() => Array.from(base().teacher.weights[0]!.data), []);
  const q = bits >= 32 ? null : quantizeValues(w, bits);
  const lim = Math.max(...w.map(Math.abs));
  const bins = 61;
  const count = new Array<number>(bins).fill(0);
  w.forEach((v) => (count[Math.min(bins - 1, Math.floor(((v + lim) / (2 * lim)) * bins))]! += 1));
  const max = Math.max(...count);
  const levels = q ? Array.from(new Set(Array.from(q.values))).sort((a, b) => a - b) : [];
  const width = 560;
  const h = 130;
  return (
    <svg
      viewBox={`0 0 ${width} ${h}`}
      className="w-full max-w-2xl"
      role="img"
      aria-label="Distribution of the first-layer weights and the values they are rounded to"
    >
      {count.map((c, i) => (
        <rect
          key={i}
          x={(i / bins) * width}
          y={h - 20 - (c / max) * (h - 30)}
          width={width / bins - 1}
          height={(c / max) * (h - 30)}
          fill={palette.css.brand}
          opacity={0.6}
        />
      ))}
      {levels.map((l) => (
        <line
          key={l}
          x1={((l + lim) / (2 * lim)) * width}
          x2={((l + lim) / (2 * lim)) * width}
          y1={4}
          y2={h - 18}
          stroke={palette.css.neg}
          strokeWidth={1}
          opacity={0.8}
        />
      ))}
      <text x={4} y={h - 4} className="fill-faint font-mono text-[10px]">
        {(-lim).toFixed(2)}
      </text>
      <text x={width - 4} y={h - 4} textAnchor="end" className="fill-faint font-mono text-[10px]">
        {lim.toFixed(2)}
      </text>
    </svg>
  );
}

export default function CompressionPage() {
  const palette = usePalette();
  const [params, setParams] = useState<ParamValues>(() => fromScenario(DEFAULT_COMPRESSION));
  const [activeId, setActiveId] = useState<string | null>(null);
  const [run, setRun] = useState<CompressionRun | null>(null);
  const [sweep, setSweep] = useState<{ x: number[]; y: number[] } | null>(null);
  const scenario = useMemo(() => toScenario(params), [params]);

  useEffect(() => {
    let cancelled = false;
    let timer = 0;
    void prepareTeacher().then(() => {
      if (cancelled) return;
      timer = window.setTimeout(() => {
        if (cancelled) return;
        setRun(runCompression(scenario));
        if (scenario.mode === 'quantize') {
          const bits = [2, 3, 4, 5, 6, 8];
          setSweep({
            x: bits,
            y: bits.map((b) => runCompression({ ...scenario, bits: b }).accuracy),
          });
        } else if (scenario.mode === 'prune') {
          const sp = [0, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99];
          setSweep({
            x: sp.map((v) => v * 100),
            y: sp.map(
              (v) => runCompression({ ...scenario, sparsity: v, fineTune: false }).accuracy,
            ),
          });
        } else setSweep(null);
      }, 30);
    });
    return () => {
      cancelled = true;
      window.clearTimeout(timer);
    };
  }, [scenario]);

  const setParam = (key: string, value: ParamValue) => setParams((p) => ({ ...p, [key]: value }));
  const apply = (exp: CompressionExperiment) => {
    setParams(fromScenario({ ...DEFAULT_COMPRESSION, ...exp.scenario }));
    setActiveId(exp.id);
  };
  const stillActive = (() => {
    const exp = compressionExperiments.find((e) => e.id === activeId);
    return (
      !!exp &&
      JSON.stringify(fromScenario({ ...DEFAULT_COMPRESSION, ...exp.scenario })) ===
        JSON.stringify(params)
    );
  })();

  const metrics: Metric[] = run
    ? [
        { label: 'Accuracy (unseen digits)', value: pct(run.accuracy), testId: 'accuracy' },
        {
          label:
            scenario.mode === 'distill'
              ? 'Student trained on labels alone'
              : scenario.mode === 'prune'
                ? 'Before fine-tuning'
                : 'Full network',
          value: pct(run.baseline),
        },
        { label: 'Full network', value: pct(run.teacherAccuracy) },
        {
          label: 'Size',
          value: `${(run.bytes / 1024).toFixed(1)} KB of ${(run.fullBytes / 1024).toFixed(1)} KB`,
          hint: 'bytes needed for the weights',
        },
      ]
    : [];

  const stages: [string, string][] = [
    ['lab-result', 'The result'],
    ['lab-sweep', 'How far can it go?'],
    ['lab-how', 'How it works'],
    ['lab-controls', 'Settings'],
  ];

  return (
    <div className="mx-auto max-w-6xl px-4 pt-8 pb-20 sm:px-6 lg:px-8">
      <LabHeader
        eyebrow="Modern topics"
        title="Making models small"
        accent="quantise, prune, distil"
        blurb="A network that recognises handwritten digits has 18,752 weights. How many of them, and how many bits of each, does it really need? Round them, delete the small ones, or train a small student to imitate it, and see where it breaks."
        badges={
          <>
            <Badge>{scenario.mode}</Badge>
            <Badge>teacher: 256 → 64 → 32 → 10</Badge>
          </>
        }
      />
      <StageNav stages={stages} />
      <ExperimentBar
        lessons={compressionExperiments}
        activeId={stillActive ? activeId : null}
        onApply={apply}
      />

      <div className="flex flex-col gap-5">
        <Panel id="lab-result" className="scroll-mt-24" title="The result" eyebrow="01">
          {run ? (
            <div className="space-y-6">
              <MetricsStrip metrics={metrics} className="lg:grid-cols-4" />
              <p className="text-sm text-muted" data-testid="detail">
                {run.detail}
              </p>
              {scenario.mode === 'quantize' && (
                <div className="space-y-2">
                  <p className="eyebrow">
                    First-layer weights (bars) and the values they are rounded to (red lines)
                  </p>
                  <Histogram bits={scenario.bits} />
                </div>
              )}
            </div>
          ) : (
            <p className="text-sm text-muted">Training the teacher…</p>
          )}
        </Panel>
        <Panel id="lab-sweep" className="scroll-mt-24" title="How far can it go?" eyebrow="02">
          {sweep ? (
            <div className="grid gap-8 lg:grid-cols-[minmax(0,1fr)_20rem]">
              <MiniChart
                title={
                  scenario.mode === 'quantize'
                    ? 'Accuracy against bits per weight'
                    : 'Accuracy against weights removed (no fine-tuning)'
                }
                x={sweep.x}
                xLabel={scenario.mode === 'quantize' ? 'bits' : '% removed'}
                yRange={{ lo: 0, hi: 1 }}
                series={[{ name: 'accuracy', color: palette.css.brand, values: sweep.y }]}
                height={200}
              />
              <p className="text-sm text-muted">
                Accuracy stays flat while the damage is smaller than the network&apos;s margin, then
                falls fast. The knee of the curve is the practical limit.
              </p>
            </div>
          ) : (
            <p className="text-sm text-muted">
              {scenario.mode === 'distill'
                ? 'Distillation has one setting at a time: change the number of labelled digits to compare.'
                : 'Computing…'}
            </p>
          )}
        </Panel>
        <Panel id="lab-how" className="scroll-mt-24" title="How it works" eyebrow="03">
          <div className="space-y-3 text-sm text-muted">
            <p>
              <strong className="text-fg">Quantisation</strong> maps each weight to the nearest of
              2^bits − 1 evenly spaced values between −max and +max. The rounding error is at most
              half a step and has variance step²/12, so each extra bit divides the error power by
              four (checked in the tests). <strong className="text-fg">Pruning</strong> sets the
              smallest-magnitude weights to zero; fine-tuning with the zeros held fixed lets the
              rest compensate. <strong className="text-fg">Distillation</strong> trains a small
              student to match the teacher&apos;s probabilities, softened by a temperature so that
              the relative sizes of the wrong answers (a 4 looks a bit like a 9) are visible.
            </p>
          </div>
        </Panel>
        <Panel id="lab-controls" className="scroll-mt-24" title="Settings" eyebrow="04">
          <ParamPanel specs={compressionParams} values={params} onChange={setParam} columns={2} />
        </Panel>
      </div>
    </div>
  );
}
