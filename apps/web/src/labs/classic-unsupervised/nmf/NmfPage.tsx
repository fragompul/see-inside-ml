import { useEffect, useMemo, useState } from 'react';
import { Badge, Panel, Slider } from '@/design-system';
import { usePalette } from '@/shared/canvas/colors';
import { ExperimentBar } from '@/shared/components/ExperimentBar';
import { MiniChart } from '@/shared/components/MiniChart';
import { PixelCanvas } from '@/shared/components/PixelCanvas';
import {
  LabHeader,
  MetricsStrip,
  ParamPanel,
  StageNav,
  fix,
  pct,
  type Metric,
  type ParamValue,
  type ParamValues,
} from '@/shared/lab-kit';
import { DEFAULT_NMF, runNmf, type NmfRun, type NmfScenario } from './nmfCore';
import { nmfExperiments, nmfParams, type NmfExperiment } from './nmfSpec';

const toScenario = (p: ParamValues): NmfScenario => ({
  ...DEFAULT_NMF,
  components: Number(p.components),
  loss: p.loss as NmfScenario['loss'],
  iterations: Number(p.iterations),
  digits: p.digits as NmfScenario['digits'],
});

const fromScenario = (s: NmfScenario): ParamValues => ({
  components: s.components,
  loss: s.loss,
  iterations: s.iterations,
  digits: s.digits,
});

function Parts({ run }: { run: NmfRun }) {
  const k = run.scenario.components;
  const shown = Math.min(k, 16);
  const px = k <= 6 ? 88 : 64;
  return (
    <div className="space-y-6">
      <div className="space-y-2">
        <p className="eyebrow">NMF parts (can only be added)</p>
        <div className="flex flex-wrap gap-2" data-testid="nmf-parts">
          {Array.from({ length: shown }, (_, c) => (
            <PixelCanvas
              key={c}
              values={Array.from(run.model.h.data.slice(c * 256, (c + 1) * 256))}
              side={16}
              size={px}
              ariaLabel={`NMF part ${c + 1}`}
            />
          ))}
        </div>
      </div>
      <div className="space-y-2">
        <p className="eyebrow">PCA components (ember adds, frost subtracts)</p>
        <div className="flex flex-wrap gap-2">
          {Array.from({ length: shown }, (_, c) => (
            <PixelCanvas
              key={c}
              values={Array.from(run.pca.components.data.slice(c * 256, (c + 1) * 256))}
              side={16}
              size={px}
              mode="diverging"
              scale={0.25}
              ariaLabel={`PCA component ${c + 1}`}
            />
          ))}
        </div>
      </div>
      {k > shown && (
        <p className="text-[0.74rem] text-faint">
          Showing the first {shown} of {k}.
        </p>
      )}
      <p className="text-sm text-muted">
        NMF&apos;s parts look like strokes, loops and blobs because every pixel of a digit is a sum
        of positive contributions: a part that darkened a pixel where the digit is light would have
        nothing to cancel it. {pct(run.pcaNegative)} of the PCA entries are negative, so those
        components are mixtures of &ldquo;add here, remove there&rdquo; that are hard to read as
        pieces.
      </p>
    </div>
  );
}

function Recon({ run }: { run: NmfRun }) {
  const palette = usePalette();
  const [index, setIndex] = useState(0);
  const i = Math.min(index, run.data.rows - 1);
  const k = run.scenario.components;
  const original = Array.from(run.data.data.slice(i * 256, (i + 1) * 256));
  const nmf = useMemo(() => {
    const out = new Float64Array(256);
    for (let c = 0; c < k; c++) {
      const w = run.model.w.get(i, c);
      for (let j = 0; j < 256; j++) out[j] = (out[j] as number) + w * run.model.h.get(c, j);
    }
    return Array.from(out, (v) => Math.min(1, Math.max(0, v)));
  }, [run, i, k]);
  const pcaBack = useMemo(() => {
    const row = run.data.data.slice(i * 256, (i + 1) * 256);
    const rowMatrix = {
      ...run.data,
      rows: 1,
      data: row,
      get: (_r: number, c: number) => row[c] as number,
    };
    const back = run.pca.inverseTransform(run.pca.transform(rowMatrix as never));
    return Array.from(back.data, (v) => Math.min(1, Math.max(0, v)));
  }, [run, i]);
  const weights = Array.from({ length: k }, (_, c) => run.model.w.get(i, c));
  const wMax = Math.max(...weights, 1e-9);
  return (
    <div className="space-y-5">
      <Slider label="Digit" min={0} max={run.data.rows - 1} value={i} onValueChange={setIndex} />
      <div className="flex flex-wrap items-end gap-8">
        {(
          [
            ['original', original],
            [`NMF, ${k} parts`, nmf],
            [`PCA, ${k} components`, pcaBack],
          ] as const
        ).map(([label, values]) => (
          <figure key={label} className="space-y-1">
            <PixelCanvas values={values as number[]} side={16} size={110} ariaLabel={label} />
            <figcaption className="text-center font-mono text-[0.7rem] text-faint">
              {label}
            </figcaption>
          </figure>
        ))}
      </div>
      <div className="space-y-1">
        <p className="eyebrow">How much of each part this digit uses</p>
        <div
          className="flex h-16 items-end gap-1"
          role="img"
          aria-label="Weights of the parts used by this digit"
        >
          {weights.map((w, c) => (
            <div
              key={c}
              className="min-w-1 flex-1 rounded-t"
              style={{ height: `${Math.max(2, (w / wMax) * 100)}%`, background: palette.css.brand }}
              title={`part ${c + 1}: ${fix(w, 2)}`}
            />
          ))}
        </div>
      </div>
      <p className="text-sm text-muted">
        A digit is a recipe: a few parts with large weights, the rest nearly zero. Because weights
        are non-negative the picture is built purely by <em>adding</em> parts together.
      </p>
    </div>
  );
}

export default function NmfPage() {
  const [params, setParams] = useState<ParamValues>(() => fromScenario(DEFAULT_NMF));
  const [activeId, setActiveId] = useState<string | null>(null);
  const [run, setRun] = useState<NmfRun | null>(null);
  const scenario = useMemo(() => toScenario(params), [params]);

  // fitting takes a moment: let the controls repaint first, then compute
  useEffect(() => {
    const timer = window.setTimeout(() => setRun(runNmf(scenario)), 20);
    return () => window.clearTimeout(timer);
  }, [scenario]);

  const setParam = (key: string, value: ParamValue) => setParams((p) => ({ ...p, [key]: value }));
  const apply = (exp: NmfExperiment) => {
    setParams(fromScenario({ ...DEFAULT_NMF, ...exp.scenario }));
    setActiveId(exp.id);
  };
  const stillActive = (() => {
    const exp = nmfExperiments.find((e) => e.id === activeId);
    return (
      !!exp &&
      JSON.stringify(fromScenario({ ...DEFAULT_NMF, ...exp.scenario })) === JSON.stringify(params)
    );
  })();

  const current = run && run.scenario === scenario ? run : null;
  const metrics: Metric[] = run
    ? [
        {
          label: 'NMF error',
          value: pct(run.nmfRelError),
          testId: 'nmf-error',
          hint: 'share of the picture’s size left unexplained (relative reconstruction error)',
        },
        {
          label: 'PCA error (same size)',
          value: pct(run.pcaRelError),
          hint: 'the same measure for PCA with as many components',
        },
        {
          label: 'NMF sparsity',
          value: fix(run.nmfSparsity, 2),
          hint: 'Hoyer sparsity of the parts: 0 = every pixel equal, 1 = one pixel',
        },
        { label: 'PCA sparsity', value: fix(run.pcaSparsity, 2) },
      ]
    : [];

  const stages: [string, string][] = [
    ['lab-parts', 'The parts'],
    ['lab-recon', 'Rebuilding a digit'],
    ['lab-curve', 'Convergence'],
    ['lab-how', 'How it works'],
    ['lab-controls', 'Settings'],
  ];

  return (
    <div className="mx-auto max-w-6xl px-4 pt-8 pb-20 sm:px-6 lg:px-8">
      <LabHeader
        eyebrow="Classic · unsupervised"
        title="Non-negative matrix factorisation"
        accent="parts that only add"
        blurb="Write every picture as a sum of non-negative parts. Since nothing can be subtracted, the parts that come out are strokes you can recognise, unlike the signed components of PCA."
        badges={
          <>
            <Badge>{run ? `${run.data.rows} digits` : 'digits'}</Badge>
            <Badge>{scenario.components} parts</Badge>
          </>
        }
      />
      <StageNav stages={stages} />
      <ExperimentBar
        lessons={nmfExperiments}
        activeId={stillActive ? activeId : null}
        onApply={apply}
      />

      <div className="flex flex-col gap-5" aria-busy={!current}>
        <Panel id="lab-parts" className="scroll-mt-24" title="The parts it found" eyebrow="01">
          {run ? (
            <div className="space-y-6">
              <MetricsStrip metrics={metrics} className="lg:grid-cols-4" />
              <Parts run={run} />
            </div>
          ) : (
            <p className="text-sm text-muted">Fitting…</p>
          )}
        </Panel>
        <Panel id="lab-recon" className="scroll-mt-24" title="Rebuilding a digit" eyebrow="02">
          {run ? <Recon run={run} /> : <p className="text-sm text-muted">Fitting…</p>}
        </Panel>
        <Panel id="lab-curve" className="scroll-mt-24" title="Convergence" eyebrow="03">
          {run && (
            <div className="grid gap-8 lg:grid-cols-[minmax(0,1fr)_20rem]">
              <MiniChart
                title="Error after each multiplicative update"
                x={run.model.history.map((_, i) => i)}
                series={[{ name: 'error', color: 'var(--brand)', values: run.model.history }]}
                log
                height={200}
              />
              <p className="text-sm text-muted">
                Every update multiplies each entry by a ratio that is positive, so non-negativity is
                kept for free, and the error provably never goes up. It falls fast at first and then
                flattens; unlike PCA, the answer depends on the random start and is only a local
                optimum.
              </p>
            </div>
          )}
        </Panel>
        <Panel id="lab-how" className="scroll-mt-24" title="How it works" eyebrow="04">
          <div className="space-y-3 text-sm text-muted">
            <p>
              A table V (here: pictures × pixels) is approximated as{' '}
              <strong className="text-fg">W · H</strong>, where H holds the <em>parts</em> (one row
              each) and W says how much of each part every picture uses; both are constrained to be
              non-negative.
            </p>
            <p>
              The Lee–Seung updates multiply each entry of H and W by the ratio (WᵀV)/(WᵀWH),
              respectively (VHᵀ)/(WHHᵀ). Choosing the KL divergence instead of squared error gives a
              similar update derived from a Poisson view of the pixel intensities. Because parts
              cannot cancel one another, they tend to be local and sparse.
            </p>
          </div>
        </Panel>
        <Panel id="lab-controls" className="scroll-mt-24" title="Settings" eyebrow="05">
          <ParamPanel specs={nmfParams} values={params} onChange={setParam} columns={3} />
        </Panel>
      </div>
    </div>
  );
}
