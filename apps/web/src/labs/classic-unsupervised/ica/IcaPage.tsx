import { kurtosis } from '@sim/classic';
import type { Matrix } from '@sim/core';
import { Dices } from 'lucide-react';
import { useMemo, useState } from 'react';
import { Badge, Button, Panel } from '@/design-system';
import { usePalette } from '@/shared/canvas/colors';
import { ExperimentBar } from '@/shared/components/ExperimentBar';
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
import {
  DEFAULT_ICA,
  SOURCE_LABELS,
  SOURCE_NOTES,
  mean,
  runIca,
  worst,
  type IcaRun,
  type IcaScenario,
} from './icaCore';
import { icaExperiments, icaParams, type IcaExperiment } from './icaSpec';

const column = (m: Matrix, c: number): number[] =>
  Array.from({ length: m.rows }, (_, i) => m.get(i, c));

const corr = (a: number[], b: number[]): number => {
  const ma = mean(a);
  const mb = mean(b);
  let num = 0;
  let da = 0;
  let db = 0;
  a.forEach((v, i) => {
    num += (v - ma) * ((b[i] as number) - mb);
    da += (v - ma) ** 2;
    db += ((b[i] as number) - mb) ** 2;
  });
  return num / Math.sqrt(da * db || 1);
};

/** For every true source, the recovered signal that matches it best, sign-flipped so it points the same way. */
function aligned(truth: Matrix, found: Matrix): { values: number[]; r: number }[] {
  return Array.from({ length: truth.cols }, (_, s) => {
    const t = column(truth, s);
    let best = { values: column(found, 0), r: 0 };
    for (let c = 0; c < found.cols; c++) {
      const f = column(found, c);
      const r = corr(t, f);
      if (Math.abs(r) >= Math.abs(best.r)) best = { values: r < 0 ? f.map((v) => -v) : f, r };
    }
    return best;
  });
}

function Trace({ values, color, label }: { values: number[]; color: string; label: string }) {
  const w = 220;
  const h = 56;
  const lim = Math.max(2.2, ...values.map(Math.abs));
  const pts = values
    .map(
      (v, i) =>
        `${((i / (values.length - 1)) * (w - 4) + 2).toFixed(1)},${(h / 2 - (v / lim) * (h / 2 - 3)).toFixed(1)}`,
    )
    .join(' ');
  return (
    <svg viewBox={`0 0 ${w} ${h}`} className="w-full" role="img" aria-label={label}>
      <line x1={0} x2={w} y1={h / 2} y2={h / 2} stroke="currentColor" opacity={0.15} />
      <polyline points={pts} fill="none" stroke={color} strokeWidth={1.4} strokeLinejoin="round" />
    </svg>
  );
}

function Signals({ run }: { run: IcaRun }) {
  const palette = usePalette();
  const ica = useMemo(() => aligned(run.sources, run.ica), [run]);
  const pca = useMemo(() => aligned(run.sources, run.pca), [run]);
  const k = run.scenario.count;
  const head = ['True source', 'What microphone hears', 'PCA output', 'ICA output'];
  return (
    <div className="space-y-4">
      <div className="overflow-x-auto">
        <div className="grid min-w-[44rem] grid-cols-4 gap-x-5 gap-y-2">
          {head.map((h) => (
            <p key={h} className="eyebrow">
              {h}
            </p>
          ))}
          {Array.from({ length: k }, (_, s) => (
            <div key={s} className="contents">
              <Trace
                values={column(run.sources, s)}
                color={palette.css.fg}
                label={`True source ${s + 1}`}
              />
              <Trace
                values={column(run.mixed, s)}
                color={palette.css.muted}
                label={`Microphone ${s + 1}`}
              />
              <div>
                <Trace
                  values={pca[s]!.values}
                  color={palette.css.neg}
                  label={`PCA signal matching source ${s + 1}`}
                />
                <p className="text-right font-mono text-[0.68rem] text-faint">
                  match {fix(Math.abs(pca[s]!.r), 2)}
                </p>
              </div>
              <div>
                <Trace
                  values={ica[s]!.values}
                  color={palette.css.pos}
                  label={`ICA signal matching source ${s + 1}`}
                />
                <p className="text-right font-mono text-[0.68rem] text-faint">
                  match {fix(Math.abs(ica[s]!.r), 2)}
                </p>
              </div>
            </div>
          ))}
        </div>
      </div>
      <p className="text-sm text-muted" data-testid="ica-note">
        Each row follows one true source. ICA and PCA know nothing about it: they only see the
        microphones. &ldquo;Match&rdquo; is the absolute correlation with the true source (1 = a
        perfect copy; ICA cannot know the original order, sign or volume, so rows are matched
        afterwards). PCA looks for uncorrelated directions, and a direction that is uncorrelated is
        still a blend; ICA asks for <em>independent</em> ones, which is much stricter.
      </p>
    </div>
  );
}

function Plane({ run }: { run: IcaRun }) {
  const palette = usePalette();
  const size = 340;
  const pts = Array.from(
    { length: run.mixed.rows },
    (_, i) => [run.mixed.get(i, 0), run.mixed.get(i, 1)] as const,
  );
  const lim = Math.max(...pts.flat().map(Math.abs)) * 1.1 || 1;
  const px = (v: number) => size / 2 + (v / lim) * (size / 2 - 8);
  const py = (v: number) => size / 2 - (v / lim) * (size / 2 - 8);
  const axes: { dx: number; dy: number; color: string; dash: string; name: string }[] = [];
  const two = run.scenario.count === 2;
  if (two) {
    // ICA directions in the observation plane are the columns of the inverse of the unmixing matrix
    const c = run.model.components;
    const det = c.get(0, 0) * c.get(1, 1) - c.get(0, 1) * c.get(1, 0);
    const inv = [
      [c.get(1, 1) / det, -c.get(0, 1) / det],
      [-c.get(1, 0) / det, c.get(0, 0) / det],
    ] as const;
    for (let j = 0; j < 2; j++) {
      const dx = inv[0][j] as number;
      const dy = inv[1][j] as number;
      const len = Math.hypot(dx, dy) || 1;
      axes.push({ dx: dx / len, dy: dy / len, color: palette.css.pos, dash: '', name: 'ICA axes' });
    }
    // PCA axes: principal directions of the cloud
    let sxx = 0;
    let sxy = 0;
    let syy = 0;
    pts.forEach(([x, y]) => {
      sxx += x * x;
      sxy += x * y;
      syy += y * y;
    });
    const theta = 0.5 * Math.atan2(2 * sxy, sxx - syy);
    axes.push({
      dx: Math.cos(theta),
      dy: Math.sin(theta),
      color: palette.css.neg,
      dash: '5 4',
      name: 'PCA axes',
    });
    axes.push({
      dx: -Math.sin(theta),
      dy: Math.cos(theta),
      color: palette.css.neg,
      dash: '5 4',
      name: 'PCA axes',
    });
  }
  return (
    <div className="grid gap-8 lg:grid-cols-[auto_minmax(0,1fr)]">
      <svg
        width={size}
        height={size}
        viewBox={`0 0 ${size} ${size}`}
        className="max-w-full"
        role="img"
        aria-label="The two microphone signals plotted against each other, with the PCA and ICA axes"
      >
        <rect
          x={0}
          y={0}
          width={size}
          height={size}
          rx={12}
          fill="none"
          stroke="currentColor"
          opacity={0.12}
        />
        {pts.map(([x, y], i) => (
          <circle key={i} cx={px(x)} cy={py(y)} r={2} fill={palette.css.brand} opacity={0.5} />
        ))}
        {axes.map((a, i) => (
          <line
            key={i}
            x1={px(-a.dx * lim)}
            y1={py(-a.dy * lim)}
            x2={px(a.dx * lim)}
            y2={py(a.dy * lim)}
            stroke={a.color}
            strokeWidth={2}
            strokeDasharray={a.dash}
          />
        ))}
      </svg>
      <div className="space-y-3 text-sm text-muted" data-testid="plane-note">
        {two ? (
          <>
            <p>
              Each dot is one instant: microphone 1 against microphone 2. Independent sources mixed
              linearly fill a <em>tilted</em> shape whose sides point along the mixing directions.
            </p>
            <p>
              <strong className="text-pos">Solid axes (ICA)</strong> follow the shape&apos;s own
              directions, one per source. <strong className="text-neg">Dashed axes (PCA)</strong>{' '}
              are forced to be perpendicular and follow the longest and shortest extent, which for a
              tilted square are the diagonals, not the sides.
            </p>
          </>
        ) : (
          <p>
            With three sources the cloud lives in 3-D; this is a flat view of the first two
            microphones. Switch to two sources to see the axes ICA and PCA choose.
          </p>
        )}
      </div>
    </div>
  );
}

function Gaussianity({ run }: { run: IcaRun }) {
  const rows = [
    ['True sources', run.sources],
    ['Microphones (mixed)', run.mixed],
    ['ICA output', run.ica],
  ] as const;
  return (
    <div className="space-y-4">
      <div className="overflow-x-auto">
        <table className="w-full min-w-[22rem] text-left font-mono text-[0.78rem] tabular-nums">
          <thead className="text-muted">
            <tr>
              <th className="pr-4 pb-2 font-medium">excess kurtosis</th>
              {Array.from({ length: run.scenario.count }, (_, c) => (
                <th key={c} className="pr-4 pb-2 font-medium">
                  signal {c + 1}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map(([name, m]) => (
              <tr key={name} className="border-t border-line">
                <td className="py-1 pr-4 font-sans">{name}</td>
                {Array.from({ length: run.scenario.count }, (_, c) => (
                  <td key={c} className="pr-4">
                    {fix(kurtosis(column(m, c)), 2)}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="text-sm text-muted">
        Excess kurtosis is 0 for a Gaussian. Mixing signals averages them, and by the central limit
        theorem an average drifts towards Gaussian, so the microphones sit closer to 0 than the
        sources. ICA turns this around: it hunts for the rotation that makes the outputs{' '}
        <em>as far from Gaussian as possible</em>, which brings back the original sources. If the
        sources were Gaussian to begin with there is no such rotation to find.
      </p>
    </div>
  );
}

const toScenario = (p: ParamValues, seed: number): IcaScenario => ({
  sources: p.sources as IcaScenario['sources'],
  count: Number(p.count) === 2 ? 2 : 3,
  noise: Number(p.noise),
  nonlinearity: p.nonlinearity as IcaScenario['nonlinearity'],
  samples: Number(p.samples),
  seed,
});

const fromScenario = (s: IcaScenario): ParamValues => ({
  sources: s.sources,
  count: String(s.count),
  noise: s.noise,
  nonlinearity: s.nonlinearity,
  samples: s.samples,
});

export default function IcaPage() {
  const [params, setParams] = useState<ParamValues>(() => fromScenario(DEFAULT_ICA));
  const [seed, setSeed] = useState(DEFAULT_ICA.seed);
  const [activeId, setActiveId] = useState<string | null>(null);

  const scenario = useMemo(() => toScenario(params, seed), [params, seed]);
  const run = useMemo(() => runIca(scenario), [scenario]);
  const setParam = (key: string, value: ParamValue) => setParams((p) => ({ ...p, [key]: value }));

  const apply = (exp: IcaExperiment) => {
    setParams(fromScenario({ ...DEFAULT_ICA, ...exp.scenario }));
    setSeed(DEFAULT_ICA.seed);
    setActiveId(exp.id);
  };
  const stillActive = (() => {
    const exp = icaExperiments.find((e) => e.id === activeId);
    return (
      !!exp &&
      seed === DEFAULT_ICA.seed &&
      JSON.stringify(fromScenario({ ...DEFAULT_ICA, ...exp.scenario })) === JSON.stringify(params)
    );
  })();

  const metrics: Metric[] = [
    {
      label: 'ICA: worst source',
      value: fix(worst(run.icaRecovery), 3),
      testId: 'recovery',
      hint: 'absolute correlation of the worst-recovered true source with its best ICA match (1 = perfect)',
    },
    { label: 'ICA: average', value: fix(mean(run.icaRecovery), 3) },
    {
      label: 'PCA: worst source',
      value: fix(worst(run.pcaRecovery), 3),
      hint: 'the same score for the PCA components',
    },
    { label: 'PCA: average', value: fix(mean(run.pcaRecovery), 3) },
    {
      label: 'Iterations',
      value: `${run.iterations}${run.converged ? '' : ' (not converged)'}`,
      testId: 'iterations',
    },
  ];

  const stages: [string, string][] = [
    ['lab-signals', 'The signals'],
    ['lab-plane', 'In the plane'],
    ['lab-gauss', 'Why it works'],
    ['lab-how', 'How it works'],
    ['lab-controls', 'Settings'],
  ];

  return (
    <div className="mx-auto max-w-6xl px-4 pt-8 pb-20 sm:px-6 lg:px-8">
      <LabHeader
        eyebrow="Classic · unsupervised"
        title="Independent component analysis"
        accent="unmixing the cocktail party"
        blurb="Several signals are blended together by unknown amounts and you only hear the blends. ICA finds the combination that makes the outputs statistically independent, and so recovers the originals. PCA, which only asks for uncorrelated outputs, cannot."
        badges={
          <>
            <Badge>{SOURCE_LABELS[scenario.sources]}</Badge>
            <Badge>{scenario.count} sources</Badge>
            <Badge>{scenario.samples} samples</Badge>
          </>
        }
      />
      <StageNav stages={stages} />
      <ExperimentBar
        lessons={icaExperiments}
        activeId={stillActive ? activeId : null}
        onApply={apply}
      />

      <div className="flex flex-col gap-5">
        <Panel
          id="lab-signals"
          className="scroll-mt-24"
          title="Separating the signals"
          eyebrow="01"
        >
          <div className="space-y-6">
            <p className="text-[0.8rem] text-muted">{SOURCE_NOTES[scenario.sources]}</p>
            <MetricsStrip metrics={metrics} className="lg:grid-cols-5" />
            <Signals run={run} />
          </div>
        </Panel>
        <Panel
          id="lab-plane"
          className="scroll-mt-24"
          title="The mixture in the plane"
          eyebrow="02"
        >
          <Plane run={run} />
        </Panel>
        <Panel
          id="lab-gauss"
          className="scroll-mt-24"
          title="Why it works: non-Gaussianity"
          eyebrow="03"
        >
          <Gaussianity run={run} />
        </Panel>
        <Panel id="lab-how" className="scroll-mt-24" title="How it works" eyebrow="04">
          <div className="space-y-3 text-sm text-muted">
            <p>
              FastICA first <strong className="text-fg">whitens</strong> the data: centre it and
              rescale along its principal axes so that every direction has unit variance and nothing
              is correlated. After whitening, the unmixing is just a <em>rotation</em>.
            </p>
            <p>
              It then looks for the rotation whose outputs are as non-Gaussian as possible, because
              a mixture of independent signals is always closer to Gaussian than the signals
              themselves. A fixed-point update (using a contrast function such as log cosh) moves
              each direction towards the most non-Gaussian projection, and a symmetric decorrelation
              step keeps the directions orthogonal. It typically converges in a handful of
              iterations.
            </p>
            <p>
              What ICA cannot recover: the <em>order</em> of the sources, their <em>sign</em> and
              their <em>volume</em>, and any source that is Gaussian (at most one is allowed).
            </p>
          </div>
        </Panel>
        <Panel id="lab-controls" className="scroll-mt-24" title="Settings" eyebrow="05">
          <div className="space-y-6">
            <ParamPanel specs={icaParams} values={params} onChange={setParam} columns={3} />
            <Button
              size="sm"
              variant="secondary"
              onClick={() => setSeed(Math.floor(Math.random() * 1e6))}
            >
              <Dices className="size-4" /> New mixing
            </Button>
          </div>
        </Panel>
      </div>
    </div>
  );
}
