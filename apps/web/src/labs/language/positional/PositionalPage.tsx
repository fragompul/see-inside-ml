import { useMemo, useState } from 'react';
import { Badge, Panel } from '@/design-system';
import { usePalette } from '@/shared/canvas/colors';
import { ExperimentBar } from '@/shared/components/ExperimentBar';
import { Heat } from '@/shared/components/Heat';
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
import { DEFAULT_POSITIONAL, runPositional, type PositionalScenario } from './positionalCore';
import {
  SCHEME_LABEL,
  positionalExperiments,
  positionalParams,
  type PositionalExperiment,
} from './positionalSpec';

const toScenario = (p: ParamValues): PositionalScenario => ({
  scheme: p.scheme as PositionalScenario['scheme'],
  dim: Number(p.dim),
  length: Number(p.length),
  trained: Number(p.trained),
});

const fromScenario = (s: PositionalScenario): ParamValues => ({ ...s });

export default function PositionalPage() {
  const palette = usePalette();
  const [params, setParams] = useState<ParamValues>(() => fromScenario(DEFAULT_POSITIONAL));
  const [activeId, setActiveId] = useState<string | null>(null);
  const scenario = useMemo(() => toScenario(params), [params]);
  const run = useMemo(() => runPositional(scenario), [scenario]);
  const setParam = (key: string, value: ParamValue) => setParams((p) => ({ ...p, [key]: value }));

  const apply = (exp: PositionalExperiment) => {
    setParams(fromScenario({ ...DEFAULT_POSITIONAL, ...exp.scenario }));
    setActiveId(exp.id);
  };
  const stillActive = (() => {
    const exp = positionalExperiments.find((e) => e.id === activeId);
    return (
      !!exp &&
      JSON.stringify(fromScenario({ ...DEFAULT_POSITIONAL, ...exp.scenario })) ===
        JSON.stringify(params)
    );
  })();

  const alibi = scenario.scheme === 'alibi';
  const cells = useMemo(
    () => Array.from(run.matrix.values, (v) => (Number.isNaN(v) ? 0 : v)),
    [run],
  );
  const scale = alibi ? Math.max(...cells.map(Math.abs), 1e-9) : 1;
  const cell = Math.max(4, Math.floor(480 / scenario.length));
  const known = run.byOffset.map((v) => (Number.isNaN(v) ? 0 : v));

  const metrics: Metric[] = [
    {
      label: 'Pairs with a score',
      value: `${fix(run.definedShare * 100, 0)}%`,
      testId: 'defined-share',
      hint: alibi ? 'of the pairs a causal model uses' : undefined,
    },
    { label: 'Close (1–4 apart)', value: fix(run.near, 2) },
    { label: 'Medium (10–20 apart)', value: fix(run.mid, 2) },
    {
      label: 'Far (30 or more)',
      value: fix(run.far, 2),
      hint: 'scores are similarity, or a penalty for ALiBi',
    },
  ];

  const stages: [string, string][] = [
    ['lab-matrix', 'Position by position'],
    ['lab-offset', 'Score by distance'],
    ['lab-how', 'How it works'],
    ['lab-controls', 'Settings'],
  ];

  return (
    <div className="mx-auto max-w-6xl px-4 pt-8 pb-20 sm:px-6 lg:px-8">
      <LabHeader
        eyebrow="Language models"
        title="Position encodings"
        accent="where a word sits"
        blurb="Attention treats a text as a bag of vectors; without help, shuffling the words changes nothing. Position encodings put order back. Compare four ways of doing it on text where every word is the same, so that the only thing left to see is position."
        badges={
          <>
            <Badge>{SCHEME_LABEL[scenario.scheme]}</Badge>
            <Badge>{scenario.length} positions</Badge>
          </>
        }
      />
      <StageNav stages={stages} />
      <ExperimentBar
        lessons={positionalExperiments}
        activeId={stillActive ? activeId : null}
        onApply={apply}
      />

      <div className="flex flex-col gap-5">
        <Panel id="lab-matrix" className="scroll-mt-24" title="Position by position" eyebrow="01">
          <div className="space-y-6">
            <MetricsStrip metrics={metrics} className="lg:grid-cols-4" />
            <div className="overflow-x-auto">
              <Heat
                values={cells}
                cols={scenario.length}
                scale={scale}
                cellW={cell}
                cellH={cell}
                ariaLabel="How position i relates to position j"
              />
            </div>
            <p className="text-sm text-muted">
              Rows and columns are positions. Warm cells mean the two positions look alike (or, for
              ALiBi, are penalised little); cool cells mean the opposite. Cells with no score, such
              as the future in ALiBi or anything past a learned table, are left blank.
            </p>
          </div>
        </Panel>
        <Panel id="lab-offset" className="scroll-mt-24" title="Score by distance" eyebrow="02">
          <MiniChart
            title="Average score for positions this far apart"
            x={known.map((_, i) => i)}
            xLabel="distance between positions"
            series={[
              { name: SCHEME_LABEL[scenario.scheme], color: palette.css.brand, values: known },
            ]}
            height={220}
          />
        </Panel>
        <Panel id="lab-how" className="scroll-mt-24" title="How it works" eyebrow="03">
          <div className="space-y-3 text-sm text-muted">
            <p>
              <strong className="text-fg">Sinusoidal</strong> gives every position a fixed vector
              made of sines and cosines of many frequencies, like the hands of clocks running at
              different speeds, and adds it to the word&rsquo;s embedding.{' '}
              <strong className="text-fg">RoPE</strong> instead rotates each pair of numbers in the
              query and the key by an angle proportional to the position; the dot product of a
              rotated query and key then depends only on the offset between them.
            </p>
            <p>
              <strong className="text-fg">ALiBi</strong> adds nothing to the vectors and instead
              subtracts a penalty proportional to the distance from each attention score, with a
              different slope per head. A <strong className="text-fg">learned table</strong> has a
              trained vector per position, which is flexible but has nothing to say about positions
              it has never seen. The tests check the offset-only property to rounding error and that
              RoPE and the sinusoidal codes agree when content is identical.
            </p>
          </div>
        </Panel>
        <Panel id="lab-controls" className="scroll-mt-24" title="Settings" eyebrow="04">
          <ParamPanel specs={positionalParams} values={params} onChange={setParam} columns={2} />
        </Panel>
      </div>
    </div>
  );
}
