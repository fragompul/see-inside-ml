import { Dices } from 'lucide-react';
import { useMemo, useState } from 'react';
import { Badge, Button, Panel } from '@/design-system';
import { usePalette } from '@/shared/canvas/colors';
import { ExperimentBar } from '@/shared/components/ExperimentBar';
import { MiniChart } from '@/shared/components/MiniChart';
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
import { DEFAULT_HMM, runHmm, type HmmRun, type HmmScenario } from './hmmCore';
import { hmmExperiments, hmmParams, type HmmExperiment } from './hmmSpec';

const toScenario = (p: ParamValues, seed: number): HmmScenario => ({
  stay: Number(p.stay),
  sixes: Number(p.sixes),
  length: Number(p.length),
  inference: p.inference as HmmScenario['inference'],
  iterations: Number(p.iterations),
  seed,
});

const fromScenario = (s: HmmScenario): ParamValues => ({
  stay: s.stay,
  sixes: s.sixes,
  length: s.length,
  inference: s.inference,
  iterations: s.iterations,
});

function Timeline({ run }: { run: HmmRun }) {
  const palette = usePalette();
  const n = run.rolls.length;
  const w = 960;
  const cell = w / n;
  const labelled = n <= 120;
  const rows = { truth: 6, rolls: 34, viterbi: 78, post: 112 };
  const height = 190;
  const band = (states: number[], y: number, label: string) => (
    <g>
      {states.map((s, i) => (
        <rect
          key={i}
          x={i * cell}
          y={y}
          width={Math.ceil(cell * 100) / 100 + 0.3}
          height={18}
          fill={s === 1 ? palette.css.neg : palette.css.pos}
          opacity={0.75}
        />
      ))}
      <title>{label}</title>
    </g>
  );
  const line = run.posterior
    .map((p, i) => `${(i * cell + cell / 2).toFixed(1)},${(rows.post + 54 - p * 54).toFixed(1)}`)
    .join(' ');
  return (
    <div className="space-y-3">
      <div className="overflow-x-auto">
        <svg
          viewBox={`0 0 ${w} ${height}`}
          className="min-w-[40rem] w-full"
          role="img"
          aria-label="The true die, the rolls, the Viterbi path and the probability that the loaded die is in use, along the sequence"
        >
          <text x={0} y={rows.truth - 1} className="fill-faint text-[9px]">
            true die (hidden)
          </text>
          {band(run.states, rows.truth + 3, 'true die')}
          <text x={0} y={rows.rolls - 1} className="fill-faint text-[9px]">
            rolls (what you see)
          </text>
          {run.rolls.map((r, i) => (
            <g key={i}>
              <rect
                x={i * cell}
                y={rows.rolls + 3}
                width={cell}
                height={30}
                fill={r === 6 ? palette.css.neg : palette.css.muted}
                opacity={r === 6 ? 0.55 : 0.08 + 0.03 * r}
              />
              {labelled && (
                <text
                  x={i * cell + cell / 2}
                  y={rows.rolls + 23}
                  textAnchor="middle"
                  className="fill-fg font-mono text-[10px]"
                >
                  {r}
                </text>
              )}
            </g>
          ))}
          <text x={0} y={rows.viterbi - 1} className="fill-faint text-[9px]">
            most likely path (Viterbi)
          </text>
          {band(run.viterbi, rows.viterbi + 3, 'Viterbi path')}
          <text x={0} y={rows.post - 1} className="fill-faint text-[9px]">
            probability the loaded die is in use
          </text>
          <line
            x1={0}
            x2={w}
            y1={rows.post + 27}
            y2={rows.post + 27}
            stroke="currentColor"
            opacity={0.15}
            strokeDasharray="4 4"
          />
          <polyline points={line} fill="none" stroke={palette.css.brand} strokeWidth={1.6} />
        </svg>
      </div>
      <p className="text-sm text-muted" data-testid="timeline-note">
        <span className="text-pos">Green</span> = fair die, <span className="text-neg">red</span> =
        loaded. The top band is the secret truth; the filter only sees the numbers in the second row
        (sixes are tinted). The bottom curve is the smoothed probability of the loaded die given{' '}
        <em>all</em> the rolls, the band above it the single most likely whole path. They agree on
        the clear stretches and can differ near a switch, because Viterbi must keep one consistent
        story while the curve averages over all of them.
      </p>
    </div>
  );
}

function Matrix({
  title,
  rows,
  labels,
  cols,
}: {
  title: string;
  rows: number[][];
  labels: string[];
  cols: string[];
}) {
  return (
    <div className="overflow-x-auto">
      <p className="eyebrow mb-1">{title}</p>
      <table className="font-mono text-[0.75rem] tabular-nums">
        <thead className="text-faint">
          <tr>
            <th />
            {cols.map((c) => (
              <th key={c} className="px-2 pb-1 font-normal">
                {c}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((r, i) => (
            <tr key={labels[i]} className="border-t border-line">
              <td className="py-1 pr-3 font-sans text-muted">{labels[i]}</td>
              {r.map((v, j) => (
                <td key={j} className="px-2">
                  <span className="relative block min-w-9 rounded px-1 text-right">
                    <span
                      className="absolute inset-y-0 left-0 rounded bg-brand/25"
                      style={{ width: `${v * 100}%` }}
                    />
                    <span className="relative">{v.toFixed(2)}</span>
                  </span>
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function ModelView({ run }: { run: HmmRun }) {
  const states = ['fair', 'loaded'];
  const faces = ['1', '2', '3', '4', '5', '6'];
  const learned = run.scenario.inference === 'learned';
  // a learned model may have called the states the other way round: show them in the order that matches the truth
  const swap = learned && run.model.emission[0]![5]! > run.model.emission[1]![5]!;
  const order = swap ? [1, 0] : [0, 1];
  const reorder = (m: number[][], cols?: boolean) =>
    order.map((i) => (cols ? order.map((j) => m[i]![j]!) : m[i]!));
  return (
    <div className="space-y-5">
      <div className="grid gap-x-10 gap-y-6 lg:grid-cols-2">
        <Matrix
          title="Model: how the die changes"
          rows={reorder(run.model.transition, true)}
          labels={states}
          cols={states}
        />
        <Matrix
          title="Truth: how the die changes"
          rows={run.truth.transition}
          labels={states}
          cols={states}
        />
        <Matrix
          title="Model: faces of each die"
          rows={reorder(run.model.emission)}
          labels={states}
          cols={faces}
        />
        <Matrix
          title="Truth: faces of each die"
          rows={run.truth.emission}
          labels={states}
          cols={faces}
        />
      </div>
      <p className="text-sm text-muted">
        The top-left pair says how likely each die is to be replaced by the other; the bottom pair
        how often each die shows each face.{' '}
        {run.scenario.inference === 'truth' &&
          'Here the model is the truth, so the two sides match.'}
        {run.scenario.inference === 'naive' &&
          'The naive model expects far more swaps and a much weaker bias than the real croupier.'}
        {learned &&
          'The learned model was found from the rolls alone; with few rolls it drifts from the truth, with many it gets close.'}
      </p>
    </div>
  );
}

function Learning({ run }: { run: HmmRun }) {
  const palette = usePalette();
  if (!run.learning) {
    return (
      <p className="text-sm text-muted">
        Choose &ldquo;Learned (Baum-Welch)&rdquo; under Settings to watch the model improve its own
        likelihood from a naive starting guess.
      </p>
    );
  }
  const ll = run.learning.map((s) => s.logLikelihood);
  return (
    <div className="grid gap-8 lg:grid-cols-[minmax(0,1fr)_20rem]">
      <MiniChart
        title="Log-likelihood of the rolls under the model being learned"
        x={ll.map((_, i) => i)}
        series={[
          { name: 'learned model', color: palette.css.brand, values: ll },
          {
            name: 'true model',
            color: palette.css.muted,
            values: ll.map(() => run.trueLogLikelihood),
            dashed: true,
          },
        ]}
        height={200}
      />
      <div className="space-y-3 text-sm text-muted" data-testid="learning-note">
        <p>
          Each Baum-Welch step (an EM step) first works out, with the current model, how responsible
          each die was for each roll, then re-estimates the model from those soft counts. The
          likelihood can only go up.
        </p>
        <p>
          It ends at <strong className="text-fg">{fix(ll[ll.length - 1], 1)}</strong>, the true
          model scores <strong className="text-fg">{fix(run.trueLogLikelihood, 1)}</strong>.{' '}
          {(ll[ll.length - 1] as number) > run.trueLogLikelihood
            ? 'The learned model explains these particular rolls better than the real one: a sign of fitting the noise.'
            : 'The learned model is no better than the truth, as it should be with this much data.'}
        </p>
      </div>
    </div>
  );
}

export default function HmmPage() {
  const [params, setParams] = useState<ParamValues>(() => fromScenario(DEFAULT_HMM));
  const [seed, setSeed] = useState(DEFAULT_HMM.seed);
  const [activeId, setActiveId] = useState<string | null>(null);
  const scenario = useMemo(() => toScenario(params, seed), [params, seed]);
  const run = useMemo(() => runHmm(scenario), [scenario]);
  const setParam = (key: string, value: ParamValue) => setParams((p) => ({ ...p, [key]: value }));

  const apply = (exp: HmmExperiment) => {
    setParams(fromScenario({ ...DEFAULT_HMM, ...exp.scenario }));
    setSeed(DEFAULT_HMM.seed);
    setActiveId(exp.id);
  };
  const stillActive = (() => {
    const exp = hmmExperiments.find((e) => e.id === activeId);
    return (
      !!exp &&
      seed === DEFAULT_HMM.seed &&
      JSON.stringify(fromScenario({ ...DEFAULT_HMM, ...exp.scenario })) === JSON.stringify(params)
    );
  })();

  const metrics: Metric[] = [
    {
      label: 'Most likely path: correct',
      value: pct(run.viterbiAccuracy),
      testId: 'viterbi',
      hint: 'share of rolls whose die the Viterbi path names correctly',
    },
    {
      label: 'Per-roll best guess: correct',
      value: pct(run.posteriorAccuracy),
      hint: 'share of rolls where the more probable die (smoothed) is the right one',
    },
    {
      label: 'Log-likelihood of the rolls',
      value: fix(run.logLikelihood, 1),
      hint: 'how probable the rolls are under the model used',
    },
    { label: 'Under the true model', value: fix(run.trueLogLikelihood, 1) },
  ];

  const stages: [string, string][] = [
    ['lab-timeline', 'The sequence'],
    ['lab-model', 'The model'],
    ['lab-learning', 'Learning'],
    ['lab-how', 'How it works'],
    ['lab-controls', 'Settings'],
  ];

  return (
    <div className="mx-auto max-w-6xl px-4 pt-8 pb-20 sm:px-6 lg:px-8">
      <LabHeader
        eyebrow="Classic · probabilistic"
        title="Hidden Markov models"
        accent="the dishonest casino"
        blurb="A croupier secretly swaps between a fair die and a loaded one. You only see the rolls. Work out which die was used when, how likely the sequence is, and whether the dice can be learned from the rolls alone."
        badges={
          <>
            <Badge>{scenario.length} rolls</Badge>
            <Badge>
              {scenario.inference === 'truth'
                ? 'true model'
                : scenario.inference === 'naive'
                  ? 'naive model'
                  : 'learned model'}
            </Badge>
          </>
        }
      />
      <StageNav stages={stages} />
      <ExperimentBar
        lessons={hmmExperiments}
        activeId={stillActive ? activeId : null}
        onApply={apply}
      />

      <div className="flex flex-col gap-5">
        <Panel id="lab-timeline" className="scroll-mt-24" title="Reading the rolls" eyebrow="01">
          <div className="space-y-6">
            <MetricsStrip metrics={metrics} />
            <Timeline run={run} />
          </div>
        </Panel>
        <Panel id="lab-model" className="scroll-mt-24" title="Dice and switches" eyebrow="02">
          <ModelView run={run} />
        </Panel>
        <Panel id="lab-learning" className="scroll-mt-24" title="Learning the model" eyebrow="03">
          <Learning run={run} />
        </Panel>
        <Panel id="lab-how" className="scroll-mt-24" title="How it works" eyebrow="04">
          <div className="space-y-3 text-sm text-muted">
            <p>
              A hidden Markov model has a hidden state that jumps between a few values according to
              a transition matrix, and each state emits an observation. Because the state at one
              step only depends on the previous one, everything can be computed by{' '}
              <strong className="text-fg">dynamic programming</strong> instead of summing over the
              exponentially many state paths.
            </p>
            <p>
              The <strong className="text-fg">forward algorithm</strong> carries the probability of
              each state given the rolls so far and yields the likelihood of the whole sequence. Add
              a backward pass and you get <strong className="text-fg">smoothing</strong>, the
              probability of each state given all the rolls.{' '}
              <strong className="text-fg">Viterbi</strong> replaces sums by maxima to find the
              single best path. <strong className="text-fg">Baum-Welch</strong> is EM: smoothing
              gives soft counts of transitions and emissions, and re-normalising them gives a better
              model.
            </p>
            <p>
              Tests check the forward likelihood, Viterbi path and smoothed marginals against brute
              force enumeration of every path on small examples.
            </p>
          </div>
        </Panel>
        <Panel id="lab-controls" className="scroll-mt-24" title="Settings" eyebrow="05">
          <div className="space-y-6">
            <ParamPanel specs={hmmParams} values={params} onChange={setParam} columns={3} />
            <Button
              size="sm"
              variant="secondary"
              onClick={() => setSeed(Math.floor(Math.random() * 1e6))}
            >
              <Dices className="size-4" /> New croupier
            </Button>
          </div>
        </Panel>
      </div>
    </div>
  );
}
