import { useMemo, useState } from 'react';
import { Badge, Panel } from '@/design-system';
import { usePalette } from '@/shared/canvas/colors';
import { ExperimentBar } from '@/shared/components/ExperimentBar';
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
import {
  DEFAULT_ASSOCIATION,
  ITEMS,
  name,
  runAssociation,
  type AssociationRun,
  type AssociationScenario,
} from './associationCore';
import {
  associationExperiments,
  associationParams,
  type AssociationExperiment,
} from './associationSpec';

const toScenario = (p: ParamValues): AssociationScenario => ({
  ...DEFAULT_ASSOCIATION,
  baskets: Number(p.baskets),
  minSupport: Number(p.minSupport),
  minConfidence: Number(p.minConfidence),
  maxSize: Number(p.maxSize),
  sortBy: p.sortBy as AssociationScenario['sortBy'],
});

const fromScenario = (s: AssociationScenario): ParamValues => ({
  baskets: s.baskets,
  minSupport: s.minSupport,
  minConfidence: s.minConfidence,
  maxSize: String(s.maxSize),
  sortBy: s.sortBy,
});

function RuleTable({ run }: { run: AssociationRun }) {
  const top = run.ranked.slice(0, 12);
  if (!top.length) {
    return (
      <p className="text-sm text-muted" data-testid="no-rules">
        No rule passes both thresholds. Lower the minimum support or confidence.
      </p>
    );
  }
  return (
    <div className="overflow-x-auto">
      <table className="w-full min-w-[34rem] text-left font-mono text-[0.78rem] tabular-nums">
        <thead className="text-muted">
          <tr>
            <th className="pr-4 pb-2 font-sans font-medium">if a basket has…</th>
            <th className="pr-4 pb-2 font-sans font-medium">…it also has</th>
            <th className="pr-4 pb-2 font-medium">support</th>
            <th className="pr-4 pb-2 font-medium">confidence</th>
            <th className="pr-4 pb-2 font-medium">lift</th>
          </tr>
        </thead>
        <tbody>
          {top.map((r, i) => (
            <tr key={i} className="border-t border-line">
              <td className="py-1.5 pr-4 font-sans">{name(r.antecedent)}</td>
              <td className="pr-4 font-sans">{name(r.consequent)}</td>
              <td className="pr-4">{pct(r.support)}</td>
              <td className="pr-4">{pct(r.confidence)}</td>
              <td className={`pr-4 ${r.lift > 1.5 ? 'text-pos' : r.lift < 1 ? 'text-neg' : ''}`}>
                {fix(r.lift, 2)}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      {run.ranked.length > top.length && (
        <p className="mt-2 text-[0.74rem] text-faint">
          Showing the top {top.length} of {run.ranked.length} rules.
        </p>
      )}
    </div>
  );
}

function Scatter({ run }: { run: AssociationRun }) {
  const palette = usePalette();
  const w = 520;
  const h = 300;
  const pad = { l: 40, r: 12, t: 10, b: 30 };
  const maxSupport = Math.max(0.35, ...run.rules.map((r) => r.support));
  const sx = (v: number) => pad.l + (v / maxSupport) * (w - pad.l - pad.r);
  const sy = (v: number) => h - pad.b - v * (h - pad.t - pad.b);
  return (
    <div className="space-y-2">
      <svg
        viewBox={`0 0 ${w} ${h}`}
        className="w-full max-w-xl"
        role="img"
        aria-label="Each rule as a point: support on the horizontal axis, confidence on the vertical axis, coloured by lift"
      >
        {[0, 0.25, 0.5, 0.75, 1].map((t) => (
          <g key={t}>
            <line
              x1={pad.l}
              x2={w - pad.r}
              y1={sy(t)}
              y2={sy(t)}
              stroke="currentColor"
              opacity={0.08}
            />
            <text
              x={pad.l - 6}
              y={sy(t) + 3}
              textAnchor="end"
              className="fill-faint font-mono text-[10px]"
            >
              {Math.round(t * 100)}%
            </text>
          </g>
        ))}
        <text x={w / 2} y={h - 6} textAnchor="middle" className="fill-faint text-[10px]">
          support
        </text>
        {run.rules.map((r, i) => (
          <circle
            key={i}
            cx={sx(r.support)}
            cy={sy(r.confidence)}
            r={4 + Math.min(r.lift, 5)}
            fill={r.lift > 1.5 ? palette.css.pos : r.lift < 1 ? palette.css.neg : palette.css.muted}
            opacity={0.55}
          />
        ))}
      </svg>
      <p className="text-[0.78rem] text-muted">
        <span className="text-pos">Green</span>: lift above 1.5 (a real association),{' '}
        <span className="text-neg">red</span>: below 1 (the items avoid each other), grey: about
        independent. Bigger = larger lift.
      </p>
    </div>
  );
}

function Frequencies({ run }: { run: AssociationRun }) {
  const palette = usePalette();
  return (
    <div className="space-y-2">
      {ITEMS.map((item, i) => (
        <div key={item} className="flex items-center gap-3 text-[0.8rem]">
          <span className="w-16 text-muted">{item}</span>
          <div className="h-2 flex-1 rounded-full bg-raised">
            <div
              className="h-2 rounded-full"
              style={{
                width: `${(run.frequency[i] as number) * 100}%`,
                background: palette.css.brand,
                opacity: 0.8,
              }}
            />
          </div>
          <span className="w-10 text-right font-mono text-faint">{pct(run.frequency[i])}</span>
        </div>
      ))}
      <p className="pt-2 text-sm text-muted">
        How often each item is in a basket. Lift compares a rule&apos;s confidence with exactly this
        base rate of its right-hand side.
      </p>
    </div>
  );
}

export default function AssociationPage() {
  const [params, setParams] = useState<ParamValues>(() => fromScenario(DEFAULT_ASSOCIATION));
  const [activeId, setActiveId] = useState<string | null>(null);
  const scenario = useMemo(() => toScenario(params), [params]);
  const run = useMemo(() => runAssociation(scenario), [scenario]);
  const setParam = (key: string, value: ParamValue) => setParams((p) => ({ ...p, [key]: value }));

  const apply = (exp: AssociationExperiment) => {
    setParams(fromScenario({ ...DEFAULT_ASSOCIATION, ...exp.scenario }));
    setActiveId(exp.id);
  };
  const stillActive = (() => {
    const exp = associationExperiments.find((e) => e.id === activeId);
    return (
      !!exp &&
      JSON.stringify(fromScenario({ ...DEFAULT_ASSOCIATION, ...exp.scenario })) ===
        JSON.stringify(params)
    );
  })();

  const metrics: Metric[] = [
    { label: 'Frequent combinations', value: String(run.itemsets.length), testId: 'itemsets' },
    { label: 'Rules', value: String(run.rules.length), testId: 'rules' },
    { label: 'Real associations (lift > 1.5)', value: String(run.strong) },
    { label: 'Strongest lift', value: run.rules.length ? fix(run.bestLift, 2) : '–' },
    {
      label: 'True habits found',
      value: `${run.recovered} of 4`,
      hint: 'bread–butter, beer–diapers, chips–salsa, wine–cheese',
    },
  ];

  const stages: [string, string][] = [
    ['lab-rules', 'The rules'],
    ['lab-map', 'Support and confidence'],
    ['lab-items', 'What is in the baskets'],
    ['lab-how', 'How it works'],
    ['lab-controls', 'Settings'],
  ];

  return (
    <div className="mx-auto max-w-6xl px-4 pt-8 pb-20 sm:px-6 lg:px-8">
      <LabHeader
        eyebrow="Classic · probabilistic"
        title="Association rules"
        accent="what goes in the same basket"
        blurb="Mine shopping baskets for combinations that appear together more than chance would. Apriori finds the frequent ones; support, confidence and lift decide which are worth believing."
        badges={
          <>
            <Badge>{run.baskets.length} baskets</Badge>
            <Badge>{ITEMS.length} products</Badge>
          </>
        }
      />
      <StageNav stages={stages} />
      <ExperimentBar
        lessons={associationExperiments}
        activeId={stillActive ? activeId : null}
        onApply={apply}
      />

      <div className="flex flex-col gap-5">
        <Panel id="lab-rules" className="scroll-mt-24" title="The rules" eyebrow="01">
          <div className="space-y-6">
            <MetricsStrip metrics={metrics} className="lg:grid-cols-5" />
            <RuleTable run={run} />
          </div>
        </Panel>
        <Panel id="lab-map" className="scroll-mt-24" title="Support and confidence" eyebrow="02">
          <Scatter run={run} />
        </Panel>
        <Panel id="lab-items" className="scroll-mt-24" title="What is in the baskets" eyebrow="03">
          <Frequencies run={run} />
        </Panel>
        <Panel id="lab-how" className="scroll-mt-24" title="How it works" eyebrow="04">
          <div className="space-y-3 text-sm text-muted">
            <p>
              <strong className="text-fg">Support</strong> of a combination is the share of baskets
              that contain all of its items. <strong className="text-fg">Confidence</strong> of a
              rule A → C is the share of A-baskets that also contain C.{' '}
              <strong className="text-fg">Lift</strong> divides that by how common C is overall: 1
              means A tells you nothing about C, 2 means C is twice as likely when A is present.
            </p>
            <p>
              <strong className="text-fg">Apriori</strong> exploits one fact: a combination cannot
              be frequent unless every sub-combination is. It counts single items, keeps the
              frequent ones, joins them into pairs, counts those, and so on, pruning any candidate
              with an infrequent part. Rules are then formed by splitting each frequent set in two.
              Tests check the result against exhaustive counting of every subset.
            </p>
          </div>
        </Panel>
        <Panel id="lab-controls" className="scroll-mt-24" title="Settings" eyebrow="05">
          <ParamPanel specs={associationParams} values={params} onChange={setParam} columns={3} />
        </Panel>
      </div>
    </div>
  );
}
