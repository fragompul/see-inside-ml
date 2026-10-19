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
import { DEFAULT_SURVIVAL, runSurvival, type SurvivalScenario } from './survivalCore';
import { survivalExperiments, survivalParams, type SurvivalExperiment } from './survivalSpec';

const toScenario = (p: ParamValues): SurvivalScenario => ({
  ...DEFAULT_SURVIVAL,
  n: Number(p.n),
  hazardRatio: Number(p.hazardRatio),
  followup: Number(p.followup),
});

const fromScenario = (s: SurvivalScenario): ParamValues => ({
  n: s.n,
  hazardRatio: s.hazardRatio,
  followup: s.followup,
});

const months = (v: number) => (Number.isFinite(v) ? `${fix(v, 1)}` : 'never reached');
const p = (v: number) => (v < 0.001 ? '< 0.001' : fix(v, 3));

export default function SurvivalPage() {
  const palette = usePalette();
  const [params, setParams] = useState<ParamValues>(() => fromScenario(DEFAULT_SURVIVAL));
  const [activeId, setActiveId] = useState<string | null>(null);
  const scenario = useMemo(() => toScenario(params), [params]);
  const run = useMemo(() => runSurvival(scenario), [scenario]);
  const setParam = (key: string, value: ParamValue) => setParams((q) => ({ ...q, [key]: value }));

  const apply = (exp: SurvivalExperiment) => {
    setParams(fromScenario({ ...DEFAULT_SURVIVAL, ...exp.scenario }));
    setActiveId(exp.id);
  };
  const stillActive = (() => {
    const exp = survivalExperiments.find((e) => e.id === activeId);
    return (
      !!exp &&
      JSON.stringify(fromScenario({ ...DEFAULT_SURVIVAL, ...exp.scenario })) ===
        JSON.stringify(params)
    );
  })();

  const metrics: Metric[] = [
    {
      label: 'Subjects censored',
      value: `${fix(run.censored * 100, 0)}%`,
      testId: 'censored',
      hint: 'still event-free when watching ended',
    },
    {
      label: 'Hazard ratio, estimated',
      value: fix(run.cox.hazardRatio[0] as number, 2),
      hint: `95% interval ${fix(run.cox.low[0] as number, 2)} to ${fix(run.cox.high[0] as number, 2)}; true ${run.trueHazardRatio}`,
    },
    {
      label: 'Log-rank test',
      value: `p ${p(run.logRank.pValue)}`,
      hint: 'do the two curves differ?',
    },
  ];
  const stages: [string, string][] = [
    ['lab-curves', 'Survival curves'],
    ['lab-medians', 'Medians and the naive estimate'],
    ['lab-how', 'How it works'],
    ['lab-controls', 'Settings'],
  ];

  return (
    <div className="mx-auto max-w-6xl px-4 pt-8 pb-20 sm:px-6 lg:px-8">
      <LabHeader
        eyebrow="Classic · supervised"
        title="Survival analysis"
        accent="when not everyone has had the event yet"
        blurb="How long until a patient relapses, a machine fails or a customer leaves? Some subjects are only watched for a while and may never be seen to have the event. Throwing them away or counting them as events gives the wrong answer; the Kaplan-Meier curve, the log-rank test and Cox regression use them correctly."
        badges={
          <>
            <Badge>{scenario.n} subjects</Badge>
            <Badge>hazard ratio {scenario.hazardRatio}</Badge>
          </>
        }
      />
      <StageNav stages={stages} />
      <ExperimentBar
        lessons={survivalExperiments}
        activeId={stillActive ? activeId : null}
        onApply={apply}
      />

      <div className="flex flex-col gap-5">
        <Panel id="lab-curves" className="scroll-mt-24" title="Survival curves" eyebrow="01">
          <div className="space-y-6">
            <MetricsStrip metrics={metrics} className="lg:grid-cols-3" />
            <MiniChart
              title="Share still event-free by time (Kaplan-Meier, with the true curves dashed)"
              x={run.grid}
              xLabel="months"
              series={[
                {
                  name: 'group 1, estimated',
                  color: palette.css.brand,
                  values: (run.curves[0] as { estimate: number[] }).estimate,
                },
                {
                  name: 'group 2, estimated',
                  color: palette.css.neg,
                  values: (run.curves[1] as { estimate: number[] }).estimate,
                },
                {
                  name: 'group 1, true',
                  color: palette.css.brand,
                  values: (run.curves[0] as { truth: number[] }).truth,
                  dashed: true,
                },
                {
                  name: 'group 2, true',
                  color: palette.css.neg,
                  values: (run.curves[1] as { truth: number[] }).truth,
                  dashed: true,
                },
              ]}
              height={260}
            />
          </div>
        </Panel>
        <Panel
          id="lab-medians"
          className="scroll-mt-24"
          title="Medians and the naive estimate"
          eyebrow="02"
        >
          <div className="space-y-3">
            <div className="overflow-x-auto">
              <table className="w-full text-left text-sm">
                <thead className="text-muted">
                  <tr>
                    <th className="py-2 pr-4 font-medium">Median survival (months)</th>
                    <th className="py-2 pr-4 font-medium">True</th>
                    <th className="py-2 pr-4 font-medium">Kaplan-Meier</th>
                    <th className="py-2 font-medium">Naive: median of the observed times</th>
                  </tr>
                </thead>
                <tbody>
                  {run.median.map((m, g) => (
                    <tr key={g} className="border-t border-line">
                      <td className="py-2 pr-4">Group {g + 1}</td>
                      <td className="py-2 pr-4 font-mono">{months(m.truth)}</td>
                      <td className="py-2 pr-4 font-mono" data-testid={`km-median-${g + 1}`}>
                        {months(m.estimate)}
                      </td>
                      <td className="py-2 font-mono">{months(m.naive)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <p className="text-sm text-muted">
              The naive column takes every observed time, censored or not, as the time of an event.
              It is always too short when there is censoring, because someone who was still well
              when watching ended counts as if they had just fallen ill.
            </p>
          </div>
        </Panel>
        <Panel id="lab-how" className="scroll-mt-24" title="How it works" eyebrow="03">
          <div className="space-y-3 text-sm text-muted">
            <p>
              <strong className="text-fg">Kaplan-Meier.</strong> At each time at which someone has
              the event, the survival curve is multiplied by the share of people still being watched
              who did not have it. People who are censored leave the count at the time they leave
              but never cause a drop. <strong className="text-fg">Log-rank test.</strong> At every
              event time it compares the number of events seen in a group with the number expected
              if the groups were the same, and adds up the differences.
            </p>
            <p>
              <strong className="text-fg">Cox regression.</strong> The hazard is the instantaneous
              risk of the event; the model says that for a given subject it is a common baseline
              multiplied by exp(x·β), and the baseline is never specified. β is found by maximising
              a likelihood in which each event is compared with everybody still at risk at that
              time. The tests check all three against statsmodels, and the Cox estimate against a
              known ratio in a large simulation.
            </p>
          </div>
        </Panel>
        <Panel id="lab-controls" className="scroll-mt-24" title="Settings" eyebrow="04">
          <ParamPanel specs={survivalParams} values={params} onChange={setParam} columns={2} />
        </Panel>
      </div>
    </div>
  );
}
