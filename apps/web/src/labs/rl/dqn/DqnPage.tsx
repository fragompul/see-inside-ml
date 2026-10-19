import { useEffect, useMemo, useState } from 'react';
import { Badge, Panel } from '@/design-system';
import { usePalette } from '@/shared/canvas/colors';
import { compute } from '@/shared/compute/client';
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
import { CONFIGS, DEFAULT_DQN_LAB, SEEDS, type ConfigResult, type DqnScenario } from './dqnCore';
import { dqnExperiments, dqnParams, type DqnExperiment } from './dqnSpec';

const toScenario = (p: ParamValues): DqnScenario => ({
  layout: p.layout as DqnScenario['layout'],
  episodes: Number(p.episodes),
  targetEvery: Number(p.targetEvery),
});

const fromScenario = (s: DqnScenario): ParamValues => ({ ...s });

export default function DqnPage() {
  const palette = usePalette();
  const colors = [palette.css.brand, palette.css.neg, palette.css.pos, palette.css.muted];
  const [params, setParams] = useState<ParamValues>(() => fromScenario(DEFAULT_DQN_LAB));
  const [activeId, setActiveId] = useState<string | null>(null);
  const scenario = useMemo(() => toScenario(params), [params]);
  const scenarioKey = JSON.stringify(scenario);
  const [store, setStore] = useState<{ key: string; byId: Record<string, ConfigResult> }>({
    key: '',
    byId: {},
  });
  // results of an earlier scenario are ignored until the new ones arrive
  const results = store.key === scenarioKey ? store.byId : {};
  const setParam = (key: string, value: ParamValue) => setParams((p) => ({ ...p, [key]: value }));

  // each configuration trains several networks: the worker does them one after another
  useEffect(() => {
    let cancelled = false;
    void (async () => {
      for (const config of CONFIGS) {
        if (cancelled) return;
        const result = await compute('dqn-config', { scenario, id: config.id }).catch(() => null);
        if (cancelled || !result) return;
        setStore((prev) => ({
          key: scenarioKey,
          byId: { ...(prev.key === scenarioKey ? prev.byId : {}), [config.id]: result },
        }));
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [scenario, scenarioKey]);

  const apply = (exp: DqnExperiment) => {
    setParams(fromScenario({ ...DEFAULT_DQN_LAB, ...exp.scenario }));
    setActiveId(exp.id);
  };
  const stillActive = (() => {
    const exp = dqnExperiments.find((e) => e.id === activeId);
    return (
      !!exp &&
      JSON.stringify(fromScenario({ ...DEFAULT_DQN_LAB, ...exp.scenario })) ===
        JSON.stringify(params)
    );
  })();

  const done = CONFIGS.filter((c) => results[c.id]);
  const full = results.full;
  const metrics: Metric[] = [
    {
      label: 'Runs that reach the goal (full method)',
      value: full ? `${full.solved} of ${SEEDS}` : '…',
      testId: 'solved',
    },
    {
      label: 'Return, full method',
      value: full ? fix(full.tail, 2) : '…',
      hint: 'last 30 episodes',
    },
    {
      label: 'Largest value predicted, full',
      value: full ? fix(full.maxQ, 1) : '…',
      hint: 'no reward is above 1',
    },
    {
      label: 'Largest value predicted, no target',
      value: results['no-target'] ? fix(results['no-target'].maxQ, 1) : '…',
    },
  ];

  const stages: [string, string][] = [
    ['lab-curves', 'Learning curves'],
    ['lab-table', 'Four ways to train'],
    ['lab-how', 'How it works'],
    ['lab-controls', 'Settings'],
  ];

  return (
    <div className="mx-auto max-w-6xl px-4 pt-8 pb-20 sm:px-6 lg:px-8">
      <LabHeader
        eyebrow="Reinforcement learning"
        title="Deep Q-learning"
        accent="a network chasing its own tail"
        blurb="Replace the Q-table with a neural network and learning becomes unstable: consecutive steps are nearly identical, and the network is trained towards targets that it produces itself. Replay and a target network are the two fixes. Switch each off and watch what breaks."
        badges={
          <>
            <Badge>{scenario.layout === 'open' ? 'Open room' : 'Maze'}</Badge>
            <Badge>{SEEDS} runs per curve</Badge>
          </>
        }
      />
      <StageNav stages={stages} />
      <ExperimentBar
        lessons={dqnExperiments}
        activeId={stillActive ? activeId : null}
        onApply={apply}
      />

      <div className="flex flex-col gap-5">
        <Panel id="lab-curves" className="scroll-mt-24" title="Learning curves" eyebrow="01">
          <div className="space-y-6" aria-busy={done.length < CONFIGS.length}>
            <MetricsStrip metrics={metrics} className="lg:grid-cols-4" />
            {done.length ? (
              <MiniChart
                title="Average return per episode (moving average over 10)"
                x={Array.from({ length: scenario.episodes }, (_, i) => i + 1)}
                xLabel="episode"
                series={done.map((c) => ({
                  name: c.label,
                  color: colors[CONFIGS.indexOf(c)] as string,
                  values: (results[c.id] as ConfigResult).curve,
                  dashed: !c.replay && !c.target,
                }))}
                height={250}
              />
            ) : (
              <p className="text-sm text-muted" role="status">
                Training the networks…
              </p>
            )}
            {done.length < CONFIGS.length && done.length > 0 && (
              <p className="text-sm text-muted" role="status">
                {done.length} of {CONFIGS.length} ways of training finished…
              </p>
            )}
          </div>
        </Panel>
        <Panel id="lab-table" className="scroll-mt-24" title="Four ways to train" eyebrow="02">
          <div className="overflow-x-auto">
            <table className="w-full text-left text-sm">
              <thead className="text-muted">
                <tr>
                  <th className="py-2 pr-4 font-medium">Method</th>
                  <th className="py-2 pr-4 font-medium">Reached the goal</th>
                  <th className="py-2 pr-4 font-medium">Final return</th>
                  <th className="py-2 font-medium">Largest value predicted</th>
                </tr>
              </thead>
              <tbody>
                {CONFIGS.map((c) => {
                  const r = results[c.id];
                  return (
                    <tr key={c.id} className="border-t border-line">
                      <td className="py-2 pr-4">{c.label}</td>
                      <td className="py-2 pr-4 font-mono">{r ? `${r.solved} of ${SEEDS}` : '…'}</td>
                      <td className="py-2 pr-4 font-mono">{r ? fix(r.tail, 2) : '…'}</td>
                      <td className="py-2 font-mono">{r ? fix(r.maxQ, 1) : '…'}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </Panel>
        <Panel id="lab-how" className="scroll-mt-24" title="How it works" eyebrow="03">
          <div className="space-y-3 text-sm text-muted">
            <p>
              A small network takes the position of the agent and predicts how good each of the four
              moves is. After every step it is trained so that its prediction for the move taken
              matches the reward plus the discounted best prediction for the next cell. The
              prediction for the next cell comes from the network itself, so without care the target
              shifts with every update and errors feed on each other: values can grow far beyond
              anything the world can pay.
            </p>
            <p>
              <strong className="text-fg">Replay</strong> stores past steps and trains on random
              batches of them, which breaks the run of near-identical updates.{' '}
              <strong className="text-fg">The target network</strong> is a copy that is refreshed
              only now and then, so the goalposts hold still between refreshes. The backpropagation
              is checked against finite differences, and the Bellman targets against a hand
              calculation, in the tests.
            </p>
          </div>
        </Panel>
        <Panel id="lab-controls" className="scroll-mt-24" title="Settings" eyebrow="04">
          <ParamPanel specs={dqnParams} values={params} onChange={setParam} columns={2} />
        </Panel>
      </div>
    </div>
  );
}
