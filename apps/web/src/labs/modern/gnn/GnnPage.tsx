import { useMemo, useState } from 'react';
import { Badge, Panel } from '@/design-system';
import { usePalette } from '@/shared/canvas/colors';
import { useCompute } from '@/shared/compute/useCompute';
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
import { DEFAULT_GNN, DEPTHS, type GnnResult, type GnnScenario } from './gnnCore';
import { gnnExperiments, gnnParams, type GnnExperiment } from './gnnSpec';

const toScenario = (p: ParamValues): GnnScenario => ({
  pOut: Number(p.pOut),
  signal: Number(p.signal),
  labelled: Number(p.labelled),
  layers: Number(p.layers),
  useGraph: Boolean(p.useGraph),
});

const fromScenario = (s: GnnScenario): ParamValues => ({ ...s });

const HUES = [20, 150, 260];
const SIZE = 340;
const at = (v: number) => ((v + 1) / 2) * SIZE;

function Results({ run }: { run: GnnResult }) {
  const palette = usePalette();
  const s = run.scenario;
  const di = DEPTHS.indexOf(s.layers);
  const metrics: Metric[] = [
    {
      label: 'Accuracy on the unlabelled nodes',
      value: `${fix(run.shownAccuracy * 100, 0)}%`,
      testId: 'accuracy',
      hint: `${s.layers} layers, ${s.useGraph ? 'using the graph' : 'ignoring the graph'}`,
    },
    {
      label: 'Same depth, using the graph',
      value: `${fix((run.graph[di] as number) * 100, 0)}%`,
      hint: 'mean of 3 graphs',
    },
    {
      label: 'Same depth, ignoring the graph',
      value: `${fix((run.plain[di] as number) * 100, 0)}%`,
    },
  ];
  return (
    <>
      <Panel id="lab-graph" className="scroll-mt-24" title="The graph" eyebrow="01">
        <div className="space-y-6">
          <MetricsStrip metrics={metrics} className="lg:grid-cols-3" />
          <figure className="space-y-2">
            <svg
              viewBox={`0 0 ${SIZE} ${SIZE}`}
              className="w-full max-w-[26rem] rounded-xl border border-line bg-surface"
              role="img"
              aria-label="A graph of three communities, coloured by the class the network predicts"
            >
              {run.edges.map(([u, v], i) => {
                const a = run.nodes[u] as GnnResult['nodes'][number];
                const b = run.nodes[v] as GnnResult['nodes'][number];
                return (
                  <line
                    key={i}
                    x1={at(a.x)}
                    y1={at(a.y)}
                    x2={at(b.x)}
                    y2={at(b.y)}
                    stroke={a.label === b.label ? palette.css.muted : palette.css.neg}
                    strokeOpacity={a.label === b.label ? 0.18 : 0.6}
                  />
                );
              })}
              {run.nodes.map((n, i) => (
                <circle
                  key={i}
                  cx={at(n.x)}
                  cy={at(n.y)}
                  r={n.labelled ? 6 : 4}
                  fill={`hsl(${HUES[n.predicted]} 65% 52%)`}
                  stroke={n.predicted === n.label ? palette.css.fg : palette.css.neg}
                  strokeWidth={n.labelled ? 2 : n.predicted === n.label ? 0.6 : 2}
                />
              ))}
            </svg>
            <figcaption className="text-sm text-muted">
              Each dot is a node, coloured by the class the network predicts. Large dots are the
              labelled ones. A red outline means a wrong prediction; red lines are edges between
              communities.
            </figcaption>
          </figure>
        </div>
      </Panel>
      <Panel id="lab-depth" className="scroll-mt-24" title="Depth" eyebrow="02">
        <div className="grid gap-6 lg:grid-cols-2">
          <MiniChart
            title="Accuracy on unlabelled nodes by number of layers"
            x={run.depths}
            xLabel="layers"
            series={[
              { name: 'using the graph', color: palette.css.pos, values: run.graph },
              {
                name: 'ignoring the graph',
                color: palette.css.muted,
                values: run.plain,
                dashed: true,
              },
            ]}
            height={220}
          />
          <MiniChart
            title="Similarity of nodes from different communities"
            x={run.depths}
            xLabel="layers"
            series={[{ name: 'cosine similarity', color: palette.css.neg, values: run.otherClass }]}
            height={220}
          />
        </div>
      </Panel>
    </>
  );
}

export default function GnnPage() {
  const [params, setParams] = useState<ParamValues>(() => fromScenario(DEFAULT_GNN));
  const [activeId, setActiveId] = useState<string | null>(null);
  const scenario = useMemo(() => toScenario(params), [params]);
  const { result: run, error } = useCompute('gnn', scenario);
  const setParam = (key: string, value: ParamValue) => setParams((p) => ({ ...p, [key]: value }));

  const apply = (exp: GnnExperiment) => {
    setParams(fromScenario({ ...DEFAULT_GNN, ...exp.scenario }));
    setActiveId(exp.id);
  };
  const stillActive = (() => {
    const exp = gnnExperiments.find((e) => e.id === activeId);
    return (
      !!exp &&
      JSON.stringify(fromScenario({ ...DEFAULT_GNN, ...exp.scenario })) === JSON.stringify(params)
    );
  })();

  const stages: [string, string][] = [
    ['lab-graph', 'The graph'],
    ['lab-depth', 'Depth'],
    ['lab-how', 'How it works'],
    ['lab-controls', 'Settings'],
  ];

  return (
    <div className="mx-auto max-w-6xl px-4 pt-8 pb-20 sm:px-6 lg:px-8">
      <LabHeader
        eyebrow="Modern topics"
        title="Graph neural networks"
        accent="labels that travel along edges"
        blurb="Some data comes as a graph: people and their friendships, papers and their citations. A graph neural network lets each node look at its neighbours, so a handful of labels can classify everything connected to them. Stack too many layers and the neighbourhood grows until all the nodes look alike."
        badges={
          <>
            <Badge>3 communities of 40 nodes</Badge>
            <Badge>
              {scenario.useGraph ? `${scenario.layers}-layer GCN` : `${scenario.layers}-layer MLP`}
            </Badge>
          </>
        }
      />
      <StageNav stages={stages} />
      <ExperimentBar
        lessons={gnnExperiments}
        activeId={stillActive ? activeId : null}
        onApply={apply}
      />

      <div className="flex flex-col gap-5">
        {run ? (
          <Results run={run} />
        ) : (
          <Panel id="lab-graph" className="scroll-mt-24" title="The graph" eyebrow="01">
            <p className="text-sm text-muted" role="status">
              {error
                ? `Something went wrong: ${error}`
                : 'Training the networks on three random graphs…'}
            </p>
          </Panel>
        )}
        <Panel id="lab-how" className="scroll-mt-24" title="How it works" eyebrow="03">
          <div className="space-y-3 text-sm text-muted">
            <p>
              Each layer replaces a node&rsquo;s features by a weighted average of its own and its
              neighbours&rsquo;, then applies a learned linear map and a nonlinearity. After{' '}
              <em>k</em> layers a node has been influenced by everything within <em>k</em> steps.
              The weights are learned from the few labelled nodes only; the structure of the graph
              spreads that knowledge to the rest. A network with the averaging switched off is an
              ordinary MLP that sees each node alone.
            </p>
            <p>
              The graph is random: nodes in the same community are linked with probability 0.15,
              nodes in different ones with the probability you choose, and each node&rsquo;s
              features are noise around a community-specific centre. The tests check the gradients,
              the normalised adjacency, and every sentence above.
            </p>
          </div>
        </Panel>
        <Panel id="lab-controls" className="scroll-mt-24" title="Settings" eyebrow="04">
          <ParamPanel specs={gnnParams} values={params} onChange={setParam} columns={2} />
        </Panel>
      </div>
    </div>
  );
}
