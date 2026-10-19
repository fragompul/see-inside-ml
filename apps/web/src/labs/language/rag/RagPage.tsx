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
import { DEFAULT_RAG, type RagResult, type RagScenario } from './ragCore';
import { ragExperiments, ragParams, type RagExperiment } from './ragSpec';

const toScenario = (p: ParamValues): RagScenario => ({
  ...DEFAULT_RAG,
  facts: Number(p.facts),
  memory: Number(p.memory),
  retriever: p.retriever as RagScenario['retriever'],
  typos: Number(p.typos),
});

const fromScenario = (s: RagScenario): ParamValues => ({
  facts: s.facts,
  memory: s.memory,
  retriever: s.retriever,
  typos: s.typos,
});

const pct = (v: number) => `${fix(v * 100, 0)}%`;

function Results({ run }: { run: RagResult }) {
  const palette = usePalette();
  const metrics: Metric[] = [
    {
      label: 'Closed-book model right',
      value: pct(run.closedBook),
      testId: 'closed-book',
      hint: `${run.parameters.toLocaleString('en-US')} weights, ${run.scenario.facts} facts`,
    },
    {
      label: 'Retrieval system right',
      value: pct(run.rag),
      hint: `right passage first for ${pct(run.recall)} of questions`,
    },
    {
      label: 'After a tenth of the facts change',
      value: `${pct(run.closedBookUpdated)} / ${pct(run.ragUpdated)}`,
      hint: 'closed-book / retrieval, on the changed facts',
    },
  ];
  return (
    <>
      <Panel id="lab-answers" className="scroll-mt-24" title="Two ways to know a fact" eyebrow="01">
        <div className="space-y-6">
          <MetricsStrip metrics={metrics} className="lg:grid-cols-3" />
          <MiniChart
            title="Share of questions answered correctly by size of the knowledge base"
            x={run.sweep.map((p) => p.facts)}
            xLabel="facts in the knowledge base"
            series={[
              {
                name: 'closed-book model',
                color: palette.css.neg,
                values: run.sweep.map((p) => p.closedBook),
              },
              {
                name: 'retrieval system',
                color: palette.css.pos,
                values: run.sweep.map((p) => p.rag),
              },
            ]}
            height={230}
          />
        </div>
      </Panel>
      <Panel id="lab-examples" className="scroll-mt-24" title="Some questions" eyebrow="02">
        <div className="overflow-x-auto">
          <table className="w-full text-left text-sm">
            <thead className="text-muted">
              <tr>
                <th className="py-2 pr-4 font-medium">Question</th>
                <th className="py-2 pr-4 font-medium">True</th>
                <th className="py-2 pr-4 font-medium">Closed book</th>
                <th className="py-2 pr-4 font-medium">Retrieved passage</th>
                <th className="py-2 font-medium">Answer</th>
              </tr>
            </thead>
            <tbody>
              {run.examples.map((e, i) => (
                <tr key={i} className="border-t border-line">
                  <td className="py-2 pr-4">{e.question}</td>
                  <td className="py-2 pr-4 font-mono">{e.truth}</td>
                  <td
                    className={`py-2 pr-4 font-mono ${e.closedBook === e.truth ? '' : 'text-neg'}`}
                  >
                    {e.closedBook}
                  </td>
                  <td className="py-2 pr-4 text-muted">{e.passage}</td>
                  <td className={`py-2 font-mono ${e.rag === e.truth ? '' : 'text-neg'}`}>
                    {e.rag}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Panel>
    </>
  );
}

export default function RagPage() {
  const [params, setParams] = useState<ParamValues>(() => fromScenario(DEFAULT_RAG));
  const [activeId, setActiveId] = useState<string | null>(null);
  const scenario = useMemo(() => toScenario(params), [params]);
  const { result: run, error } = useCompute('rag', scenario);
  const setParam = (key: string, value: ParamValue) => setParams((p) => ({ ...p, [key]: value }));

  const apply = (exp: RagExperiment) => {
    setParams(fromScenario({ ...DEFAULT_RAG, ...exp.scenario }));
    setActiveId(exp.id);
  };
  const stillActive = (() => {
    const exp = ragExperiments.find((e) => e.id === activeId);
    return (
      !!exp &&
      JSON.stringify(fromScenario({ ...DEFAULT_RAG, ...exp.scenario })) === JSON.stringify(params)
    );
  })();

  const stages: [string, string][] = [
    ['lab-answers', 'Two ways to know a fact'],
    ['lab-examples', 'Some questions'],
    ['lab-how', 'How it works'],
    ['lab-controls', 'Settings'],
  ];

  return (
    <div className="mx-auto max-w-6xl px-4 pt-8 pb-20 sm:px-6 lg:px-8">
      <LabHeader
        eyebrow="Language models"
        title="Retrieval-augmented generation"
        accent="look it up instead of remembering it"
        blurb="A language model that has to answer from memory must carry every fact in its weights. One that can look passages up needs only to read. Compare a small closed-book network with a retrieval system on a knowledge base of invented facts: how much each can hold, what happens when a fact changes, and how the retriever fails."
        badges={
          <>
            <Badge>{scenario.facts} facts</Badge>
            <Badge>
              {scenario.retriever === 'words' ? 'word retriever' : 'letter-group retriever'}
            </Badge>
          </>
        }
      />
      <StageNav stages={stages} />
      <ExperimentBar
        lessons={ragExperiments}
        activeId={stillActive ? activeId : null}
        onApply={apply}
      />

      <div className="flex flex-col gap-5">
        {run ? (
          <Results run={run} />
        ) : (
          <Panel
            id="lab-answers"
            className="scroll-mt-24"
            title="Two ways to know a fact"
            eyebrow="01"
          >
            <p className="text-sm text-muted" role="status">
              {error
                ? `Something went wrong: ${error}`
                : 'Building the knowledge bases and training the closed-book network…'}
            </p>
          </Panel>
        )}
        <Panel id="lab-how" className="scroll-mt-24" title="How it works" eyebrow="03">
          <div className="space-y-3 text-sm text-muted">
            <p>
              Each fact is a sentence such as &ldquo;The colour of Velmora is teal.&rdquo; about an
              invented name, with one of twelve colours chosen at random, so nothing can be guessed.
              The <strong className="text-fg">closed-book</strong> model is a small network that
              reads a name as groups of letters and must output the colour; it is trained on all the
              facts until it has learned what it can. The{' '}
              <strong className="text-fg">retrieval</strong> system turns the question and every
              sentence into weighted word (or letter-group) vectors, picks the most similar
              sentence, and reads the colour from its last word.
            </p>
            <p>
              In a real system a language model reads the retrieved text and writes the answer; here
              the reading step is trivial so that what is being tested is the retrieval itself. The
              tests check the similarity measure (a document is exactly 1 similar to itself), the
              retriever on misspelt names, and every sentence on this page.
            </p>
          </div>
        </Panel>
        <Panel id="lab-controls" className="scroll-mt-24" title="Settings" eyebrow="04">
          <ParamPanel specs={ragParams} values={params} onChange={setParam} columns={2} />
        </Panel>
      </div>
    </div>
  );
}
