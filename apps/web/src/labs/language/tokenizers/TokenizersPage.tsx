import { useMemo, useState } from 'react';
import { Badge, Panel } from '@/design-system';
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
  DEFAULT_TOKENIZER,
  SCHEME_LABEL,
  runTokenizer,
  type TokenizerScenario,
} from './tokenizerCore';
import { TESTS, type TestText } from './tokenizerData';
import { tokenizerExperiments, tokenizerParams, type TokenizerExperiment } from './tokenizerSpec';

const toScenario = (p: ParamValues): TokenizerScenario => ({
  scheme: p.scheme as TokenizerScenario['scheme'],
  merges: Number(p.merges),
  trainOn: p.trainOn as TokenizerScenario['trainOn'],
  testOn: p.testOn as TokenizerScenario['testOn'],
});

const fromScenario = (s: TokenizerScenario): ParamValues => ({ ...s });

const visible = (t: string) => t.replace(/ /g, '·').replace(/\n/g, '↵');

export default function TokenizersPage() {
  const [params, setParams] = useState<ParamValues>(() => fromScenario(DEFAULT_TOKENIZER));
  const [activeId, setActiveId] = useState<string | null>(null);
  const scenario = useMemo(() => toScenario(params), [params]);
  const run = useMemo(() => runTokenizer(scenario), [scenario]);
  const setParam = (key: string, value: ParamValue) => setParams((p) => ({ ...p, [key]: value }));

  const apply = (exp: TokenizerExperiment) => {
    setParams(fromScenario({ ...DEFAULT_TOKENIZER, ...exp.scenario }));
    setActiveId(exp.id);
  };
  const stillActive = (() => {
    const exp = tokenizerExperiments.find((e) => e.id === activeId);
    return (
      !!exp &&
      JSON.stringify(fromScenario({ ...DEFAULT_TOKENIZER, ...exp.scenario })) ===
        JSON.stringify(params)
    );
  })();

  const byText = useMemo(
    () =>
      (Object.keys(TESTS) as TestText[]).map((id) => ({
        id,
        run: runTokenizer({ ...scenario, testOn: id }),
      })),
    [scenario],
  );

  const metrics: Metric[] = [
    { label: 'Vocabulary size', value: String(run.vocab), testId: 'vocab' },
    { label: 'Tokens for this text', value: String(run.tokens) },
    { label: 'Tokens per 100 characters', value: fix(run.per100, 1) },
    {
      label: 'Pieces with no token',
      value: `${fix(run.unknown * 100, 0)}%`,
      hint: scenario.scheme === 'words' ? 'replaced by an unknown token' : 'never happens',
    },
  ];

  const stages: [string, string][] = [
    ['lab-text', 'The text, cut up'],
    ['lab-compare', 'Across texts'],
    ['lab-how', 'How it works'],
    ['lab-controls', 'Settings'],
  ];

  return (
    <div className="mx-auto max-w-6xl px-4 pt-8 pb-20 sm:px-6 lg:px-8">
      <LabHeader
        eyebrow="Language models"
        title="Tokenizers compared"
        accent="how text becomes numbers"
        blurb="A language model never sees letters, only token numbers, and the way text is cut up shapes what is cheap, what is expensive and what is impossible. Cut the same text into bytes, whole words or learned pieces, and see what each choice does to length, vocabulary and unfamiliar words."
        badges={
          <>
            <Badge>{SCHEME_LABEL[scenario.scheme]}</Badge>
            <Badge>{run.vocab} tokens in the vocabulary</Badge>
          </>
        }
      />
      <StageNav stages={stages} />
      <ExperimentBar
        lessons={tokenizerExperiments}
        activeId={stillActive ? activeId : null}
        onApply={apply}
      />

      <div className="flex flex-col gap-5">
        <Panel id="lab-text" className="scroll-mt-24" title="The text, cut up" eyebrow="01">
          <div className="space-y-6">
            <MetricsStrip metrics={metrics} className="lg:grid-cols-4" />
            <div
              className="flex flex-wrap gap-y-1 rounded-xl border border-line bg-surface p-3 font-mono text-[13px] leading-6"
              aria-label="The test text as tokens"
            >
              {run.pieces.map((p, i) => (
                <span
                  key={i}
                  title={p.unknown ? 'unknown token' : undefined}
                  className={
                    p.unknown
                      ? 'rounded-sm bg-neg/30 px-[1px] text-fg'
                      : i % 2
                        ? 'rounded-sm bg-brand/20 px-[1px] text-fg'
                        : 'rounded-sm bg-pos/20 px-[1px] text-fg'
                  }
                >
                  {p.unknown ? '⟨unk⟩' : visible(p.text)}
                </span>
              ))}
            </div>
            <p className="text-sm text-muted">
              Each coloured block is one token; a dot stands for a space. With whole words, a red
              ⟨unk⟩ is a word the vocabulary has never seen. Characters made of several bytes show
              up as the hexadecimal values of those bytes when they have not been merged.
            </p>
          </div>
        </Panel>
        <Panel id="lab-compare" className="scroll-mt-24" title="Across texts" eyebrow="02">
          <div className="overflow-x-auto">
            <table className="w-full text-left text-sm">
              <thead className="text-muted">
                <tr>
                  <th className="py-2 pr-4 font-medium">Text</th>
                  <th className="py-2 pr-4 font-medium">Tokens per 100 characters</th>
                  <th className="py-2 font-medium">Pieces with no token</th>
                </tr>
              </thead>
              <tbody>
                {byText.map(({ id, run: r }) => (
                  <tr key={id} className="border-t border-line">
                    <td className="py-2 pr-4">{(TESTS[id] as { label: string }).label}</td>
                    <td className="py-2 pr-4 font-mono">{fix(r.per100, 1)}</td>
                    <td className="py-2 font-mono">{fix(r.unknown * 100, 0)}%</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Panel>
        <Panel id="lab-how" className="scroll-mt-24" title="How it works" eyebrow="03">
          <div className="space-y-3 text-sm text-muted">
            <p>
              <strong className="text-fg">Bytes</strong> need no learning and can write anything,
              but a sentence is as long as it has bytes.{' '}
              <strong className="text-fg">Whole words</strong> are short but need a huge vocabulary,
              and any word outside it is lost.{' '}
              <strong className="text-fg">Byte-pair encoding</strong> starts from bytes and
              repeatedly fuses the most frequent neighbouring pair into a new token, so common words
              end up as one token and rare ones as a few pieces, with bytes as a safety net.
            </p>
            <p>
              The trainer here learns exactly the same merges as the reference Python implementation
              used for the Transformer lab, and the tests check this on texts with accents, digits
              and code. The training texts here are short, so the lists of merges run out at a few
              hundred; real tokenizers learn tens of thousands from billions of words.
            </p>
          </div>
        </Panel>
        <Panel id="lab-controls" className="scroll-mt-24" title="Settings" eyebrow="04">
          <ParamPanel specs={tokenizerParams} values={params} onChange={setParam} columns={2} />
        </Panel>
      </div>
    </div>
  );
}
