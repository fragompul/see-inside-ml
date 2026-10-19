import { cellInfo, type CellName } from '@sim/deep';
import { Pause, Play, RotateCcw } from 'lucide-react';
import { useEffect, useRef, useState } from 'react';
import { Badge, Button, Panel, Slider } from '@/design-system';
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
import { DEFAULT_CHARLM, makeTrainer, type CharLmScenario } from './charlmCore';
import { CORPORA, type CorpusId } from './charlmData';
import { charlmExperiments, charlmParams, type CharLmExperiment } from './charlmSpec';

const SNAPSHOT_STEPS = [0, 10, 50, 200, 600, 1500, 4000];

const toScenario = (p: ParamValues, steps = 0): CharLmScenario => ({
  ...DEFAULT_CHARLM,
  corpus: p.corpus as CorpusId,
  cell: p.cell as CellName,
  hidden: Number(p.hidden),
  learningRate: Number(p.learningRate),
  temperature: Number(p.temperature),
  steps,
});

const fromScenario = (s: CharLmScenario): ParamValues => ({
  corpus: s.corpus,
  cell: s.cell,
  hidden: s.hidden,
  learningRate: s.learningRate,
  temperature: s.temperature,
});

const prefixOf = (c: CorpusId) => (c === 'hamlet' ? 'to ' : c === 'grammar' ? 'the ' : 'twinkle');

interface View {
  steps: number;
  loss: number[];
  validation: number;
  sample: string;
  snapshots: { step: number; text: string }[];
}

/** One training session. Remounted (via `key`) whenever the model or the text changes. */
function Session({
  scenario,
  target,
  running,
  setRunning,
  onTarget,
}: {
  scenario: CharLmScenario;
  target: number | null;
  running: boolean;
  setRunning: (v: boolean) => void;
  onTarget: () => void;
}) {
  const [trainer] = useState(() => makeTrainer(scenario));
  const temperature = useRef(scenario.temperature);
  useEffect(() => {
    temperature.current = scenario.temperature;
  }, [scenario.temperature]);
  const snaps = useRef<{ step: number; text: string }[]>([]);
  const nextSnap = useRef(0);
  const [view, setView] = useState<View>(() => ({
    steps: 0,
    loss: [],
    validation: trainer.validationLoss(),
    sample: '',
    snapshots: [],
  }));
  const prefix = prefixOf(scenario.corpus);

  const refresh = (withSample: boolean) => {
    const sample = withSample ? trainer.generate(200, temperature.current, 4, prefix) : undefined;
    while (
      nextSnap.current < SNAPSHOT_STEPS.length &&
      trainer.steps >= (SNAPSHOT_STEPS[nextSnap.current] as number)
    ) {
      snaps.current = [
        ...snaps.current,
        {
          step: SNAPSHOT_STEPS[nextSnap.current] as number,
          text: trainer.generate(100, 0.7, 4, prefix),
        },
      ];
      nextSnap.current += 1;
    }
    const h = trainer.history;
    const stride = Math.max(1, Math.floor(h.length / 400));
    setView((v) => ({
      steps: trainer.steps,
      loss: h.filter((_, i) => i % stride === 0),
      validation: trainer.validationLoss(),
      sample: sample ?? v.sample,
      snapshots: snaps.current,
    }));
  };

  // first render: the untrained model and its first snapshot
  useEffect(() => {
    refresh(true);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // the same sample, redrawn when only the temperature changes
  useEffect(() => {
    if (!running) refresh(true);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [scenario.temperature]);

  useEffect(() => {
    if (!running) return;
    let raf = 0;
    let last = 0;
    let lastSample = 0;
    const tick = (now: number) => {
      const perFrame = 4;
      for (let i = 0; i < perFrame; i++) {
        trainer.step();
        if (target !== null && trainer.steps >= target) break;
      }
      const done = target !== null && trainer.steps >= target;
      if (done || now - last > 120) {
        last = now;
        const sample = done || now - lastSample > 600;
        if (sample) lastSample = now;
        refresh(sample);
      }
      if (done) {
        setRunning(false);
        onTarget();
        return;
      }
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [running, target]);

  const train = view.loss.length
    ? (view.loss[view.loss.length - 1] as number)
    : Math.log(trainer.corpus.alphabet.length);
  const uniform = Math.log(trainer.corpus.alphabet.length);
  const metrics: Metric[] = [
    { label: 'Steps', value: String(view.steps), testId: 'steps' },
    {
      label: 'Training loss',
      value: fix(train, 3),
      testId: 'loss',
      hint: 'average negative log-probability of the next character, in nats',
    },
    {
      label: trainer.holdout ? 'Loss on unseen text' : 'Loss on the text',
      value: fix(view.validation, 3),
      hint: 'the same measure on text not used for training (the last 80 characters of Hamlet)',
    },
    {
      label: 'Perplexity',
      value: fix(Math.exp(view.validation), 1),
      hint: 'the effective number of equally likely next characters',
    },
    {
      label: 'Random guessing',
      value: fix(uniform, 3),
      hint: 'the loss of picking every character with equal probability',
    },
  ];

  return (
    <>
      <Panel id="lab-train" className="scroll-mt-24" title="Training" eyebrow="01">
        <div className="space-y-6">
          <MetricsStrip metrics={metrics} className="lg:grid-cols-5" />
          {view.loss.length > 1 && (
            <MiniChart
              title="Training loss (nats per character)"
              x={view.loss.map((_, i) => i)}
              xLabel="update (thinned)"
              series={[
                { name: 'training loss', color: 'var(--brand)', values: view.loss },
                {
                  name: 'random guessing',
                  color: 'var(--muted)',
                  values: view.loss.map(() => uniform),
                  dashed: true,
                },
              ]}
              height={190}
            />
          )}
        </div>
      </Panel>
      <Panel id="lab-sample" className="scroll-mt-24" title="What it writes" eyebrow="02">
        <div className="space-y-4">
          <pre
            className="overflow-x-auto rounded-xl border border-line bg-raised/40 p-4 font-mono text-[0.82rem] leading-relaxed whitespace-pre-wrap"
            data-testid="sample"
            aria-label="Text generated by the network"
          >
            {view.sample || ' '}
          </pre>
          <p className="text-sm text-muted">
            Generated one character at a time from the current model, starting from &ldquo;{prefix}
            &rdquo;; each character is drawn from the predicted distribution at temperature{' '}
            {scenario.temperature.toFixed(1)}. Move the temperature slider under Settings to change
            how adventurous it is.
          </p>
        </div>
      </Panel>
      <Panel
        id="lab-progress"
        className="scroll-mt-24"
        title="How the writing improves"
        eyebrow="03"
      >
        <div className="space-y-3">
          {view.snapshots.length === 0 && (
            <p className="text-sm text-muted">Press Train to start.</p>
          )}
          {view.snapshots.map((s) => (
            <div key={s.step} className="grid gap-1 sm:grid-cols-[6.5rem_minmax(0,1fr)] sm:gap-4">
              <span className="font-mono text-[0.74rem] text-faint">after {s.step} steps</span>
              <pre className="overflow-x-auto font-mono text-[0.78rem] whitespace-pre-wrap text-muted">
                {s.text}
              </pre>
            </div>
          ))}
        </div>
      </Panel>
    </>
  );
}

export default function CharLmPage() {
  const [params, setParams] = useState<ParamValues>(() => fromScenario(DEFAULT_CHARLM));
  const [running, setRunning] = useState(false);
  const [target, setTarget] = useState<number | null>(null);
  const [activeId, setActiveId] = useState<string | null>(null);
  const [epoch, setEpoch] = useState(0);
  const scenario = toScenario(params);
  const modelKey = `${params.corpus}|${params.cell}|${params.hidden}|${params.learningRate}|${epoch}`;
  const setParam = (key: string, value: ParamValue) => {
    setParams((p) => ({ ...p, [key]: value }));
    if (key !== 'temperature') {
      setRunning(false);
      setTarget(null);
    }
  };

  const apply = (exp: CharLmExperiment) => {
    const s = { ...DEFAULT_CHARLM, ...exp.scenario };
    setParams(fromScenario(s));
    setEpoch((e) => e + 1);
    setTarget(s.steps > 0 ? s.steps : null);
    setRunning(s.steps > 0);
    setActiveId(exp.id);
    window.setTimeout(
      () =>
        document
          .getElementById('lab-train')
          ?.scrollIntoView({ behavior: 'smooth', block: 'start' }),
      60,
    );
  };
  const stillActive = (() => {
    const exp = charlmExperiments.find((e) => e.id === activeId);
    return (
      !!exp &&
      JSON.stringify(fromScenario({ ...DEFAULT_CHARLM, ...exp.scenario })) ===
        JSON.stringify(params)
    );
  })();

  const stages: [string, string][] = [
    ['lab-train', 'Training'],
    ['lab-sample', 'What it writes'],
    ['lab-progress', 'Progress'],
    ['lab-how', 'How it works'],
    ['lab-controls', 'Settings'],
  ];

  return (
    <div className="mx-auto max-w-6xl px-4 pt-8 pb-20 sm:px-6 lg:px-8">
      <LabHeader
        eyebrow="Deep learning"
        title="A network that writes"
        accent="one character at a time"
        blurb="A recurrent network reads text a character at a time and learns to predict the next one. Train it on a rhyme, a grammar or a few lines of Shakespeare, watch its writing go from random letters to words to sentences, and sample it hot or cold."
        badges={
          <>
            <Badge>{CORPORA[scenario.corpus].label}</Badge>
            <Badge>{cellInfo[scenario.cell].label}</Badge>
            <Badge>{scenario.hidden} hidden units</Badge>
          </>
        }
      />
      <StageNav stages={stages} />
      <ExperimentBar
        lessons={charlmExperiments}
        activeId={stillActive ? activeId : null}
        onApply={apply}
      />

      <div className="mb-5 flex flex-wrap items-center gap-3">
        <Button
          onClick={() => {
            setTarget(null);
            setRunning((r) => !r);
          }}
        >
          {running ? <Pause className="size-4" /> : <Play className="size-4" />}{' '}
          {running ? 'Pause' : 'Train'}
        </Button>
        <Button
          variant="secondary"
          onClick={() => {
            setRunning(false);
            setTarget(null);
            setEpoch((e) => e + 1);
          }}
        >
          <RotateCcw className="size-4" /> Start again
        </Button>
        <p className="text-[0.78rem] text-muted">{CORPORA[scenario.corpus].note}</p>
      </div>

      <div className="flex flex-col gap-5">
        <Session
          key={modelKey}
          scenario={scenario}
          target={target}
          running={running}
          setRunning={setRunning}
          onTarget={() => setTarget(null)}
        />
        <Panel id="lab-how" className="scroll-mt-24" title="How it works" eyebrow="04">
          <div className="space-y-3 text-sm text-muted">
            <p>
              Each character is a one-hot input to a recurrent cell (vanilla RNN, LSTM or GRU). The
              cell updates its hidden state, a small vector that is the network&apos;s only memory
              of what came before, and a softmax layer turns that state into a probability for every
              possible next character. Training minimises the cross-entropy between that prediction
              and the character that really came next, using{' '}
              <strong className="text-fg">backpropagation through time</strong> over windows of 24
              characters and the Adam optimiser with gradient clipping. The gradients of all three
              cells are checked against finite differences in the tests.
            </p>
            <p>
              To write, the network predicts a distribution, one character is drawn from it
              (reshaped by the <strong className="text-fg">temperature</strong>: below 1 it sharpens
              the distribution, above 1 it flattens it), appended to the text, and the process
              repeats. This is, in miniature, how large language models generate: the difference is
              scale, the tokens (pieces of words rather than characters) and the architecture
              (attention instead of a recurrent state).
            </p>
          </div>
        </Panel>
        <Panel id="lab-controls" className="scroll-mt-24" title="Settings" eyebrow="05">
          <div className="space-y-6">
            <ParamPanel specs={charlmParams} values={params} onChange={setParam} columns={3} />
            <p className="text-[0.74rem] text-faint">
              Changing the text, the cell, the memory size or the learning rate starts a fresh
              network.
            </p>
            <Slider
              label="Sampling temperature (live)"
              min={0.1}
              max={3}
              step={0.1}
              value={Number(params.temperature)}
              onValueChange={(v) => setParam('temperature', v)}
              format={(v) => v.toFixed(1)}
            />
          </div>
        </Panel>
      </div>
    </div>
  );
}
