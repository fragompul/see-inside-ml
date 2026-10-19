import {
  type PlaygroundConfig,
  type SceneData,
  type Snapshot,
  type StepInspection,
} from '@sim/deep';
import { AnimatePresence, motion } from 'framer-motion';
import { ChevronLeft, ChevronRight, Dices, Pause, Play, X } from 'lucide-react';
import { useEffect, useMemo, useState } from 'react';
import { Button, Tex, cn } from '@/design-system';
import { buildEdges } from '../graph';
import { STAGES, biggestUpdate, formatValue } from '../inspectView';
import { usePalette } from '@/shared/canvas/colors';

interface Props {
  inspection: StepInspection;
  stage: number;
  onStage: (stage: number) => void;
  onRandomSample: () => void;
  onClose: () => void;
  scene: SceneData;
  snapshot: Snapshot;
  config: PlaygroundConfig;
}

const AUTOPLAY_MS = 2600;

const argmax = (values: number[]) =>
  values.reduce((best, v, i) => (v > (values[best] as number) ? i : best), 0);

/** Narrated, number-by-number walk through one training step (forward → loss → backward → update). */
export function InspectPanel({
  inspection,
  stage,
  onStage,
  onRandomSample,
  onClose,
  scene,
  snapshot,
  config,
}: Props) {
  const palette = usePalette();
  const [playing, setPlaying] = useState(false);

  useEffect(() => {
    if (!playing) return;
    const timer = window.setInterval(() => onStage(stage === 3 ? 0 : stage + 1), AUTOPLAY_MS);
    return () => window.clearInterval(timer);
  }, [playing, stage, onStage]);

  const classification = scene.problem === 'classification';
  const output = inspection.activations[inspection.activations.length - 1] as number[];
  const predicted = classification ? argmax(output) : 0;
  const truth = classification ? inspection.targets.indexOf(1) : 0;
  const correct = classification && predicted === truth;

  const edges = useMemo(
    () => buildEdges(snapshot.weights, scene.layerSizes, config.skipConnections),
    [snapshot.weights, scene.layerSizes, config.skipConnections],
  );
  const biggest = stage === 3 ? biggestUpdate(edges, inspection) : null;
  const nameOf = (layer: number, index: number) =>
    layer === 0 ? (scene.featureLabels[index] ?? `x${index + 1}`) : `L${layer}·n${index + 1}`;

  const vector = (values: number[]) =>
    `\\left[${values.map((v) => formatValue(v).replace('−', '-')).join(',\\ ')}\\right]`;

  const dot = (k: number) => (
    <span
      className="mr-1 inline-block size-2 rounded-full align-baseline"
      style={{ background: palette.css.classes[k % palette.css.classes.length] }}
    />
  );

  const body = [
    // 0 · forward
    <div key="forward" className="space-y-3">
      <p>
        The sample{' '}
        <span className="font-mono text-fg">
          ({inspection.point.x.toFixed(2)}, {inspection.point.y.toFixed(2)})
        </span>
        {classification ? (
          <>
            {' '}
            truly belongs to class {dot(truth)}
            <span className="font-mono text-fg">{truth}</span>
          </>
        ) : (
          <>
            {' '}
            has target{' '}
            <span className="font-mono text-fg">{formatValue(inspection.targets[0] ?? 0)}</span>
          </>
        )}
        . It enters on the left and every neuron computes a weighted sum of the previous layer, then
        applies its activation. Numbers on neurons are their outputs <Tex tex="a" />.
      </p>
      <p className="text-fg">
        <Tex tex="a^{(\ell)} = f\!\left(W^{(\ell)}\,a^{(\ell-1)} + b^{(\ell)}\right)" block />
      </p>
      <p>
        {classification ? (
          <>
            Prediction: class {dot(predicted)}
            <span className="font-mono text-fg">{predicted}</span> with{' '}
            <span className="font-mono text-fg">
              {((output[predicted] as number) * 100).toFixed(1)}%
            </span>{' '}
            confidence.{' '}
            <span className={correct ? 'text-ok' : 'text-danger'}>
              {correct ? 'Correct.' : 'Wrong, so there is something to learn here.'}
            </span>
          </>
        ) : (
          <>
            Prediction <span className="font-mono text-fg">ŷ = {formatValue(output[0] ?? 0)}</span>.
          </>
        )}
      </p>
    </div>,

    // 1 · loss
    <div key="loss" className="space-y-3">
      <p>
        The loss turns “how wrong was that?” into one number the optimiser can reduce.
        {classification
          ? ' Cross-entropy only looks at the probability given to the true class: confident and right means a small loss, confident and wrong a huge one.'
          : ' Squared error grows quickly as the prediction drifts from the target.'}
      </p>
      <p className="text-fg">
        {classification ? (
          <Tex
            tex={`L = -\\log \\hat y_{${truth}} = -\\log(${formatValue(output[truth] ?? 0)}) = ${formatValue(inspection.loss)}`}
            block
          />
        ) : (
          <Tex
            tex={`L = \\tfrac12(\\hat y - y)^2 = \\tfrac12(${formatValue(output[0] ?? 0)} - ${formatValue(inspection.targets[0] ?? 0)})^2 = ${formatValue(inspection.loss)}`}
            block
          />
        )}
      </p>
      {classification && (
        <p className="font-mono text-[0.78rem]">
          <Tex tex={`\\hat y = ${vector(output)}`} /> <span className="text-faint">vs</span>{' '}
          <Tex tex={`y = ${vector(inspection.targets)}`} />
        </p>
      )}
    </div>,

    // 2 · backward
    <div key="backward" className="space-y-3">
      <p>
        Now the question flips: <em>which weights are to blame, and by how much?</em> The chain rule
        sends the error backwards. Violet pulses travel right → left; the number on each neuron is
        its error signal <Tex tex="\delta" />, and an edge’s width is its gradient{' '}
        <Tex tex="\partial L/\partial w" />.
      </p>
      <div className="space-y-1 text-fg">
        <Tex
          tex={classification ? '\\delta^{(L)} = \\hat y - y' : '\\delta^{(L)} = \\hat y - y'}
          block
        />
        <Tex
          tex="\delta^{(\ell)} = \left(W^{(\ell+1)\top}\delta^{(\ell+1)}\right)\odot f'\!\left(z^{(\ell)}\right)"
          block
        />
        <Tex
          tex="\frac{\partial L}{\partial W^{(\ell)}} = a^{(\ell-1)}\,\delta^{(\ell)\top}"
          block
        />
      </div>
      <p className="text-[0.82rem]">
        Hover any edge to see its input, its <Tex tex="\delta" /> and the product between them.
      </p>
    </div>,

    // 3 · update
    <div key="update" className="space-y-3">
      <p>
        Every weight takes a small step <em>against</em> its gradient, scaled by the learning rate{' '}
        <Tex tex="\eta" /> = <span className="font-mono text-fg">{formatValue(inspection.lr)}</span>
        . Edge width is now the size of that change.
      </p>
      <p className="text-fg">
        <Tex tex="w \leftarrow w - \eta\,\frac{\partial L}{\partial w}" block />
      </p>
      {biggest && (
        <p>
          Biggest move:{' '}
          <span className="font-mono text-fg">
            {nameOf(biggest.edge.fromLayer, biggest.edge.fromIndex)}
          </span>{' '}
          →{' '}
          <span className="font-mono text-fg">
            {nameOf(biggest.edge.toLayer, biggest.edge.toIndex)}
          </span>
          :{' '}
          <span className="font-mono text-fg">
            {formatValue(biggest.edge.weight)} → {formatValue(biggest.edge.weight + biggest.delta)}
          </span>
          .
        </p>
      )}
      <p className="text-[0.82rem] text-faint">
        This used a single sample. A real step averages the gradient over a mini-batch of{' '}
        {config.batchSize} and lets <span className="text-muted">{config.optimizer}</span> reshape
        the step. Press Train to see thousands of them.
      </p>
    </div>,
  ];

  return (
    <motion.section
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: 12 }}
      transition={{ duration: 0.3 }}
      aria-label="Anatomy of a training step"
      className="rounded-2xl border border-brand/40 bg-surface/80 shadow-[0_0_0_4px_var(--brand-soft)] backdrop-blur-sm"
    >
      <header className="flex flex-wrap items-center justify-between gap-3 border-b border-line px-5 py-3">
        <div className="flex items-baseline gap-2.5">
          <span className="eyebrow text-brand">Inspector</span>
          <h2 className="font-display text-[1.02rem] font-semibold tracking-tight">
            Anatomy of one step
          </h2>
        </div>
        <div className="flex items-center gap-1.5">
          <Button variant="ghost" size="sm" onClick={onRandomSample}>
            <Dices className="size-4" /> Another sample
          </Button>
          <Button variant="ghost" size="sm" onClick={onClose} aria-label="Close the inspector">
            <X className="size-4" /> Close
          </Button>
        </div>
      </header>

      <div className="space-y-5 p-5">
        <ol className="flex flex-wrap items-center gap-2" aria-label="Stages">
          {STAGES.map((s, i) => (
            <li key={s.id}>
              <button
                type="button"
                aria-current={stage === i ? 'step' : undefined}
                onClick={() => {
                  setPlaying(false);
                  onStage(i);
                }}
                className={cn(
                  'flex cursor-pointer items-center gap-2 rounded-full border px-3 py-1.5 text-[0.8rem] transition-all duration-200',
                  stage === i
                    ? 'border-brand bg-brand text-brand-fg shadow-[0_0_0_3px_var(--brand-soft)]'
                    : i < stage
                      ? 'border-brand/40 text-fg hover:border-brand'
                      : 'border-line-strong text-muted hover:border-brand hover:text-fg',
                )}
              >
                <span className="font-mono text-[0.7rem] opacity-80">{i + 1}</span>
                {s.short}
              </button>
            </li>
          ))}
          <li className="ml-auto flex items-center gap-1">
            <Button
              variant="secondary"
              size="sm"
              onClick={() => setPlaying((p) => !p)}
              aria-pressed={playing}
              aria-label={playing ? 'Pause the walkthrough' : 'Play the walkthrough'}
            >
              {playing ? <Pause className="size-3.5" /> : <Play className="size-3.5" />}
              {playing ? 'Pause' : 'Auto-play'}
            </Button>
            <Button
              variant="ghost"
              size="sm"
              aria-label="Previous stage"
              disabled={stage === 0}
              onClick={() => {
                setPlaying(false);
                onStage(stage - 1);
              }}
            >
              <ChevronLeft className="size-4" />
            </Button>
            <Button
              variant="ghost"
              size="sm"
              aria-label="Next stage"
              disabled={stage === 3}
              onClick={() => {
                setPlaying(false);
                onStage(stage + 1);
              }}
            >
              <ChevronRight className="size-4" />
            </Button>
          </li>
        </ol>

        <div className="min-h-[11rem] text-[0.92rem] leading-relaxed text-muted">
          <AnimatePresence mode="wait" initial={false}>
            <motion.div
              key={stage}
              initial={{ opacity: 0, x: 14 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: -14 }}
              transition={{ duration: 0.2 }}
            >
              <h3 className="mb-2 font-display text-lg font-semibold tracking-tight text-fg">
                {stage + 1}. {STAGES[stage]?.title}
              </h3>
              {body[stage]}
            </motion.div>
          </AnimatePresence>
        </div>
      </div>
    </motion.section>
  );
}
