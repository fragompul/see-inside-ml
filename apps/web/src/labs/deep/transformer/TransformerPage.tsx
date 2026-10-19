import { nextTokenDistribution } from '@sim/deep';
import { Eraser, Pause, Play, SkipForward } from 'lucide-react';
import { useEffect, useMemo } from 'react';
import { Badge, Button, Panel } from '@/design-system';
import { AttentionPanel } from './components/AttentionPanel';
import { BpeDemo } from './components/BpeDemo';
import { BlocksPanel } from './components/BlocksPanel';
import { EmbeddingsPanel } from './components/EmbeddingsPanel';
import { LessonBar } from './components/LessonBar';
import { NextTokenPanel } from './components/NextTokenPanel';
import { SamplingPanel } from './components/SamplingPanel';
import { TokenStrip } from './components/TokenStrip';
import { ThreeDPanel } from './three/ThreeDPanel';
import { currentIds, useTransformerStore } from './store';
import { EXAMPLE_PROMPTS, visible } from './tokens';

const STAGES = [
  ['tf-tokens', 'Tokens'],
  ['tf-embeddings', 'Embeddings'],
  ['tf-attention', 'Attention'],
  ['tf-blocks', 'MLP & residual'],
  ['tf-output', 'Next token'],
  ['tf-3d', '3D'],
] as const;

export default function TransformerPage() {
  const s = useTransformerStore();
  const { status, error, loaded, prompt, generated, sampling, seed, maxNew, playing, selected } = s;

  useEffect(() => {
    void useTransformerStore.getState().load();
    return () => useTransformerStore.getState().stop();
  }, []);

  const ctx = loaded?.model.config.ctx ?? 0;
  const allIds = useMemo(
    () => currentIds({ loaded, prompt, generated }),
    [loaded, prompt, generated],
  );
  const windowStart = Math.max(0, allIds.length - ctx);
  const ids = useMemo(() => allIds.slice(windowStart), [allIds, windowStart]);
  const promptTokens = useMemo(
    () => (loaded ? loaded.tokenizer.encode(prompt).length : 0),
    [loaded, prompt],
  );

  const trace = useMemo(
    () => (loaded && ids.length > 0 ? loaded.model.forward(ids) : null),
    [loaded, ids],
  );
  const position = selected !== null && selected < ids.length ? selected : ids.length - 1;
  const distribution = useMemo(() => {
    if (!trace || position < 0) return null;
    const row = trace.logits.data.slice(
      position * trace.logits.cols,
      (position + 1) * trace.logits.cols,
    );
    return nextTokenDistribution(row, sampling);
  }, [trace, position, sampling]);

  const ready = status === 'ready' && loaded;
  const finished =
    ready && generated.length > 0 && generated[generated.length - 1] === loaded.tokenizer.eotId;
  const text = ready
    ? loaded.tokenizer.decode(generated.filter((id) => id !== loaded.tokenizer.eotId))
    : '';

  return (
    <div className="mx-auto max-w-6xl px-4 pt-8 pb-20 sm:px-6 lg:px-8">
      <header className="mb-8">
        <p className="eyebrow text-brand">05 · Lab</p>
        <h1 className="mt-2 font-display text-[clamp(1.9rem,3.4vw,2.8rem)] leading-none font-semibold tracking-[-0.03em]">
          Transformer <span className="font-serif font-normal italic text-muted">explorer</span>
        </h1>
        <p className="mt-3 max-w-2xl text-[0.95rem] text-muted">
          A real GPT, small enough to open up. It has the same architecture as GPT-2, runs entirely
          in your browser and was trained on short stories about people, pets and objects.
        </p>
        <div className="mt-4 flex flex-wrap gap-2">
          {loaded && (
            <>
              <Badge>{loaded.model.parameterCount.toLocaleString('en-US')} parameters</Badge>
              <Badge>{loaded.model.config.layers} layers</Badge>
              <Badge>{loaded.model.config.heads} heads</Badge>
              <Badge>width {loaded.model.config.d}</Badge>
              <Badge>{loaded.model.config.vocab} tokens</Badge>
            </>
          )}
        </div>
      </header>

      {ready && (
        <nav
          aria-label="Stages of the model"
          className="sticky top-16 z-30 -mx-4 mb-6 overflow-x-auto border-b border-line bg-bg/80 px-4 py-2.5 backdrop-blur-xl sm:-mx-6 sm:px-6 lg:-mx-8 lg:px-8"
        >
          <ol className="flex min-w-max items-center gap-1 font-mono text-[0.72rem]">
            {STAGES.map(([id, label], i) => (
              <li key={id} className="flex items-center gap-1">
                {i > 0 && <span className="text-faint">→</span>}
                <a
                  href={`#${id}`}
                  onClick={(e) => {
                    e.preventDefault();
                    document
                      .getElementById(id)
                      ?.scrollIntoView({ behavior: 'smooth', block: 'start' });
                  }}
                  className="rounded-full px-3 py-1 text-muted transition-colors hover:bg-brand-soft hover:text-fg"
                >
                  <span className="mr-1.5 text-brand">{String(i + 1).padStart(2, '0')}</span>
                  {label}
                </a>
              </li>
            ))}
          </ol>
        </nav>
      )}

      {status === 'error' && (
        <div role="alert" className="rounded-2xl border border-danger/40 bg-danger/10 p-5 text-sm">
          <p className="font-medium text-danger">The model could not be loaded.</p>
          <p className="mt-1 text-muted">{error}</p>
        </div>
      )}

      {(status === 'idle' || status === 'loading') && (
        <div className="space-y-5" role="status" aria-busy="true" aria-label="Loading the model">
          <div className="h-56 animate-pulse rounded-2xl bg-raised/60" />
          <div className="h-96 animate-pulse rounded-2xl bg-raised/60" />
        </div>
      )}

      {ready && trace && distribution && (
        <div className="flex flex-col gap-5">
          <LessonBar
            activeId={s.activeLesson}
            onApply={(lesson) => {
              s.applyLesson(lesson);
              window.setTimeout(
                () =>
                  document
                    .getElementById(lesson.scrollTo)
                    ?.scrollIntoView({ behavior: 'smooth', block: 'start' }),
                60,
              );
            }}
          />
          <Panel id="tf-tokens" className="scroll-mt-24" title="Prompt and tokens" eyebrow="01">
            <div className="space-y-5">
              <div>
                <label htmlFor="prompt" className="mb-2 block text-[0.82rem] text-muted">
                  Write something for the model to continue
                </label>
                <textarea
                  id="prompt"
                  value={prompt}
                  rows={2}
                  maxLength={160}
                  spellCheck={false}
                  onChange={(e) => s.setPrompt(e.target.value)}
                  className="w-full resize-none rounded-xl border border-line-strong bg-raised/50 px-4 py-3 font-mono text-[0.95rem] text-fg transition-colors outline-none focus:border-brand"
                />
                <div className="mt-2 flex flex-wrap gap-1.5">
                  {EXAMPLE_PROMPTS.map((example) => (
                    <button
                      key={example}
                      type="button"
                      onClick={() => s.setPrompt(example)}
                      className="cursor-pointer rounded-full border border-line-strong px-3 py-1 text-[0.74rem] text-muted transition-colors hover:border-brand hover:text-fg"
                    >
                      {example}
                    </button>
                  ))}
                </div>
              </div>

              <div>
                <p className="eyebrow mb-3">The model sees {ids.length} tokens</p>
                {ids.length === 0 ? (
                  <p className="text-sm text-muted">Type something above.</p>
                ) : (
                  <TokenStrip
                    ids={ids}
                    tokenizer={loaded.tokenizer}
                    promptLength={Math.max(0, promptTokens - windowStart)}
                    selected={selected}
                    onSelect={s.select}
                  />
                )}
                {windowStart > 0 && (
                  <p className="mt-3 text-[0.78rem] text-pos">
                    The context window holds {ctx} tokens, so the first {windowStart} are forgotten.
                  </p>
                )}
                <p className="mt-3 text-[0.78rem] text-muted">
                  Dashed chips were written by the model. Click a token to focus on it; the middle
                  dot <span className="font-mono">{visible(' ')}</span> marks a space.
                </p>
              </div>

              <div>
                <p className="eyebrow mb-3">How text becomes tokens</p>
                <BpeDemo tokenizer={loaded.tokenizer} />
              </div>
            </div>
          </Panel>

          <Panel
            id="tf-embeddings"
            className="scroll-mt-24"
            title="Embeddings: from ids to vectors"
            eyebrow="02"
          >
            <EmbeddingsPanel
              trace={trace}
              tokenizer={loaded.tokenizer}
              tokenEmbeddings={loaded.model.tokenEmbeddings}
              position={position}
              onPosition={s.select}
            />
          </Panel>

          <Panel
            id="tf-attention"
            className="scroll-mt-24"
            title="Attention: who reads whom"
            eyebrow="03"
          >
            <AttentionPanel
              trace={trace}
              labels={ids.map((id) => loaded.tokenizer.tokenString(id))}
              headDim={loaded.model.headDim}
              layer={s.layer}
              head={s.head}
              position={position}
              onLayer={s.setLayer}
              onHead={s.setHead}
              onPosition={s.select}
            />
          </Panel>

          <Panel
            id="tf-blocks"
            className="scroll-mt-24"
            title="The residual stream and the MLP"
            eyebrow="04"
          >
            <BlocksPanel
              trace={trace}
              tokenizer={loaded.tokenizer}
              layer={s.layer}
              position={position}
              onLayer={s.setLayer}
            />
          </Panel>

          <Panel
            id="tf-output"
            className="scroll-mt-24"
            title="Choosing the next token"
            eyebrow="05"
            action={
              <div className="flex flex-wrap items-center gap-2">
                <Button
                  size="sm"
                  onClick={s.togglePlay}
                  aria-label={playing ? 'Pause generation' : 'Generate text'}
                >
                  {playing ? <Pause className="size-4" /> : <Play className="size-4" />}
                  {playing ? 'Pause' : 'Generate'}
                </Button>
                <Button
                  size="sm"
                  variant="secondary"
                  onClick={() => {
                    s.stop();
                    s.generateOne();
                  }}
                  aria-label="Generate one token"
                >
                  <SkipForward className="size-4" /> One token
                </Button>
                <Button
                  size="sm"
                  variant="ghost"
                  onClick={s.clearGenerated}
                  aria-label="Clear generated text"
                >
                  <Eraser className="size-4" /> Clear
                </Button>
              </div>
            }
          >
            <div className="space-y-6">
              <div className="rounded-xl border border-line bg-raised/40 px-4 py-3 font-mono text-[0.92rem] leading-relaxed">
                <span className="text-muted">{prompt}</span>
                <span className="text-fg">{text}</span>
                {playing && (
                  <span className="ml-0.5 inline-block h-4 w-1.5 translate-y-0.5 animate-pulse bg-brand" />
                )}
                {finished && <span className="ml-2 text-[0.72rem] text-faint">‹end of text›</span>}
              </div>

              <div className="grid grid-cols-[minmax(0,1fr)] gap-8 lg:grid-cols-[1.25fr_1fr]">
                <NextTokenPanel
                  distribution={distribution}
                  tokenizer={loaded.tokenizer}
                  vocabSize={loaded.model.config.vocab}
                  onPick={s.append}
                  context={
                    selected !== null
                      ? `after token ${position + 1} “${visible(loaded.tokenizer.tokenString(ids[position] as number))}”`
                      : `after “${visible(loaded.tokenizer.tokenString(ids[position] as number))}”`
                  }
                />
                <SamplingPanel
                  sampling={sampling}
                  seed={seed}
                  maxNew={maxNew}
                  onSampling={s.setSampling}
                  onSeed={s.setSeed}
                  onMaxNew={s.setMaxNew}
                />
              </div>
            </div>
          </Panel>

          <Panel id="tf-3d" className="scroll-mt-24" title="The whole model in 3D" eyebrow="06">
            <ThreeDPanel
              trace={trace}
              tokenizer={loaded.tokenizer}
              distribution={distribution}
              position={position}
              onSelect={s.select}
            />
          </Panel>
        </div>
      )}
    </div>
  );
}
