import { TASK_NAMES, argmax, taskInfo } from '@sim/deep';
import { Dices, Shuffle } from 'lucide-react';
import { Button, Segmented, Slider, cn } from '@/design-system';
import { useRnnStore } from '../store';
import { relevantTokens } from '../view';

/** Colour of a token symbol (the five class colours of the design system). */
export const tokenColor = (token: number) => `var(--class-${token % 5})`;

/** Task picker, the sequence as clickable chips, and the network's answer. */
export function SequencePanel() {
  const s = useRnnStore();
  const { config, tokens, trace, label, source } = s;
  const info = taskInfo[config.task];
  const probs = trace?.probs ?? [];
  const best = probs.length ? argmax(probs) : -1;
  const relevant = new Set(relevantTokens(config.task, tokens));

  return (
    <div className="space-y-6">
      <div className="grid gap-5 lg:grid-cols-[minmax(0,1fr)_minmax(0,18rem)] lg:gap-10">
        <div className="space-y-3">
          <Segmented
            label="Task"
            size="sm"
            value={config.task}
            onValueChange={(task) => s.patch({ task })}
            options={TASK_NAMES.map((t) => ({ value: t, label: taskInfo[t].label }))}
          />
          <p className="text-[0.9rem] text-fg">
            <span className="text-muted">Rule: </span>
            {info.rule}
          </p>
          <p className="max-w-2xl text-[0.82rem] text-muted">{info.note}</p>
        </div>
        <Slider
          label="Sequence length (restarts training)"
          min={info.minLength}
          max={info.maxLength}
          value={config.seqLen}
          onValueChange={(seqLen) => s.patch({ seqLen })}
        />
      </div>

      <div>
        <p className="eyebrow mb-3">
          The network reads {tokens.length} tokens, one per step
          <span className="text-faint"> · click a token to change it</span>
        </p>
        <ol className="flex flex-wrap gap-1.5" aria-label="Input sequence">
          {Array.from(tokens, (token, i) => (
            <li key={i}>
              <button
                type="button"
                onClick={() => s.cycleToken(i)}
                aria-label={`Token ${i + 1}: ${info.symbols[token]}. Click to change it.`}
                style={{ ['--c' as string]: tokenColor(token) }}
                className={cn(
                  'relative grid size-10 cursor-pointer place-items-center rounded-lg border font-mono text-[0.95rem] font-medium transition-transform duration-150 hover:-translate-y-0.5',
                  'border-[color:var(--c)] bg-[color-mix(in_srgb,var(--c)_16%,transparent)] text-fg',
                )}
              >
                {info.symbols[token]}
                {relevant.has(i) && (
                  <span
                    aria-hidden
                    className="absolute -bottom-1.5 left-1/2 h-[3px] w-4 -translate-x-1/2 rounded-full bg-brand"
                  />
                )}
              </button>
              <span className="mt-2 block text-center font-mono text-[0.62rem] text-faint">
                {i + 1}
              </span>
            </li>
          ))}
        </ol>
        <p className="mt-2 text-[0.74rem] text-faint">
          The dash marks the tokens the rule depends on; the rest is noise to be ignored.
        </p>
        <div className="mt-4 flex flex-wrap gap-2">
          <Button size="sm" variant="secondary" onClick={s.newSequence}>
            <Dices className="size-4" /> New random sequence
          </Button>
          <Button
            size="sm"
            variant="ghost"
            onClick={() => s.loadSample(Math.floor(Math.random() * 200))}
          >
            <Shuffle className="size-4" /> Random test example
          </Button>
        </div>
      </div>

      <div className="grid gap-6 lg:grid-cols-[auto_minmax(0,1fr)] lg:gap-10">
        <div>
          <p className="eyebrow mb-1">Its answer</p>
          <p className="font-display text-5xl leading-none font-semibold tracking-tight">
            {info.classSymbols[best] ?? '?'}
            <span className="ml-3 font-mono text-base font-normal text-muted">
              {((probs[best] ?? 0) * 100).toFixed(0)}% sure
            </span>
          </p>
          <p className="mt-2 text-[0.8rem] text-muted">
            {label === null ? (
              <span className="text-neg">
                This sequence breaks the rule (there is no ★ with a symbol after it).
              </span>
            ) : (
              <>
                The rule says <span className="font-mono text-fg">{info.classSymbols[label]}</span>
                {best === label ? (
                  <span className="text-ok"> · correct</span>
                ) : (
                  <span className="text-neg"> · wrong</span>
                )}
                {source === 'edited' && <span className="text-faint"> · edited</span>}
              </>
            )}
          </p>
        </div>
        <ol className="space-y-1.5" aria-label="Probability of each answer">
          {info.classSymbols.map((symbol, k) => {
            const p = probs[k] ?? 0;
            const top = k === best;
            return (
              <li key={symbol} className="flex items-center gap-3">
                <span
                  className={cn(
                    'w-9 text-right font-mono text-sm',
                    top ? 'font-semibold text-fg' : 'text-muted',
                  )}
                >
                  {symbol}
                </span>
                <div className="h-5 grow overflow-hidden rounded-md bg-raised/70">
                  <div
                    className={cn(
                      'h-full rounded-md transition-[width] duration-200 ease-out',
                      top ? 'bg-brand' : 'bg-line-strong',
                    )}
                    style={{ width: `${p * 100}%` }}
                  />
                </div>
                <span className="w-12 text-right font-mono text-[0.74rem] text-muted tabular-nums">
                  {(p * 100).toFixed(1)}%
                </span>
              </li>
            );
          })}
        </ol>
      </div>
      {s.epoch === 0 && (
        <p className="rounded-xl border border-line bg-raised/40 px-4 py-3 text-[0.82rem] text-muted">
          This network has not learned anything yet, so its answers are guesses. Try an experiment
          above, or scroll down to train it.
        </p>
      )}
    </div>
  );
}
