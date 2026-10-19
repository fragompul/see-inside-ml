import { type BpeTokenizer, type ForwardTrace, type NextTokenDistribution } from '@sim/deep';
import { ChevronLeft, ChevronRight, Orbit, Pause, Play } from 'lucide-react';
import { Suspense, lazy, useEffect, useMemo, useRef, useState } from 'react';
import { Button, cn } from '@/design-system';
import { tokenLabel } from '../tokens';
import { buildSlabs, buildStages } from './layout';
import type { HoverInfo } from './Scene';

// The 3-D engine is a big chunk: it is only fetched when this panel is about to be seen.
const Scene = lazy(() => import('./Scene'));

interface Props {
  trace: ForwardTrace;
  tokenizer: BpeTokenizer;
  distribution: NextTokenDistribution;
  /** Token position in focus. */
  position: number;
  onSelect: (token: number) => void;
}

const AUTOPLAY_MS = 3200;

function webglAvailable(): boolean {
  try {
    const canvas = document.createElement('canvas');
    return Boolean(canvas.getContext('webgl2') ?? canvas.getContext('webgl'));
  } catch {
    return false;
  }
}

const pct = (p: number) => (p >= 0.1 ? `${(p * 100).toFixed(0)}%` : `${(p * 100).toFixed(1)}%`);

/**
 * The whole model as one 3-D object. The residual stream is a tunnel of slabs (a column per token,
 * a cell per dimension); attention heads hang under their layer; the probabilities stand at the end.
 */
export function ThreeDPanel({ trace, tokenizer, distribution, position, onSelect }: Props) {
  const hostRef = useRef<HTMLDivElement>(null);
  // Without IntersectionObserver (very old browsers, some test runners) just load right away.
  const [near, setNear] = useState(() => typeof IntersectionObserver === 'undefined');
  const [supported] = useState(webglAvailable);
  const [reducedMotion] = useState(
    () =>
      typeof window !== 'undefined' &&
      window.matchMedia('(prefers-reduced-motion: reduce)').matches,
  );
  const [stage, setStage] = useState<number | null>(null);
  const [playing, setPlaying] = useState(false);
  const [hover, setHover] = useState<HoverInfo | null>(null);

  const slabs = useMemo(() => buildSlabs(trace), [trace]);
  const stages = useMemo(() => buildStages(trace, slabs), [trace, slabs]);
  const tokenStrings = useMemo(
    () => trace.ids.map((id) => tokenizer.tokenString(id)),
    [trace, tokenizer],
  );
  const selected = Math.min(position, trace.ids.length - 1);
  const current = stage === null ? null : stages[stage];

  // Start loading the 3-D chunk shortly before the panel scrolls into view.
  useEffect(() => {
    const el = hostRef.current;
    if (!el) return;
    if (near) return;
    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry?.isIntersecting) setNear(true);
      },
      { rootMargin: '500px 0px' },
    );
    observer.observe(el);
    return () => observer.disconnect();
  }, [near]);

  useEffect(() => {
    if (!playing) return;
    const timer = window.setInterval(() => {
      setStage((s) => (s === null || s >= stages.length - 1 ? 0 : s + 1));
    }, AUTOPLAY_MS);
    return () => window.clearInterval(timer);
  }, [playing, stages.length]);

  const go = (to: number | null) => {
    setPlaying(false);
    setStage(to === null ? null : Math.max(0, Math.min(stages.length - 1, to)));
  };

  const tooltip = (h: HoverInfo): { title: string; lines: string[] } => {
    switch (h.kind) {
      case 'cell':
        return {
          title: h.slab.title,
          lines: [
            `token ${h.token + 1}: “${tokenLabel(tokenStrings[h.token] ?? '', trace.ids[h.token] as number)}”`,
            `dimension ${h.row}`,
            `value ${h.value.toFixed(3)}`,
          ],
        };
      case 'head': {
        const q = tokenLabel(tokenStrings[h.cell.query] ?? '', trace.ids[h.cell.query] as number);
        const k = tokenLabel(tokenStrings[h.cell.key] ?? '', trace.ids[h.cell.key] as number);
        return {
          title: `Layer ${h.layer + 1} · head ${h.head + 1}`,
          lines: [`“${q}” reads “${k}”`, `attention ${pct(h.cell.weight)}`],
        };
      }
      case 'bar':
        return {
          title: `“${tokenLabel(tokenizer.tokenString(h.id), h.id)}”`,
          lines: [
            `probability ${pct(h.prob)}`,
            h.kept ? 'can be sampled' : 'filtered out by top-k / top-p',
          ],
        };
    }
  };

  if (!supported) {
    return (
      <p
        role="status"
        className="rounded-xl border border-line bg-raised/30 p-5 text-sm text-muted"
      >
        Your browser cannot create a WebGL context, so the 3-D view is unavailable. The five
        sections above show the same model in 2-D.
      </p>
    );
  }

  return (
    <div>
      <div
        ref={hostRef}
        data-testid="three-host"
        data-stage={stage === null ? 'overview' : stages[stage]?.id}
        className="relative h-[34rem] max-h-[75vh] min-h-[22rem] overflow-hidden rounded-xl border border-line-strong bg-raised/25"
        onPointerLeave={() => setHover(null)}
      >
        {near ? (
          <Suspense
            fallback={
              <div
                className="grid h-full place-items-center font-mono text-xs text-muted"
                aria-busy="true"
              >
                Loading the 3-D engine…
              </div>
            }
          >
            <Scene
              trace={trace}
              slabs={slabs}
              stages={stages}
              stage={stage}
              selected={selected}
              tokenStrings={tokenStrings}
              distribution={distribution}
              tokenName={(id) => tokenLabel(tokenizer.tokenString(id), id)}
              reducedMotion={reducedMotion}
              onSelect={onSelect}
              onHover={setHover}
            />
          </Suspense>
        ) : (
          <div className="grid h-full place-items-center font-mono text-xs text-muted">
            <span className="flex items-center gap-2">
              <Orbit className="size-4" /> The 3-D view loads when you scroll here
            </span>
          </div>
        )}

        <p className="pointer-events-none absolute top-3 left-4 font-mono text-[0.68rem] text-muted">
          drag to orbit · scroll to zoom · click a column to follow a token
        </p>

        {hover && (
          <div
            className="pointer-events-none absolute z-10 min-w-40 rounded-lg border border-line-strong bg-raised/95 px-3 py-2 font-mono text-[0.7rem] shadow-[var(--shadow-card)] backdrop-blur"
            style={{ left: Math.min(hover.x + 14, 9999), top: hover.y + 14 }}
          >
            <p className="text-fg">{tooltip(hover).title}</p>
            {tooltip(hover).lines.map((line) => (
              <p key={line} className="text-muted">
                {line}
              </p>
            ))}
          </div>
        )}
      </div>

      <div
        className="mt-4 flex flex-wrap items-center gap-2"
        role="group"
        aria-label="Tour of the model"
      >
        <Button
          size="sm"
          variant={stage === null ? 'primary' : 'secondary'}
          aria-pressed={stage === null}
          onClick={() => go(null)}
        >
          Overview
        </Button>
        {stages.map((s, i) => (
          <button
            key={s.id}
            type="button"
            aria-pressed={stage === i}
            onClick={() => go(i)}
            className={cn(
              'cursor-pointer rounded-full border px-3 py-1 text-[0.74rem] transition-all duration-200',
              stage === i
                ? 'border-brand bg-brand text-brand-fg shadow-[0_0_0_3px_var(--brand-soft)]'
                : 'border-line-strong text-muted hover:border-brand hover:text-fg',
            )}
          >
            <span className="mr-1.5 font-mono text-[0.66rem]">{i + 1}</span>
            {s.title}
          </button>
        ))}
        <span className="ml-auto flex items-center gap-1">
          <Button
            size="sm"
            variant="secondary"
            aria-pressed={playing}
            aria-label={playing ? 'Pause the tour' : 'Play the tour'}
            onClick={() => {
              setPlaying((p) => !p);
              setStage((s) => s ?? 0);
            }}
          >
            {playing ? <Pause className="size-3.5" /> : <Play className="size-3.5" />}
            {playing ? 'Pause' : 'Tour'}
          </Button>
          <Button
            size="sm"
            variant="ghost"
            aria-label="Previous stage"
            disabled={stage === null || stage === 0}
            onClick={() => go((stage ?? 0) - 1)}
          >
            <ChevronLeft className="size-4" />
          </Button>
          <Button
            size="sm"
            variant="ghost"
            aria-label="Next stage"
            disabled={stage === stages.length - 1}
            onClick={() => go(stage === null ? 0 : stage + 1)}
          >
            <ChevronRight className="size-4" />
          </Button>
        </span>
      </div>

      <div
        className="mt-4 min-h-[5.5rem] max-w-3xl text-[0.9rem] leading-relaxed text-muted"
        aria-live="polite"
      >
        {current ? (
          <>
            <h3 className="mb-1 font-display text-base font-semibold text-fg">
              {(stage ?? 0) + 1}. {current.title}
            </h3>
            <p>{current.text}</p>
          </>
        ) : (
          <p>
            The tunnel is the <span className="text-fg">residual stream</span>: every slab is one
            stage of the forward pass, one column per token and one cell per number. Press{' '}
            <span className="text-fg">Tour</span> to fly through it, or pick a stage. Cells use the
            same colours as everywhere else: ember positive, frost negative.
          </p>
        )}
      </div>
    </div>
  );
}
