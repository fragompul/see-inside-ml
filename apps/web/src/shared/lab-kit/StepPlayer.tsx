import { Pause, Play, SkipBack, SkipForward } from 'lucide-react';
import { useEffect, useState } from 'react';
import { Button, Slider } from '@/design-system';

/**
 * Plays, pauses and scrubs through the states of an iterative algorithm (tree growth, boosting
 * rounds, EM iterations…). The caller keeps the states; this only owns the cursor.
 */
export function StepPlayer({
  count,
  step,
  onStep,
  label = 'Step',
  intervalMs = 450,
  describe,
}: {
  /** Number of states (steps are 0 … count − 1). */
  count: number;
  step: number;
  onStep: (step: number) => void;
  label?: string;
  intervalMs?: number;
  describe?: (step: number) => string;
}) {
  const [playing, setPlaying] = useState(false);
  const last = Math.max(0, count - 1);
  // playback ends by itself at the last state (derived, so no state update in an effect)
  const running = playing && step < last;

  useEffect(() => {
    if (!running) return;
    const id = window.setInterval(() => onStep(Math.min(last, step + 1)), intervalMs);
    return () => window.clearInterval(id);
  }, [running, step, last, onStep, intervalMs]);

  const lower = label.toLowerCase();
  return (
    <div className="space-y-3">
      <div className="flex flex-wrap items-center gap-2">
        <Button
          size="sm"
          variant="secondary"
          aria-label={`${label}: back`}
          onClick={() => {
            setPlaying(false);
            onStep(Math.max(0, step - 1));
          }}
          disabled={step <= 0}
        >
          <SkipBack className="size-4" />
        </Button>
        <Button
          size="sm"
          aria-label={running ? `Pause ${lower}s` : `Play ${lower}s`}
          onClick={() => {
            if (!running && step >= last) onStep(0);
            setPlaying(!running);
          }}
          className="w-24"
        >
          {running ? <Pause className="size-4" /> : <Play className="size-4" />}
          {running ? 'Pause' : step >= last ? 'Replay' : 'Play'}
        </Button>
        <Button
          size="sm"
          variant="secondary"
          aria-label={`${label}: forward`}
          onClick={() => {
            setPlaying(false);
            onStep(Math.min(last, step + 1));
          }}
          disabled={step >= last}
        >
          <SkipForward className="size-4" />
        </Button>
        <span className="font-mono text-[0.78rem] text-muted tabular-nums" data-testid="step-label">
          {lower} {step} / {last}
        </span>
      </div>
      <Slider
        label={label}
        min={0}
        max={last}
        value={Math.min(step, last)}
        onValueChange={(v) => {
          setPlaying(false);
          onStep(v);
        }}
        format={(v) => (describe ? describe(v) : `${v} / ${last}`)}
      />
    </div>
  );
}
