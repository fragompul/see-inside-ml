import { Check, Link2, Pause, Play, RotateCcw, SkipForward } from 'lucide-react';
import { useState } from 'react';
import { Button, Kbd, Segmented, Tooltip } from '@/design-system';
import { SPEEDS } from '../store';

interface Props {
  running: boolean;
  epoch: number;
  lr: number | undefined;
  speed: number;
  onToggle: () => void;
  onStep: () => void;
  onReset: () => void;
  onSpeed: (speed: number) => void;
  /** Copies a link to the current setup; resolves false if the browser refused. */
  onShare: () => Promise<boolean>;
}

export function Toolbar({
  running,
  epoch,
  lr,
  speed,
  onToggle,
  onStep,
  onReset,
  onSpeed,
  onShare,
}: Props) {
  const [shared, setShared] = useState<'idle' | 'copied' | 'failed'>('idle');
  const share = async () => {
    setShared((await onShare()) ? 'copied' : 'failed');
    window.setTimeout(() => setShared('idle'), 2200);
  };
  return (
    <div className="flex flex-wrap items-center gap-x-6 gap-y-3 rounded-2xl border border-line bg-surface/80 px-4 py-3 shadow-[var(--shadow-card)] backdrop-blur-sm">
      <div className="flex flex-wrap items-center gap-2">
        <Tooltip
          content={
            <span className="flex items-center gap-2">
              {running ? 'Pause' : 'Train'} <Kbd>Space</Kbd>
            </span>
          }
        >
          <Button
            onClick={onToggle}
            aria-label={running ? 'Pause training' : 'Train the network'}
            className="w-28"
          >
            {running ? <Pause className="size-4" /> : <Play className="size-4" />}
            {running ? 'Pause' : 'Train'}
          </Button>
        </Tooltip>
        <Tooltip
          content={
            <span className="flex items-center gap-2">
              One epoch <Kbd>→</Kbd>
            </span>
          }
        >
          <Button variant="secondary" onClick={onStep} aria-label="Step one epoch">
            <SkipForward className="size-4" /> Step
          </Button>
        </Tooltip>
        <Tooltip
          content={
            <span className="flex items-center gap-2">
              New weights, same data <Kbd>R</Kbd>
            </span>
          }
        >
          <Button variant="ghost" onClick={onReset} aria-label="Reset weights">
            <RotateCcw className="size-4" /> Reset
          </Button>
        </Tooltip>
        <Button variant="ghost" onClick={share} aria-label="Share a link to this setup">
          {shared === 'copied' ? (
            <Check className="size-4 text-ok" />
          ) : (
            <Link2 className="size-4" />
          )}
          {shared === 'copied'
            ? 'Link copied'
            : shared === 'failed'
              ? 'Copy the address bar'
              : 'Share'}
        </Button>
      </div>

      <div className="flex items-center gap-6">
        <div>
          <p className="eyebrow">Epoch</p>
          <p className="font-mono text-2xl leading-tight tabular-nums" data-testid="epoch">
            {String(epoch).padStart(5, '0')}
          </p>
        </div>
        <div className="hidden sm:block">
          <p className="eyebrow">Learning rate</p>
          <p className="font-mono text-2xl leading-tight tabular-nums">
            {lr === undefined ? '–' : lr.toPrecision(2)}
          </p>
        </div>
      </div>

      <div className="ml-auto w-full min-w-48 sm:w-72">
        <p className="eyebrow mb-1.5">Speed · epochs per frame</p>
        <Segmented
          label="Training speed"
          size="sm"
          value={String(speed)}
          onValueChange={(v) => onSpeed(Number(v))}
          options={SPEEDS.map((s) => ({ value: String(s), label: `${s}` }))}
        />
      </div>
    </div>
  );
}
