import { type SamplingOptions } from '@sim/deep';
import { Dices } from 'lucide-react';
import { Button, Slider } from '@/design-system';
import { MAX_NEW_TOKENS } from '../store';

interface Props {
  sampling: SamplingOptions;
  seed: number;
  maxNew: number;
  onSampling: (patch: Partial<SamplingOptions>) => void;
  onSeed: (seed: number) => void;
  onMaxNew: (n: number) => void;
}

export function SamplingPanel({ sampling, seed, maxNew, onSampling, onSeed, onMaxNew }: Props) {
  return (
    <div className="flex flex-col gap-5 self-start">
      <Slider
        label="Temperature"
        min={0}
        max={2}
        step={0.05}
        value={sampling.temperature}
        onValueChange={(temperature) => onSampling({ temperature })}
        format={(v) => (v === 0 ? '0 · greedy' : v.toFixed(2))}
      />
      <Slider
        label="Top-k (keep the k likeliest)"
        min={0}
        max={40}
        step={1}
        value={sampling.topK}
        onValueChange={(topK) => onSampling({ topK })}
        format={(v) => (v === 0 ? 'off' : String(v))}
      />
      <Slider
        label="Top-p (nucleus mass)"
        min={0.05}
        max={1}
        step={0.05}
        value={sampling.topP}
        onValueChange={(topP) => onSampling({ topP })}
        format={(v) => (v >= 1 ? 'off' : v.toFixed(2))}
      />
      <Slider
        label="Max new tokens"
        min={1}
        max={MAX_NEW_TOKENS}
        step={1}
        value={maxNew}
        onValueChange={onMaxNew}
      />
      <div>
        <Button
          variant="secondary"
          size="sm"
          onClick={() => onSeed(Math.floor(Math.random() * 1_000_000))}
        >
          <Dices className="size-4" /> New random seed
          <span className="ml-1 font-mono text-[0.7rem] text-muted">#{seed}</span>
        </Button>
      </div>
    </div>
  );
}
