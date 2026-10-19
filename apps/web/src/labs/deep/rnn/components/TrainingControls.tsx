import { OPTIMIZER_NAMES, optimizerInfo } from '@sim/core';
import {
  CELL_NAMES,
  RECURRENT_LIMITS,
  RNN_LAB_LIMITS,
  cellInfo,
  type RnnLabConfig,
} from '@sim/deep';
import { Dices } from 'lucide-react';
import { Button, Segmented, Slider } from '@/design-system';
import { LR_LADDER, nearestIndex } from '@/shared/lab/ladders';

const BATCHES = [4, 8, 16, 32, 64, 128] as const;
const trim = (v: number) => String(Number(v.toPrecision(3)));

export function TrainingControls({
  config,
  onPatch,
}: {
  config: RnnLabConfig;
  onPatch: (patch: Partial<RnnLabConfig>) => void;
}) {
  return (
    <div className="grid gap-x-10 gap-y-6 sm:grid-cols-2 lg:grid-cols-3">
      <div className="space-y-5">
        <p className="eyebrow">Learning (changes apply instantly)</p>
        <div className="flex flex-col gap-2">
          <span className="text-[0.82rem] text-muted">Optimizer</span>
          <Segmented
            label="Optimizer"
            size="sm"
            value={config.optimizer}
            onValueChange={(optimizer) => onPatch({ optimizer })}
            options={OPTIMIZER_NAMES.map((o) => ({ value: o, label: optimizerInfo[o].label }))}
          />
        </div>
        <Slider
          label="Learning rate"
          min={0}
          max={LR_LADDER.length - 1}
          value={nearestIndex(LR_LADDER, config.lr)}
          onValueChange={(i) => onPatch({ lr: LR_LADDER[i] as number })}
          format={() => trim(config.lr)}
        />
        <Slider
          label="Batch size"
          min={0}
          max={BATCHES.length - 1}
          value={Math.max(0, BATCHES.indexOf(config.batchSize as (typeof BATCHES)[number]))}
          onValueChange={(i) => onPatch({ batchSize: BATCHES[i] as number })}
          format={() => String(config.batchSize)}
        />
        <div>
          <Slider
            label="Gradient clipping"
            min={RNN_LAB_LIMITS.clip[0]}
            max={RNN_LAB_LIMITS.clip[1]}
            step={0.5}
            value={config.clip}
            onValueChange={(clip) => onPatch({ clip })}
            format={(v) => (v === 0 ? 'off' : v.toFixed(1))}
          />
          <p className="mt-1.5 text-[0.72rem] leading-snug text-faint">
            Caps the size of a weight update, which keeps recurrent networks from blowing up.
          </p>
        </div>
      </div>

      <div className="space-y-5">
        <p className="eyebrow">Architecture (restarts training)</p>
        <div className="flex flex-col gap-2">
          <span className="text-[0.82rem] text-muted">Cell</span>
          <Segmented
            label="Cell"
            size="sm"
            value={config.cell}
            onValueChange={(cell) => onPatch({ cell })}
            options={CELL_NAMES.map((c) => ({
              value: c,
              label: c === 'rnn' ? 'RNN' : c.toUpperCase(),
            }))}
          />
          <p className="text-[0.72rem] leading-snug text-faint">{cellInfo[config.cell].note}</p>
        </div>
        <Slider
          label="Hidden units"
          min={RECURRENT_LIMITS.hidden[0]}
          max={RECURRENT_LIMITS.hidden[1]}
          value={config.hidden}
          onValueChange={(hidden) => onPatch({ hidden })}
        />
      </div>

      <div className="space-y-5">
        <p className="eyebrow">Data (restarts training)</p>
        <Slider
          label="Training sequences"
          min={RNN_LAB_LIMITS.trainSize[0]}
          max={RNN_LAB_LIMITS.trainSize[1]}
          step={32}
          value={config.trainSize}
          onValueChange={(trainSize) => onPatch({ trainSize })}
        />
        <p className="text-[0.72rem] leading-snug text-faint">
          The task and the sequence length are chosen in section 01.
        </p>
        <Button
          size="sm"
          variant="secondary"
          onClick={() => onPatch({ seed: Math.floor(Math.random() * 1e6) })}
        >
          <Dices className="size-4" /> New random start
        </Button>
      </div>
    </div>
  );
}
