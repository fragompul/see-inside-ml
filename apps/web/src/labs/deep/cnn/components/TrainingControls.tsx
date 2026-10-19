import { OPTIMIZER_NAMES, optimizerInfo } from '@sim/core';
import { CNN_LAB_LIMITS, CNN_LIMITS } from '@sim/deep';
import { type CnnLabConfig } from '@sim/deep';
import { Dices } from 'lucide-react';
import { Button, Segmented, Slider } from '@/design-system';
import { LR_LADDER, nearestIndex } from '@/shared/lab/ladders';

const BATCHES = [8, 16, 32, 64, 128] as const;
const trim = (v: number) => String(Number(v.toPrecision(3)));

export function TrainingControls({
  config,
  onPatch,
}: {
  config: CnnLabConfig;
  onPatch: (patch: Partial<CnnLabConfig>) => void;
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
      </div>

      <div className="space-y-5">
        <p className="eyebrow">Architecture (restarts training)</p>
        <Slider
          label="Filters in convolution 1"
          min={CNN_LIMITS.filters1[0]}
          max={CNN_LIMITS.filters1[1]}
          value={config.filters1}
          onValueChange={(filters1) => onPatch({ filters1 })}
        />
        <Slider
          label="Filters in convolution 2"
          min={CNN_LIMITS.filters2[0]}
          max={CNN_LIMITS.filters2[1]}
          value={config.filters2}
          onValueChange={(filters2) => onPatch({ filters2 })}
        />
        <div className="flex flex-col gap-2">
          <span className="text-[0.82rem] text-muted">Kernel size</span>
          <Segmented<'3' | '5'>
            label="Kernel size"
            size="sm"
            value={String(config.kernel) as '3' | '5'}
            onValueChange={(v) => onPatch({ kernel: Number(v) as 3 | 5 })}
            options={[
              { value: '3', label: '3 × 3' },
              { value: '5', label: '5 × 5' },
            ]}
          />
        </div>
      </div>

      <div className="space-y-5">
        <p className="eyebrow">Data (restarts training)</p>
        <Slider
          label="Training images per digit"
          min={CNN_LAB_LIMITS.perClass[0]}
          max={CNN_LAB_LIMITS.perClass[1]}
          step={10}
          value={config.perClass}
          onValueChange={(perClass) => onPatch({ perClass })}
        />
        <Slider
          label="Pixel noise"
          min={CNN_LAB_LIMITS.noise[0]}
          max={CNN_LAB_LIMITS.noise[1]}
          step={0.05}
          value={config.noise}
          onValueChange={(noise) => onPatch({ noise: Number(noise.toFixed(2)) })}
          format={(v) => v.toFixed(2)}
        />
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
