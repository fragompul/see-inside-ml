import {
  DIFFUSION_LIMITS,
  DIFFUSION_LAB_LIMITS,
  GAN_DATASETS,
  SCHEDULE_KINDS,
  DENOISER_LIMITS,
  ganDatasetInfo,
  noiseScheduleInfo,
  type DiffusionLabConfig,
} from '@sim/deep';
import { Dices } from 'lucide-react';
import { Button, Segmented, Select, Slider } from '@/design-system';
import { LR_LADDER, nearestIndex } from '@/shared/lab/ladders';

const BATCHES = [32, 64, 128, 256, 512] as const;
const trim = (v: number) => String(Number(v.toPrecision(3)));

export function TrainingControls({
  config,
  onPatch,
}: {
  config: DiffusionLabConfig;
  onPatch: (patch: Partial<DiffusionLabConfig>) => void;
}) {
  return (
    <div className="grid gap-x-10 gap-y-6 sm:grid-cols-2 lg:grid-cols-3">
      <div className="space-y-5">
        <p className="eyebrow">Learning (changes apply instantly)</p>
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
        <p className="text-[0.72rem] leading-snug text-faint">
          The sampler and its number of steps are chosen in section 03.
        </p>
      </div>

      <div className="space-y-5">
        <p className="eyebrow">The noise (restarts training)</p>
        <Slider
          label="Noising steps T"
          min={DIFFUSION_LIMITS.steps[0]}
          max={DIFFUSION_LIMITS.steps[1]}
          step={10}
          value={config.steps}
          onValueChange={(steps) =>
            onPatch({ steps, sampleSteps: Math.min(config.sampleSteps, steps) })
          }
        />
        <div className="flex flex-col gap-2">
          <span className="text-[0.82rem] text-muted">Noise schedule</span>
          <Segmented
            label="Noise schedule"
            size="sm"
            value={config.schedule}
            onValueChange={(schedule) => onPatch({ schedule })}
            options={SCHEDULE_KINDS.map((k) => ({ value: k, label: noiseScheduleInfo[k].label }))}
          />
        </div>
        <Slider
          label="Hidden units in the denoiser"
          min={DENOISER_LIMITS.hidden[0]}
          max={DENOISER_LIMITS.hidden[1]}
          step={16}
          value={config.hidden}
          onValueChange={(hidden) => onPatch({ hidden })}
        />
      </div>

      <div className="space-y-5">
        <p className="eyebrow">Data (restarts training)</p>
        <Select
          label="Distribution to learn"
          value={config.dataset}
          onValueChange={(dataset) => onPatch({ dataset })}
          options={GAN_DATASETS.map((d) => ({ value: d, label: ganDatasetInfo[d].label }))}
        />
        <Button
          size="sm"
          variant="secondary"
          onClick={() => onPatch({ seed: Math.floor(Math.random() * 1e6) })}
        >
          <Dices className="size-4" /> New random start
        </Button>
        <p className="text-[0.72rem] leading-snug text-faint">
          Batch sizes up to {DIFFUSION_LAB_LIMITS.batchSize[1]} are allowed; larger batches give
          smoother but slower steps.
        </p>
      </div>
    </div>
  );
}
