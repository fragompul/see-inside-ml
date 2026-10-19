import {
  GAN_LAB_LIMITS,
  GAN_LIMITS,
  GENERATOR_LOSSES,
  generatorLossInfo,
  type GanLabConfig,
} from '@sim/deep';
import { Dices } from 'lucide-react';
import { Button, Segmented, Slider } from '@/design-system';
import { LR_LADDER, nearestIndex } from '@/shared/lab/ladders';

const BATCHES = [16, 32, 64, 128, 256, 512] as const;
const trim = (v: number) => String(Number(v.toPrecision(3)));

export function TrainingControls({
  config,
  onPatch,
}: {
  config: GanLabConfig;
  onPatch: (patch: Partial<GanLabConfig>) => void;
}) {
  const lrIndex = (lr: number) => nearestIndex(LR_LADDER, lr);
  return (
    <div className="grid gap-x-10 gap-y-6 sm:grid-cols-2 lg:grid-cols-3">
      <div className="space-y-5">
        <p className="eyebrow">The two players (apply instantly)</p>
        <Slider
          label="Discriminator learning rate"
          min={0}
          max={LR_LADDER.length - 1}
          value={lrIndex(config.lrD)}
          onValueChange={(i) => onPatch({ lrD: LR_LADDER[i] as number })}
          format={() => trim(config.lrD)}
        />
        <Slider
          label="Generator learning rate"
          min={0}
          max={LR_LADDER.length - 1}
          value={lrIndex(config.lrG)}
          onValueChange={(i) => onPatch({ lrG: LR_LADDER[i] as number })}
          format={() => trim(config.lrG)}
        />
        <div>
          <Slider
            label="Discriminator steps per generator step"
            min={GAN_LAB_LIMITS.dSteps[0]}
            max={GAN_LAB_LIMITS.dSteps[1]}
            value={config.dSteps}
            onValueChange={(dSteps) => onPatch({ dSteps })}
          />
          <p className="mt-1.5 text-[0.72rem] leading-snug text-faint">
            More steps make a stronger critic, which is not always better.
          </p>
        </div>
      </div>

      <div className="space-y-5">
        <p className="eyebrow">The loss (applies instantly)</p>
        <div className="flex flex-col gap-2">
          <span className="text-[0.82rem] text-muted">Generator loss</span>
          <Segmented
            label="Generator loss"
            size="sm"
            value={config.lossG}
            onValueChange={(lossG) => onPatch({ lossG })}
            options={GENERATOR_LOSSES.map((l) => ({ value: l, label: generatorLossInfo[l].label }))}
          />
          <p className="text-[0.72rem] leading-snug text-faint">
            {generatorLossInfo[config.lossG].note}
          </p>
        </div>
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
        <p className="eyebrow">Networks (restart training)</p>
        <Slider
          label="Hidden units (both networks)"
          min={GAN_LIMITS.hidden[0]}
          max={GAN_LIMITS.hidden[1]}
          step={4}
          value={config.hidden}
          onValueChange={(hidden) => onPatch({ hidden })}
        />
        <Slider
          label="Size of the noise the generator reads"
          min={GAN_LIMITS.noiseDim[0]}
          max={GAN_LIMITS.noiseDim[1]}
          value={config.noiseDim}
          onValueChange={(noiseDim) => onPatch({ noiseDim })}
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
