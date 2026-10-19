import { OPTIMIZER_NAMES, optimizerInfo } from '@sim/core';
import { AE_LAB_LIMITS, AE_LIMITS, AE_MODES, type AutoencoderLabConfig } from '@sim/deep';
import { Dices } from 'lucide-react';
import { Button, Segmented, Slider } from '@/design-system';
import { LR_LADDER, nearestIndex } from '@/shared/lab/ladders';

const BATCHES = [8, 16, 32, 64, 128] as const;
const trim = (v: number) => String(Number(v.toPrecision(3)));

export function TrainingControls({
  config,
  onPatch,
}: {
  config: AutoencoderLabConfig;
  onPatch: (patch: Partial<AutoencoderLabConfig>) => void;
}) {
  const vae = config.mode === 'vae';
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
            label="Noise added to training inputs"
            min={AE_LAB_LIMITS.corruption[0]}
            max={AE_LAB_LIMITS.corruption[1]}
            step={0.05}
            value={config.corruption}
            onValueChange={(corruption) => onPatch({ corruption: Number(corruption.toFixed(2)) })}
            format={(v) => v.toFixed(2)}
          />
          <p className="mt-1.5 text-[0.72rem] leading-snug text-faint">
            Above 0 this is a <em>denoising</em> autoencoder: it must rebuild the clean image from a
            damaged one.
          </p>
        </div>
      </div>

      <div className="space-y-5">
        <p className="eyebrow">Model (restarts training)</p>
        <div className="flex flex-col gap-2">
          <span className="text-[0.82rem] text-muted">Kind</span>
          <Segmented
            label="Model kind"
            size="sm"
            value={config.mode}
            onValueChange={(mode) => onPatch({ mode })}
            options={AE_MODES.map((m) => ({
              value: m,
              label: m === 'ae' ? 'Autoencoder' : 'Variational (VAE)',
            }))}
          />
        </div>
        <Slider
          label="Numbers in the code"
          min={AE_LIMITS.latent[0]}
          max={AE_LIMITS.latent[1]}
          value={config.latent}
          onValueChange={(latent) => onPatch({ latent })}
        />
        <Slider
          label="Hidden units"
          min={16}
          max={AE_LIMITS.hidden[1]}
          step={16}
          value={Math.max(16, config.hidden)}
          onValueChange={(hidden) => onPatch({ hidden })}
        />
        {vae && (
          <div>
            <Slider
              label="β (weight of the KL term)"
              min={AE_LAB_LIMITS.beta[0]}
              max={AE_LAB_LIMITS.beta[1]}
              step={0.25}
              value={config.beta}
              onValueChange={(beta) => onPatch({ beta })}
              format={(v) => v.toFixed(2)}
            />
            <p className="mt-1.5 text-[0.72rem] leading-snug text-faint">
              High β forces tidy codes at the cost of blurrier pictures; applies instantly.
            </p>
          </div>
        )}
      </div>

      <div className="space-y-5">
        <p className="eyebrow">Data (restarts training)</p>
        <Slider
          label="Training images per digit"
          min={AE_LAB_LIMITS.perClass[0]}
          max={AE_LAB_LIMITS.perClass[1]}
          step={10}
          value={config.perClass}
          onValueChange={(perClass) => onPatch({ perClass })}
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
