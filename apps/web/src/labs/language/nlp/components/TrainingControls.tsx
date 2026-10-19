import { NLP_LAB_LIMITS, W2V_LIMITS, type NlpLabConfig } from '@sim/deep';
import { Dices } from 'lucide-react';
import { Button, Slider } from '@/design-system';

const LRS = [0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5] as const;
const BATCHES = [16, 32, 64, 128, 256] as const;

const nearest = (ladder: readonly number[], value: number) => {
  let best = 0;
  ladder.forEach((v, i) => {
    if (Math.abs(Math.log(v / value)) < Math.abs(Math.log((ladder[best] as number) / value)))
      best = i;
  });
  return best;
};

export function TrainingControls({
  config,
  onPatch,
}: {
  config: NlpLabConfig;
  onPatch: (patch: Partial<NlpLabConfig>) => void;
}) {
  return (
    <div className="grid gap-x-10 gap-y-6 sm:grid-cols-2 lg:grid-cols-3">
      <div className="space-y-5">
        <p className="eyebrow">Learning (changes apply instantly)</p>
        <div>
          <Slider
            label="Learning rate"
            min={0}
            max={LRS.length - 1}
            value={nearest(LRS, config.lr)}
            onValueChange={(i) => onPatch({ lr: LRS[i] as number })}
            format={() => String(config.lr)}
          />
          <p className="mt-1.5 text-[0.72rem] leading-snug text-faint">
            Per training pair. Try 0.5 to see training blow up.
          </p>
        </div>
        <Slider
          label="Random words per real pair"
          min={W2V_LIMITS.negatives[0]}
          max={W2V_LIMITS.negatives[1]}
          value={config.negatives}
          onValueChange={(negatives) => onPatch({ negatives })}
        />
        <Slider
          label="Pairs per step"
          min={0}
          max={BATCHES.length - 1}
          value={Math.max(0, BATCHES.indexOf(config.batchSize as (typeof BATCHES)[number]))}
          onValueChange={(i) => onPatch({ batchSize: BATCHES[i] as number })}
          format={() => String(config.batchSize)}
        />
      </div>

      <div className="space-y-5">
        <p className="eyebrow">The vectors (restart training)</p>
        <Slider
          label="Numbers per word (dimensions)"
          min={W2V_LIMITS.dim[0]}
          max={W2V_LIMITS.dim[1]}
          value={config.dim}
          onValueChange={(dim) => onPatch({ dim })}
        />
        <div>
          <Slider
            label="Context window (words on each side)"
            min={W2V_LIMITS.window[0]}
            max={W2V_LIMITS.window[1]}
            value={config.window}
            onValueChange={(window) => onPatch({ window })}
          />
          <p className="mt-1.5 text-[0.72rem] leading-snug text-faint">
            Narrow windows capture syntax; wide ones capture topic.
          </p>
        </div>
      </div>

      <div className="space-y-5">
        <p className="eyebrow">The corpus (restarts training)</p>
        <Slider
          label="Sentences in the corpus"
          min={NLP_LAB_LIMITS.sentences[0]}
          max={NLP_LAB_LIMITS.sentences[1]}
          step={100}
          value={config.sentences}
          onValueChange={(sentences) => onPatch({ sentences })}
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
