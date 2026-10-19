import { RL_ALGORITHMS, RL_LAB_LIMITS, algorithmInfo, type RlLabConfig } from '@sim/deep';
import { Dices } from 'lucide-react';
import { Button, Segmented, Slider, Tex } from '@/design-system';

const ALPHAS = [0.01, 0.05, 0.1, 0.2, 0.5, 1] as const;
const DECAYS = [0.9, 0.95, 0.99, 0.995, 0.999, 1] as const;

const nearest = (ladder: readonly number[], value: number) => {
  let best = 0;
  ladder.forEach((v, i) => {
    if (Math.abs(v - value) < Math.abs((ladder[best] as number) - value)) best = i;
  });
  return best;
};

export function Controls({
  config,
  onPatch,
}: {
  config: RlLabConfig;
  onPatch: (patch: Partial<RlLabConfig>) => void;
}) {
  return (
    <div className="space-y-8">
      <div className="grid gap-6 lg:grid-cols-2">
        <div className="space-y-2">
          <Segmented
            label="Algorithm"
            size="sm"
            value={config.algorithm}
            onValueChange={(algorithm) => onPatch({ algorithm })}
            options={RL_ALGORITHMS.map((a) => ({ value: a, label: algorithmInfo[a].label }))}
          />
          <p className="text-[0.82rem] text-muted">{algorithmInfo[config.algorithm].note}</p>
        </div>
        <div className="overflow-x-auto rounded-xl border border-line bg-raised/30 px-4 py-3 text-[0.95rem]">
          <Tex tex={algorithmInfo[config.algorithm].formula} />
        </div>
      </div>

      <div className="grid gap-x-10 gap-y-6 sm:grid-cols-2 lg:grid-cols-3">
        <div className="space-y-5">
          <p className="eyebrow">The learner (apply instantly)</p>
          <div>
            <Slider
              label="Learning rate α"
              min={0}
              max={ALPHAS.length - 1}
              value={nearest(ALPHAS, config.alpha)}
              onValueChange={(i) => onPatch({ alpha: ALPHAS[i] as number })}
              format={() => String(config.alpha)}
            />
            <p className="mt-1.5 text-[0.72rem] leading-snug text-faint">
              How far each experience moves the estimate towards what just happened.
            </p>
          </div>
          <div>
            <Slider
              label="Discount γ"
              min={RL_LAB_LIMITS.gamma[0]}
              max={0.99}
              step={0.01}
              value={Math.min(0.99, config.gamma)}
              onValueChange={(gamma) => onPatch({ gamma })}
              format={(v) => v.toFixed(2)}
            />
            <p className="mt-1.5 text-[0.72rem] leading-snug text-faint">
              How much future rewards count. 0 = only the next reward; close to 1 = far-sighted.
            </p>
          </div>
        </div>

        <div className="space-y-5">
          <p className="eyebrow">Exploration (apply instantly)</p>
          <Slider
            label="Random-move rate ε at the start"
            min={0}
            max={1}
            step={0.05}
            value={config.epsilon}
            onValueChange={(epsilon) =>
              onPatch({ epsilon, epsilonMin: Math.min(config.epsilonMin, epsilon) })
            }
            format={(v) => v.toFixed(2)}
          />
          <Slider
            label="Smallest ε it decays to"
            min={0}
            max={config.epsilon}
            step={0.01}
            value={Math.min(config.epsilonMin, config.epsilon)}
            onValueChange={(epsilonMin) => onPatch({ epsilonMin })}
            format={(v) => v.toFixed(2)}
          />
          <Slider
            label="ε kept after each episode"
            min={0}
            max={DECAYS.length - 1}
            value={nearest(DECAYS, config.epsilonDecay)}
            onValueChange={(i) => onPatch({ epsilonDecay: DECAYS[i] as number })}
            format={() => String(config.epsilonDecay)}
          />
        </div>

        <div className="space-y-5">
          <p className="eyebrow">The world (restarts learning)</p>
          <div>
            <Slider
              label="Slippery floor"
              min={RL_LAB_LIMITS.slip[0]}
              max={RL_LAB_LIMITS.slip[1]}
              step={0.05}
              value={config.slip}
              onValueChange={(slip) => onPatch({ slip: Number(slip.toFixed(2)) })}
              format={(v) => `${Math.round(v * 100)}%`}
            />
            <p className="mt-1.5 text-[0.72rem] leading-snug text-faint">
              Chance that a random move happens instead of the chosen one.
            </p>
          </div>
          <Slider
            label="Reward for each step"
            min={-3}
            max={0}
            step={0.5}
            value={Math.max(-3, Math.min(0, config.stepReward))}
            onValueChange={(stepReward) => onPatch({ stepReward })}
            format={(v) => v.toFixed(1)}
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
    </div>
  );
}
