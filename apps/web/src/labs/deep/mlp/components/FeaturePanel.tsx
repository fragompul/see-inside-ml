import { FEATURES, type PlaygroundConfig } from '@sim/deep';
import { Tex, Tooltip, cn } from '@/design-system';

export function FeaturePanel({
  config,
  onPatch,
}: {
  config: PlaygroundConfig;
  onPatch: (patch: Partial<PlaygroundConfig>) => void;
}) {
  const selected = new Set(config.features);

  const toggle = (id: string) => {
    if (selected.has(id)) {
      if (selected.size === 1) return; // a network needs at least one input
      onPatch({ features: config.features.filter((f) => f !== id) });
    } else {
      // keep the canonical catalogue order so the input column is stable
      onPatch({
        features: FEATURES.filter((f) => selected.has(f.id) || f.id === id).map((f) => f.id),
      });
    }
  };

  return (
    <div>
      <div className="grid grid-cols-3 gap-2">
        {FEATURES.map((feature) => {
          const on = selected.has(feature.id);
          const locked = on && selected.size === 1;
          return (
            <Tooltip
              key={feature.id}
              content={
                <span className="flex items-center gap-2">
                  <Tex tex={feature.formula} />
                  {locked && <span className="text-muted">· at least one input is required</span>}
                </span>
              }
            >
              <button
                type="button"
                aria-pressed={on}
                onClick={() => toggle(feature.id)}
                className={cn(
                  'h-9 cursor-pointer rounded-lg border px-1 font-mono text-[0.76rem] transition-all duration-200',
                  on
                    ? 'border-brand bg-brand-soft text-fg shadow-[0_0_0_3px_var(--brand-soft)]'
                    : 'border-line bg-raised/40 text-muted hover:border-line-strong hover:text-fg',
                  locked && 'cursor-not-allowed',
                )}
              >
                {feature.label}
              </button>
            </Tooltip>
          );
        })}
      </div>
      <p className="mt-3 text-[0.78rem] leading-relaxed text-muted">
        Feed the network transformed inputs. A linear model with{' '}
        <span className="font-mono text-fg">x₁²</span> and{' '}
        <span className="font-mono text-fg">x₂²</span> can already separate circles.
      </p>
    </div>
  );
}
