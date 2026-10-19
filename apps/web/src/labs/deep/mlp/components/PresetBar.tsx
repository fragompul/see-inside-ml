import { FlaskConical } from 'lucide-react';
import { AnimatePresence, motion } from 'framer-motion';
import { cn } from '@/design-system';
import { PRESETS, presetConfig, type Preset } from '../presets';

export function PresetBar({
  activeId,
  onApply,
}: {
  activeId: string | null;
  onApply: (preset: Preset) => void;
}) {
  const active = PRESETS.find((p) => p.id === activeId);
  return (
    <div className="rounded-2xl border border-line bg-surface/60 px-4 py-3 backdrop-blur-sm">
      <div className="flex flex-wrap items-center gap-2">
        <span className="eyebrow mr-1 flex items-center gap-1.5">
          <FlaskConical className="size-3.5" /> Experiments
        </span>
        {PRESETS.map((preset) => (
          <button
            key={preset.id}
            type="button"
            aria-pressed={preset.id === activeId}
            onClick={() => onApply(preset)}
            className={cn(
              'cursor-pointer rounded-full border px-3 py-1 text-[0.78rem] transition-all duration-200',
              preset.id === activeId
                ? 'border-brand bg-brand text-brand-fg shadow-[0_0_0_3px_var(--brand-soft)]'
                : 'border-line-strong text-muted hover:border-brand hover:text-fg',
            )}
          >
            {preset.title}
          </button>
        ))}
      </div>
      <AnimatePresence mode="wait" initial={false}>
        {active && (
          <motion.p
            key={active.id}
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            transition={{ duration: 0.25 }}
            className="overflow-hidden pt-3 text-[0.88rem] leading-relaxed text-muted"
          >
            <span className="text-fg">What to look for: </span>
            {active.lesson}
          </motion.p>
        )}
      </AnimatePresence>
    </div>
  );
}

export { presetConfig };
