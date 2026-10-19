import { motion } from 'framer-motion';
import { Monitor, Moon, Sun, type LucideIcon } from 'lucide-react';
import { ToggleGroup } from 'radix-ui';
import { cn } from './cn';
import { Tooltip } from './Tooltip';
import { spring } from './motion';
import { useTheme, type ThemePreference } from './theme';

const options: { value: ThemePreference; label: string; Icon: LucideIcon }[] = [
  { value: 'light', label: 'Light theme', Icon: Sun },
  { value: 'system', label: 'Match system', Icon: Monitor },
  { value: 'dark', label: 'Dark theme', Icon: Moon },
];

/** Light / System / Dark selector with a sliding highlight. */
export function ThemeToggle() {
  const { preference, setPreference } = useTheme();
  return (
    <ToggleGroup.Root
      type="single"
      value={preference}
      // Radix emits "" when the active item is clicked again; keep the current choice.
      onValueChange={(value) => value && setPreference(value as ThemePreference)}
      aria-label="Color theme"
      className="relative flex items-center gap-0.5 rounded-full border border-line-strong bg-raised/60 p-0.5 backdrop-blur"
    >
      {options.map(({ value, label, Icon }) => (
        <Tooltip key={value} content={label} side="bottom">
          <ToggleGroup.Item
            value={value}
            aria-label={label}
            // Not `data-[state=on]`: Tooltip.Trigger also writes data-state and would override it.
            className={cn(
              'relative grid size-7 cursor-pointer place-items-center rounded-full transition-colors',
              preference === value ? 'text-brand-fg' : 'text-muted hover:text-fg',
            )}
          >
            {preference === value && (
              <motion.span
                layoutId="theme-pill"
                transition={spring}
                className="absolute inset-0 rounded-full bg-brand"
              />
            )}
            <Icon className="relative size-3.5" strokeWidth={2.2} />
          </ToggleGroup.Item>
        </Tooltip>
      ))}
    </ToggleGroup.Root>
  );
}
