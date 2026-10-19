import { useId, type ReactNode } from 'react';
import { motion } from 'framer-motion';
import { ToggleGroup } from 'radix-ui';
import { cn } from './cn';
import { spring } from './motion';

export interface SegmentedOption<T extends string> {
  value: T;
  label: ReactNode;
  /** Accessible name when the label is an icon or abbreviation. */
  ariaLabel?: string;
}

/** Single-choice segmented control with a sliding highlight. */
export function Segmented<T extends string>({
  value,
  onValueChange,
  options,
  label,
  size = 'md',
  fit = false,
  className,
}: {
  value: T;
  onValueChange: (value: T) => void;
  options: SegmentedOption<T>[];
  label: string;
  size?: 'sm' | 'md';
  /** Shrink to content instead of filling the row. */
  fit?: boolean;
  className?: string;
}) {
  const id = useId();
  return (
    <ToggleGroup.Root
      type="single"
      value={value}
      // Radix emits "" when the active item is clicked again; keep the current choice.
      onValueChange={(next) => next && onValueChange(next as T)}
      aria-label={label}
      className={cn(
        'relative flex max-w-full items-center gap-0.5 overflow-x-auto rounded-full border border-line-strong bg-raised/60 p-0.5',
        fit ? 'w-fit' : 'w-full',
        className,
      )}
    >
      {options.map((option) => (
        <ToggleGroup.Item
          key={option.value}
          value={option.value}
          aria-label={option.ariaLabel}
          className={cn(
            'relative flex-1 shrink-0 cursor-pointer rounded-full font-medium whitespace-nowrap transition-colors',
            size === 'sm' ? 'h-6 px-2 text-[0.7rem]' : 'h-8 px-3 text-[0.78rem]',
            value === option.value ? 'text-brand-fg' : 'text-muted hover:text-fg',
          )}
        >
          {value === option.value && (
            <motion.span
              layoutId={`${id}-pill`}
              transition={spring}
              className="absolute inset-0 rounded-full bg-brand"
            />
          )}
          <span className="relative">{option.label}</span>
        </ToggleGroup.Item>
      ))}
    </ToggleGroup.Root>
  );
}
