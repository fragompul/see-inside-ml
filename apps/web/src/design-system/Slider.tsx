import { useId } from 'react';
import { Slider as S } from 'radix-ui';
import { cn } from './cn';

export interface SliderProps {
  label: string;
  value: number;
  onValueChange: (value: number) => void;
  min: number;
  max: number;
  step?: number;
  /** Formats the readout next to the label (defaults to the raw number). */
  format?: (value: number) => string;
  className?: string;
}

/** Labeled slider with a live monospace readout. Built on Radix (full keyboard + ARIA support). */
export function Slider({
  label,
  value,
  onValueChange,
  min,
  max,
  step = 1,
  format = String,
  className,
}: SliderProps) {
  const id = useId();
  return (
    <div className={cn('flex flex-col gap-2.5', className)}>
      <div className="flex items-baseline justify-between">
        <label htmlFor={id} className="text-[0.82rem] text-muted">
          {label}
        </label>
        <output className="font-mono text-[0.8rem] tabular-nums text-fg">{format(value)}</output>
      </div>
      <S.Root
        value={[value]}
        onValueChange={([next]) => next !== undefined && onValueChange(next)}
        min={min}
        max={max}
        step={step}
        className="relative flex h-5 w-full touch-none items-center select-none"
      >
        <S.Track className="relative h-1 grow rounded-full bg-line-strong">
          <S.Range className="absolute h-full rounded-full bg-gradient-to-r from-neg via-brand to-pos" />
        </S.Track>
        <S.Thumb
          id={id}
          aria-label={label}
          className="block size-4 cursor-grab rounded-full border-2 border-brand bg-bg shadow-[0_0_0_4px_var(--brand-soft)] transition-shadow duration-200 hover:shadow-[0_0_0_7px_var(--brand-soft)] active:cursor-grabbing"
        />
      </S.Root>
    </div>
  );
}
