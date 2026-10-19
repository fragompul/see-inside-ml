import { useId } from 'react';
import { ChevronDown } from 'lucide-react';
import { cn } from './cn';

export interface SelectOption<T extends string> {
  value: T;
  label: string;
}

/** Native select (best accessibility and mobile behaviour) with the design-system look. */
export function Select<T extends string>({
  label,
  value,
  onValueChange,
  options,
  className,
}: {
  label: string;
  value: T;
  onValueChange: (value: T) => void;
  options: SelectOption<T>[];
  className?: string;
}) {
  const id = useId();
  return (
    <div className={cn('flex flex-col gap-2', className)}>
      <label htmlFor={id} className="text-[0.82rem] text-muted">
        {label}
      </label>
      <div className="relative">
        <select
          id={id}
          value={value}
          onChange={(event) => onValueChange(event.target.value as T)}
          className="h-9 w-full cursor-pointer appearance-none rounded-lg border border-line-strong bg-raised/60 pr-8 pl-3 text-[0.82rem] text-fg transition-colors hover:border-brand"
        >
          {options.map((option) => (
            <option key={option.value} value={option.value}>
              {option.label}
            </option>
          ))}
        </select>
        <ChevronDown className="pointer-events-none absolute top-1/2 right-2.5 size-4 -translate-y-1/2 text-muted" />
      </div>
    </div>
  );
}
