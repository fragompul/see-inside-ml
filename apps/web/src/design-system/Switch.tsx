import { useId } from 'react';
import { Switch as S } from 'radix-ui';

export function Switch({
  label,
  checked,
  onCheckedChange,
}: {
  label: string;
  checked: boolean;
  onCheckedChange: (checked: boolean) => void;
}) {
  const id = useId();
  return (
    <div className="flex items-center justify-between gap-4">
      <label htmlFor={id} className="cursor-pointer text-[0.82rem] text-muted">
        {label}
      </label>
      <S.Root
        id={id}
        checked={checked}
        onCheckedChange={onCheckedChange}
        className="relative h-6 w-11 shrink-0 cursor-pointer rounded-full border border-line-strong bg-raised transition-colors duration-300 data-[state=checked]:border-brand data-[state=checked]:bg-brand"
      >
        <S.Thumb className="block size-[18px] translate-x-[3px] rounded-full bg-fg shadow transition-transform duration-300 ease-out data-[state=checked]:translate-x-[22px] data-[state=checked]:bg-brand-fg" />
      </S.Root>
    </div>
  );
}
