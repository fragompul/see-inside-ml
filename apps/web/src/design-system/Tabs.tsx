import { useId, useState, type ReactNode } from 'react';
import { motion } from 'framer-motion';
import { Tabs as T } from 'radix-ui';
import { spring } from './motion';

export interface TabItem {
  value: string;
  label: string;
  content: ReactNode;
}

/** Tabs with a shared-layout sliding underline (mounted only on the active trigger). */
export function Tabs({ items, defaultValue }: { items: TabItem[]; defaultValue?: string }) {
  const groupId = useId();
  const [active, setActive] = useState(defaultValue ?? items[0]?.value ?? '');
  return (
    <T.Root value={active} onValueChange={setActive}>
      <T.List className="flex gap-1 border-b border-line">
        {items.map((item) => (
          <T.Trigger
            key={item.value}
            value={item.value}
            className="relative cursor-pointer px-4 py-2.5 text-sm text-muted transition-colors hover:text-fg data-[state=active]:text-fg"
          >
            {item.label}
            {active === item.value && (
              <motion.span
                layoutId={`${groupId}-underline`}
                transition={spring}
                className="absolute inset-x-2 -bottom-px h-0.5 bg-gradient-to-r from-neg via-brand to-pos"
              />
            )}
          </T.Trigger>
        ))}
      </T.List>
      {items.map((item) => (
        <T.Content key={item.value} value={item.value} className="pt-5">
          {item.content}
        </T.Content>
      ))}
    </T.Root>
  );
}
