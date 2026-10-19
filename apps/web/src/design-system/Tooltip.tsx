import type { ReactNode } from 'react';
import { Tooltip as T } from 'radix-ui';

export const TooltipProvider = ({ children }: { children: ReactNode }) => (
  <T.Provider delayDuration={150} skipDelayDuration={300}>
    {children}
  </T.Provider>
);

/** Accessible tooltip (keyboard focus + hover). `content` may be rich JSX (e.g. a formula). */
export function Tooltip({
  content,
  children,
  side = 'top',
}: {
  content: ReactNode;
  children: ReactNode;
  side?: 'top' | 'right' | 'bottom' | 'left';
}) {
  return (
    <T.Root>
      <T.Trigger asChild>{children}</T.Trigger>
      <T.Portal>
        <T.Content
          side={side}
          sideOffset={8}
          className="z-50 max-w-xs rounded-lg border border-line-strong bg-raised px-3 py-2 text-[0.8rem] leading-snug text-fg shadow-[var(--shadow-card)] data-[state=delayed-open]:animate-[tip_0.18s_ease-out]"
        >
          {content}
          <T.Arrow className="fill-[var(--raised)]" />
        </T.Content>
      </T.Portal>
    </T.Root>
  );
}
