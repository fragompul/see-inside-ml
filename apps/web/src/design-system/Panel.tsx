import type { HTMLAttributes, ReactNode } from 'react';
import { cn } from './cn';

/** Titled container for the controls and visualizations of a lab. */
export function Panel({
  title,
  eyebrow,
  action,
  className,
  children,
  ...props
}: Omit<HTMLAttributes<HTMLElement>, 'title'> & {
  title: string;
  eyebrow?: string;
  action?: ReactNode;
}) {
  return (
    <section
      className={cn(
        'rounded-2xl border border-line bg-surface/80 shadow-[var(--shadow-card)] backdrop-blur-sm',
        className,
      )}
      {...props}
    >
      <header className="flex min-h-14 flex-wrap items-center justify-between gap-x-4 gap-y-2 border-b border-line px-5 py-3">
        <div className="flex items-baseline gap-2.5">
          {eyebrow && <span className="eyebrow text-brand">{eyebrow}</span>}
          <h2 className="font-display text-[1.02rem] font-semibold tracking-tight">{title}</h2>
        </div>
        {action}
      </header>
      <div className="p-5">{children}</div>
    </section>
  );
}
