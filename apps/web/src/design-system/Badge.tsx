import type { HTMLAttributes } from 'react';
import { cn } from './cn';

type Tone = 'neutral' | 'brand' | 'pos' | 'neg' | 'ok';

const tones: Record<Tone, string> = {
  neutral: 'border-line-strong text-muted',
  brand: 'border-brand/40 bg-brand-soft text-brand',
  pos: 'border-pos/40 bg-pos/10 text-pos',
  neg: 'border-neg/40 bg-neg/10 text-neg',
  ok: 'border-ok/40 bg-ok/10 text-ok',
};

export function Badge({
  tone = 'neutral',
  className,
  ...props
}: HTMLAttributes<HTMLSpanElement> & { tone?: Tone }) {
  return (
    <span
      className={cn(
        'inline-flex items-center gap-1.5 rounded-full border px-2.5 py-0.5',
        'font-mono text-[0.68rem] tracking-[0.08em] uppercase',
        tones[tone],
        className,
      )}
      {...props}
    />
  );
}

export function Kbd({ className, ...props }: HTMLAttributes<HTMLElement>) {
  return (
    <kbd
      className={cn(
        'inline-flex min-w-5 items-center justify-center rounded-md border border-line-strong',
        'bg-raised px-1.5 py-0.5 font-mono text-[0.7rem] text-muted shadow-[0_1px_0_var(--line-strong)]',
        className,
      )}
      {...props}
    />
  );
}
