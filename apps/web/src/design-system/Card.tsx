import { useRef, type HTMLAttributes, type PointerEvent } from 'react';
import { cn } from './cn';

/**
 * Surface with a "spotlight": a soft glow that follows the pointer.
 * `--mx/--my` are updated imperatively (no React re-render per mouse move).
 */
export function Card({
  className,
  onPointerMove,
  children,
  ...props
}: HTMLAttributes<HTMLDivElement>) {
  const ref = useRef<HTMLDivElement>(null);

  const handleMove = (event: PointerEvent<HTMLDivElement>) => {
    const el = ref.current;
    if (el) {
      const rect = el.getBoundingClientRect();
      el.style.setProperty('--mx', `${event.clientX - rect.left}px`);
      el.style.setProperty('--my', `${event.clientY - rect.top}px`);
    }
    onPointerMove?.(event);
  };

  return (
    <div
      ref={ref}
      onPointerMove={handleMove}
      className={cn(
        'group relative overflow-hidden rounded-2xl border border-line bg-surface shadow-[var(--shadow-card)]',
        'transition-[border-color,transform] duration-500 ease-out hover:border-line-strong',
        className,
      )}
      {...props}
    >
      <div
        aria-hidden
        className="pointer-events-none absolute inset-0 opacity-0 transition-opacity duration-500 group-hover:opacity-100"
        style={{
          background:
            'radial-gradient(360px circle at var(--mx, 50%) var(--my, 50%), var(--brand-soft), transparent 70%)',
        }}
      />
      <div className="relative h-full">{children}</div>
    </div>
  );
}
