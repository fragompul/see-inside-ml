import { useId } from 'react';

/** Brand mark: a tiny two-in / two-out network. Gradient follows the neg → brand → pos scale. */
export function LogoMark({ size = 28 }: { size?: number }) {
  const id = useId();
  return (
    <svg width={size} height={size} viewBox="0 0 32 32" role="img" aria-label="See Inside ML">
      <defs>
        <linearGradient id={id} x1="0" y1="0" x2="1" y2="1">
          <stop offset="0" stopColor="var(--neg)" />
          <stop offset="0.55" stopColor="var(--brand)" />
          <stop offset="1" stopColor="var(--pos)" />
        </linearGradient>
      </defs>
      <g stroke={`url(#${id})`} strokeWidth="1.6" strokeLinecap="round" fill="none">
        <path d="M7 9.5 16 16M7 22.5 16 16M16 16l9-6.5M16 16l9 6.5" />
      </g>
      <g fill={`url(#${id})`}>
        <circle cx="7" cy="9.5" r="2.4" />
        <circle cx="7" cy="22.5" r="2.4" />
        <circle cx="25" cy="9.5" r="2.4" />
        <circle cx="25" cy="22.5" r="2.4" />
        <circle cx="16" cy="16" r="3.4" />
      </g>
    </svg>
  );
}

export function Wordmark() {
  return (
    <span className="flex items-center gap-2.5">
      <LogoMark />
      <span className="font-display text-[1.05rem] font-semibold tracking-tight">
        See<span className="text-muted">·</span>Inside<span className="text-muted">·</span>ML
      </span>
    </span>
  );
}
