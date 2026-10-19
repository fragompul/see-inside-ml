import type { ReactNode } from 'react';

/** The sticky row of section links at the top of a lab. */
export function StageNav({ stages }: { stages: readonly (readonly [string, string])[] }) {
  return (
    <nav
      aria-label="Sections of the lab"
      className="sticky top-16 z-30 -mx-4 mb-6 overflow-x-auto border-b border-line bg-bg/80 px-4 py-2.5 backdrop-blur-xl sm:-mx-6 sm:px-6 lg:-mx-8 lg:px-8"
    >
      <ol className="flex min-w-max items-center gap-1 font-mono text-[0.72rem]">
        {stages.map(([id, label], i) => (
          <li key={id} className="flex items-center gap-1">
            {i > 0 && <span className="text-faint">→</span>}
            <a
              href={`#${id}`}
              onClick={(e) => {
                e.preventDefault();
                document.getElementById(id)?.scrollIntoView({ behavior: 'smooth', block: 'start' });
              }}
              className="rounded-full px-3 py-1 text-muted transition-colors hover:bg-brand-soft hover:text-fg"
            >
              <span className="mr-1.5 text-brand">{String(i + 1).padStart(2, '0')}</span>
              {label}
            </a>
          </li>
        ))}
      </ol>
    </nav>
  );
}

/** Title block shared by the new labs. */
export function LabHeader({
  eyebrow,
  title,
  accent,
  blurb,
  badges,
}: {
  eyebrow: string;
  title: string;
  accent: string;
  blurb: string;
  badges?: ReactNode;
}) {
  return (
    <header className="mb-8">
      <p className="eyebrow text-brand">{eyebrow}</p>
      <h1 className="mt-2 font-display text-[clamp(1.9rem,3.4vw,2.8rem)] leading-none font-semibold tracking-[-0.03em]">
        {title} <span className="font-serif font-normal italic text-muted">{accent}</span>
      </h1>
      <p className="mt-3 max-w-2xl text-[0.95rem] text-muted">{blurb}</p>
      {badges && <div className="mt-4 flex flex-wrap gap-2">{badges}</div>}
    </header>
  );
}
