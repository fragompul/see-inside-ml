import { Link, NavLink, Outlet } from 'react-router-dom';
import { GithubIcon, LinkedinIcon, Tooltip, ThemeToggle, Wordmark, cn } from '@/design-system';
import { LabsMenu } from './LabsMenu';

export const REPO_URL = 'https://github.com/fragompul/see-inside-ml';

const socials = [
  { href: 'https://www.linkedin.com/in/frangomezpulido', label: 'LinkedIn', Icon: LinkedinIcon },
  { href: 'https://github.com/fragompul', label: 'GitHub profile', Icon: GithubIcon },
];

/** NavLink className callback; `extra` classes apply in every state. */
const navLink =
  (extra?: string) =>
  ({ isActive }: { isActive: boolean }) =>
    cn(
      'rounded-full px-3 py-1.5 text-sm transition-colors',
      isActive ? 'text-fg' : 'text-muted hover:text-fg',
      extra,
    );

export function Layout() {
  return (
    <div className="relative flex min-h-screen flex-col">
      <header className="sticky top-0 z-40 border-b border-line bg-bg/70 backdrop-blur-xl">
        <div className="mx-auto flex h-16 max-w-7xl items-center justify-between px-5 sm:px-8">
          <Link to="/" aria-label="See Inside ML, home">
            <Wordmark />
          </Link>
          <nav aria-label="Primary" className="flex items-center gap-1">
            <NavLink to="/" end className={navLink('max-sm:hidden')}>
              Home
            </NavLink>
            <LabsMenu />
            <NavLink to="/design" className={navLink('max-sm:hidden')}>
              Design system
            </NavLink>
            <Tooltip content="Source on GitHub" side="bottom">
              <a
                href={REPO_URL}
                target="_blank"
                rel="noreferrer"
                aria-label="GitHub repository"
                className="mx-1 grid size-9 place-items-center rounded-full text-muted transition-colors hover:text-fg max-sm:hidden"
              >
                <GithubIcon className="size-[18px]" />
              </a>
            </Tooltip>
            <ThemeToggle />
          </nav>
        </div>
      </header>

      <main className="flex-1">
        <Outlet />
      </main>

      <footer className="border-t border-line">
        <div className="mx-auto flex max-w-7xl flex-col items-start justify-between gap-3 px-5 py-8 text-sm text-muted sm:flex-row sm:items-center sm:px-8">
          <div className="flex items-center gap-3">
            <p>
              Built by <span className="text-fg">Francisco Javier Gómez Pulido</span>. MIT licensed.
            </p>
            <div className="flex items-center gap-1">
              {socials.map(({ href, label, Icon }) => (
                <Tooltip key={label} content={label}>
                  <a
                    href={href}
                    target="_blank"
                    rel="noreferrer"
                    aria-label={label}
                    className="grid size-9 place-items-center rounded-full text-muted transition-colors hover:bg-brand-soft hover:text-fg"
                  >
                    <Icon className="size-[18px]" />
                  </a>
                </Tooltip>
              ))}
            </div>
          </div>
          <p className="font-mono text-xs text-faint">Every model here runs in your browser.</p>
        </div>
      </footer>
    </div>
  );
}
