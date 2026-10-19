import { ChevronDown } from 'lucide-react';
import { DropdownMenu } from 'radix-ui';
import { useNavigate } from 'react-router-dom';
import { cn } from '@/design-system';
import { activeTracks, labsOf } from './registry';

/**
 * Every lab grouped by track, in learning order, in one menu: a row of links no longer fits.
 * Modules that are not built yet are listed (so the path stays visible) but disabled.
 */
export function LabsMenu() {
  const navigate = useNavigate();
  return (
    <DropdownMenu.Root>
      <DropdownMenu.Trigger
        className={cn(
          'flex cursor-pointer items-center gap-1 rounded-full px-3 py-1.5 text-sm text-muted transition-colors outline-none',
          'hover:text-fg data-[state=open]:text-fg',
        )}
      >
        Labs
        <ChevronDown aria-hidden className="size-3.5" />
      </DropdownMenu.Trigger>
      <DropdownMenu.Portal>
        <DropdownMenu.Content
          align="end"
          sideOffset={10}
          aria-label="Labs"
          className="z-50 max-h-[min(34rem,calc(100vh-5rem))] w-[min(22rem,calc(100vw-2rem))] overflow-y-auto rounded-2xl border border-line-strong bg-surface p-2 shadow-[var(--shadow-card)] backdrop-blur-xl"
        >
          {activeTracks().map((track) => (
            <DropdownMenu.Group key={track.id} className="mb-1">
              <DropdownMenu.Label className="eyebrow px-3 pt-3 pb-1">
                {track.title}
              </DropdownMenu.Label>
              {labsOf(track.id).map((m, i) => {
                const live = m.status === 'live' && m.path;
                return (
                  <DropdownMenu.Item
                    key={m.id}
                    disabled={!live}
                    onSelect={() => live && navigate(m.path as string)}
                    className={cn(
                      'flex cursor-pointer items-start gap-3 rounded-xl px-3 py-1.5 outline-none',
                      'data-[highlighted]:bg-brand-soft data-[disabled]:cursor-default data-[disabled]:opacity-50',
                    )}
                  >
                    <span className="mt-0.5 font-mono text-[0.66rem] text-brand">
                      {String(i + 1).padStart(2, '0')}
                    </span>
                    <span className="min-w-0">
                      <span className="block text-[0.86rem] font-medium text-fg">
                        {m.title}
                        {!live && (
                          <span className="ml-2 font-mono text-[0.62rem] font-normal text-faint">
                            soon
                          </span>
                        )}
                      </span>
                      <span className="block text-[0.72rem] leading-snug text-muted">
                        {m.level}
                      </span>
                    </span>
                  </DropdownMenu.Item>
                );
              })}
            </DropdownMenu.Group>
          ))}
        </DropdownMenu.Content>
      </DropdownMenu.Portal>
    </DropdownMenu.Root>
  );
}
