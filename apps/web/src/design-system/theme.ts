import { useEffect } from 'react';
import { create } from 'zustand';

export type ThemePreference = 'light' | 'dark' | 'system';
export type ResolvedTheme = 'light' | 'dark';

export const THEME_STORAGE_KEY = 'sim-theme';
/** Key used before the project was renamed: still read so nobody loses their choice. */
const LEGACY_THEME_KEY = 'nnp-theme';

const isPreference = (value: unknown): value is ThemePreference =>
  value === 'light' || value === 'dark' || value === 'system';

function readStoredPreference(): ThemePreference {
  try {
    const stored =
      window.localStorage.getItem(THEME_STORAGE_KEY) ??
      window.localStorage.getItem(LEGACY_THEME_KEY);
    if (isPreference(stored)) return stored;
  } catch {
    /* storage unavailable (private mode, blocked cookies): fall back to system */
  }
  return 'system';
}

function systemPrefersDark(): boolean {
  return window.matchMedia('(prefers-color-scheme: dark)').matches;
}

export function resolveTheme(preference: ThemePreference, systemDark: boolean): ResolvedTheme {
  if (preference === 'system') return systemDark ? 'dark' : 'light';
  return preference;
}

let changingTimer: number | undefined;

function applyTheme(resolved: ResolvedTheme, animate: boolean) {
  const root = document.documentElement;
  if (animate) {
    root.setAttribute('data-theme-changing', '');
    window.clearTimeout(changingTimer);
    changingTimer = window.setTimeout(() => root.removeAttribute('data-theme-changing'), 500);
  }
  root.dataset.theme = resolved;
}

interface ThemeState {
  preference: ThemePreference;
  resolved: ResolvedTheme;
  setPreference: (preference: ThemePreference) => void;
}

export const useThemeStore = create<ThemeState>((set) => {
  const preference = readStoredPreference();
  return {
    preference,
    resolved: resolveTheme(preference, systemPrefersDark()),
    setPreference: (next) => {
      try {
        window.localStorage.setItem(THEME_STORAGE_KEY, next);
      } catch {
        /* non-fatal: the choice just won't persist */
      }
      const resolved = resolveTheme(next, systemPrefersDark());
      applyTheme(resolved, true);
      set({ preference: next, resolved });
    },
  };
});

/** Mount once at the app root: syncs the DOM and follows OS changes in "system" mode. */
export function useThemeSync() {
  useEffect(() => {
    applyTheme(useThemeStore.getState().resolved, false);
    const media = window.matchMedia('(prefers-color-scheme: dark)');
    const onChange = () => {
      const { preference } = useThemeStore.getState();
      if (preference !== 'system') return;
      const resolved = resolveTheme(preference, media.matches);
      applyTheme(resolved, true);
      useThemeStore.setState({ resolved });
    };
    media.addEventListener('change', onChange);
    return () => media.removeEventListener('change', onChange);
  }, []);
}

export const useTheme = () => useThemeStore();
