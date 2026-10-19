import { fireEvent, render, screen } from '@testing-library/react';
import { beforeEach, describe, expect, it } from 'vitest';
import { THEME_STORAGE_KEY, resolveTheme, useThemeStore } from './theme';
import { ThemeToggle } from './ThemeToggle';
import { TooltipProvider } from './Tooltip';

describe('resolveTheme', () => {
  it('follows the system only in "system" mode', () => {
    expect(resolveTheme('system', true)).toBe('dark');
    expect(resolveTheme('system', false)).toBe('light');
    expect(resolveTheme('light', true)).toBe('light');
    expect(resolveTheme('dark', false)).toBe('dark');
  });
});

describe('theme store', () => {
  beforeEach(() => {
    window.localStorage.clear();
    useThemeStore.getState().setPreference('system');
  });

  it('applies and persists an explicit choice', () => {
    useThemeStore.getState().setPreference('dark');
    expect(document.documentElement.dataset.theme).toBe('dark');
    expect(window.localStorage.getItem(THEME_STORAGE_KEY)).toBe('dark');

    useThemeStore.getState().setPreference('light');
    expect(document.documentElement.dataset.theme).toBe('light');
    expect(useThemeStore.getState().resolved).toBe('light');
  });
});

describe('<ThemeToggle />', () => {
  it('switches the theme and exposes the current choice accessibly', () => {
    render(
      <TooltipProvider>
        <ThemeToggle />
      </TooltipProvider>,
    );
    fireEvent.click(screen.getByRole('radio', { name: 'Dark theme' }));
    expect(document.documentElement.dataset.theme).toBe('dark');
    expect(screen.getByRole('radio', { name: 'Dark theme' })).toHaveAttribute(
      'aria-checked',
      'true',
    );

    fireEvent.click(screen.getByRole('radio', { name: 'Light theme' }));
    expect(document.documentElement.dataset.theme).toBe('light');
  });

  it('keeps the selection when the active option is clicked again', () => {
    render(
      <TooltipProvider>
        <ThemeToggle />
      </TooltipProvider>,
    );
    const dark = screen.getByRole('radio', { name: 'Dark theme' });
    fireEvent.click(dark);
    fireEvent.click(dark);
    expect(useThemeStore.getState().preference).toBe('dark');
  });
});
