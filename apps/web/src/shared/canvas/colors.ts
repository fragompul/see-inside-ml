import { useMemo } from 'react';
import { useTheme } from '@/design-system';

export type Rgb = readonly [number, number, number];

export interface Palette {
  bg: Rgb;
  surface: Rgb;
  raised: Rgb;
  fg: Rgb;
  muted: Rgb;
  line: string;
  pos: Rgb;
  neg: Rgb;
  brand: Rgb;
  classes: Rgb[];
  /** CSS strings for the same colours (SVG / canvas strokes). */
  css: {
    pos: string;
    neg: string;
    brand: string;
    fg: string;
    muted: string;
    surface: string;
    classes: string[];
  };
}

let scratch: CanvasRenderingContext2D | null = null;

/** Normalises any CSS colour (hex, rgb(), hsl()…) to [r, g, b] using the browser's own parser. */
export function parseColor(value: string, fallback: Rgb = [128, 128, 128]): Rgb {
  scratch ??= document.createElement('canvas').getContext('2d', { willReadFrequently: true });
  if (!scratch) return fallback;
  scratch.clearRect(0, 0, 1, 1);
  scratch.fillStyle = '#000';
  scratch.fillStyle = value;
  scratch.fillRect(0, 0, 1, 1);
  const data = scratch.getImageData(0, 0, 1, 1).data;
  return [data[0] as number, data[1] as number, data[2] as number];
}

const rgbCss = (c: Rgb) => `rgb(${c[0]}, ${c[1]}, ${c[2]})`;

/** Reads the current theme's tokens from CSS custom properties. */
export function readPalette(): Palette {
  const styles = getComputedStyle(document.documentElement);
  const token = (name: string) => styles.getPropertyValue(name).trim();
  const color = (name: string) => parseColor(token(name));
  const classes = [0, 1, 2, 3, 4].map((i) => color(`--class-${i}`));
  const pos = color('--pos');
  const neg = color('--neg');
  const brand = color('--brand');
  const fg = color('--fg');
  const muted = color('--muted');
  const surface = color('--surface');
  return {
    bg: color('--bg'),
    surface,
    raised: color('--raised'),
    fg,
    muted,
    line: token('--line-strong') || 'rgba(128,128,128,0.4)',
    pos,
    neg,
    brand,
    classes,
    css: {
      pos: rgbCss(pos),
      neg: rgbCss(neg),
      brand: rgbCss(brand),
      fg: rgbCss(fg),
      muted: rgbCss(muted),
      surface: rgbCss(surface),
      classes: classes.map(rgbCss),
    },
  };
}

/** Palette that follows the active theme (re-read when the user switches). */
export function usePalette(): Palette {
  const { resolved } = useTheme();
  // `resolved` is the dependency on purpose: tokens change when the theme does.
  // eslint-disable-next-line react-hooks/exhaustive-deps
  return useMemo(() => readPalette(), [resolved]);
}

export function mix(a: Rgb, b: Rgb, t: number): Rgb {
  const k = Math.min(1, Math.max(0, t));
  return [a[0] + (b[0] - a[0]) * k, a[1] + (b[1] - a[1]) * k, a[2] + (b[2] - a[2]) * k];
}

const luminance = (c: Rgb) => {
  const [r, g, b] = c.map((v) => {
    const s = v / 255;
    return s <= 0.03928 ? s / 12.92 : ((s + 0.055) / 1.055) ** 2.4;
  }) as [number, number, number];
  return 0.2126 * r + 0.7152 * g + 0.0722 * b;
};

/** WCAG contrast ratio of two colours. */
export function contrast(a: Rgb, b: Rgb): number {
  const [hi, lo] = [luminance(a), luminance(b)].sort((x, y) => y - x) as [number, number];
  return (hi + 0.05) / (lo + 0.05);
}

/** Near-black or white, whichever reads better on every colour a painted cell may end up on. */
export function inkOn(...bgs: Rgb[]): string {
  const worst = (ink: Rgb) => Math.min(...bgs.map((b) => contrast(b, ink)));
  return worst([10, 10, 20]) >= worst([255, 255, 255]) ? '#0a0a14' : '#ffffff';
}
