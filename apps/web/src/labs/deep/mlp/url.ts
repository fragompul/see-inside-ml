import { DEFAULT_CONFIG, validatePlaygroundConfig, type PlaygroundConfig } from '@sim/deep';

/** The query parameter that carries a shared configuration. */
export const CONFIG_PARAM = 'c';

const KEYS = Object.keys(DEFAULT_CONFIG) as (keyof PlaygroundConfig)[];

function toBase64Url(text: string): string {
  const bytes = new TextEncoder().encode(text);
  let binary = '';
  for (const byte of bytes) binary += String.fromCharCode(byte);
  return btoa(binary).replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, '');
}

function fromBase64Url(value: string): string {
  const padded = value
    .replace(/-/g, '+')
    .replace(/_/g, '/')
    .padEnd(Math.ceil(value.length / 4) * 4, '=');
  const binary = atob(padded);
  return new TextDecoder().decode(Uint8Array.from(binary, (c) => c.charCodeAt(0)));
}

/**
 * Compact, shareable form of a configuration: only the fields that differ from the defaults.
 * Returns null when the configuration IS the default (no parameter needed).
 */
export function encodeConfig(config: PlaygroundConfig): string | null {
  const diff: Partial<PlaygroundConfig> = {};
  for (const key of KEYS) {
    if (JSON.stringify(config[key]) !== JSON.stringify(DEFAULT_CONFIG[key])) {
      (diff as Record<string, unknown>)[key] = config[key];
    }
  }
  return Object.keys(diff).length === 0 ? null : toBase64Url(JSON.stringify(diff));
}

/**
 * Parses a shared configuration. Never throws: anything malformed, unknown or invalid
 * (URLs are untrusted input) yields null, and the caller falls back to the defaults.
 */
export function decodeConfig(param: string | null | undefined): PlaygroundConfig | null {
  if (!param) return null;
  try {
    const parsed: unknown = JSON.parse(fromBase64Url(param));
    if (typeof parsed !== 'object' || parsed === null || Array.isArray(parsed)) return null;
    const merged: Record<string, unknown> = { ...DEFAULT_CONFIG };
    for (const key of KEYS) {
      if (Object.prototype.hasOwnProperty.call(parsed, key)) {
        merged[key] = (parsed as Record<string, unknown>)[key];
      }
    }
    const config = merged as unknown as PlaygroundConfig;
    validatePlaygroundConfig(config);
    return config;
  } catch {
    return null;
  }
}
