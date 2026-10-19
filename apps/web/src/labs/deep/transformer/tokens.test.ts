import { describe, expect, it } from 'vitest';
import { tokenHue, tokenLabel, visible } from './tokens';

describe('token display helpers', () => {
  it('makes spaces, newlines and tabs visible', () => {
    expect(visible(' cat')).toBe('·cat');
    expect(visible('a\nb\tc')).toBe('a↵b⇥c');
  });

  it('shows lone bytes of multi-byte characters as hex instead of the replacement character', () => {
    expect(tokenLabel('�', 0xc3)).toBe('<0xC3>');
    expect(tokenLabel('�', 0x0a9)).toBe('<0xA9>');
    expect(tokenLabel('\u0001', 1)).toBe('<0x01>');
  });

  it('keeps normal text, spaces and newlines readable', () => {
    expect(tokenLabel(' found', 287)).toBe('·found');
    expect(tokenLabel('\n', 10)).toBe('↵');
  });

  it('spreads hues so neighbouring tokens get different colours', () => {
    const hues = Array.from({ length: 12 }, (_, i) => tokenHue(i));
    expect(new Set(hues).size).toBe(12);
    expect(hues.every((h) => h >= 0 && h < 360)).toBe(true);
    expect(tokenHue(0)).not.toBe(tokenHue(1));
  });
});
