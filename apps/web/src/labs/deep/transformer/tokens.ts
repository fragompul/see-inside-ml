/** Makes invisible characters visible so token boundaries can be read at a glance. */
export function visible(text: string): string {
  return text.replace(/ /g, '·').replace(/\n/g, '↵').replace(/\t/g, '⇥');
}

/** Stable, well-spread hue per token position (the chip colours come from CSS via `--h`). */
export const tokenHue = (position: number): number => (position * 47 + 12) % 360;

export const EXAMPLE_PROMPTS = [
  'Anna found a ball. She',
  'Tom found a key. He',
  'The small mice play in the forest. They',
  'Mia and Leo went to the park. Mia said hello to',
  'Ben, Sam and Zoe like cakes. Ben brings a',
  'The tired fox runs in the',
] as const;

/**
 * Label for a token. A lone byte of a multi-byte character has no printable form (it decodes to
 * U+FFFD), so it is shown as its hexadecimal value instead.
 */
export function tokenLabel(text: string, id: number): string {
  if (
    text.includes('�') ||
    (text.length === 1 && text.charCodeAt(0) < 32 && text !== '\n' && text !== '\t')
  ) {
    return `<0x${id.toString(16).toUpperCase().padStart(2, '0')}>`;
  }
  return visible(text);
}
