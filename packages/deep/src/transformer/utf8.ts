/**
 * Minimal UTF-8 codec. Written by hand (instead of TextEncoder/TextDecoder) so the engine has no
 * dependency on platform globals and so invalid byte sequences behave exactly like Python's
 * `errors="replace"`: each maximal invalid subpart becomes one U+FFFD. Needed because BPE tokens
 * can split a multi-byte character in the middle.
 */

export function encodeUtf8(text: string): number[] {
  const bytes: number[] = [];
  for (const char of text) {
    const code = char.codePointAt(0) as number;
    if (code < 0x80) {
      bytes.push(code);
    } else if (code < 0x800) {
      bytes.push(0xc0 | (code >> 6), 0x80 | (code & 0x3f));
    } else if (code < 0x10000) {
      bytes.push(0xe0 | (code >> 12), 0x80 | ((code >> 6) & 0x3f), 0x80 | (code & 0x3f));
    } else {
      bytes.push(
        0xf0 | (code >> 18),
        0x80 | ((code >> 12) & 0x3f),
        0x80 | ((code >> 6) & 0x3f),
        0x80 | (code & 0x3f),
      );
    }
  }
  return bytes;
}

const REPLACEMENT = '�';

export function decodeUtf8(bytes: ArrayLike<number>): string {
  let out = '';
  let i = 0;
  const n = bytes.length;
  const at = (j: number) => bytes[j] as number;
  const isCont = (j: number, lo = 0x80, hi = 0xbf) => j < n && at(j) >= lo && at(j) <= hi;

  while (i < n) {
    const b = at(i);
    if (b < 0x80) {
      out += String.fromCharCode(b);
      i += 1;
    } else if (b >= 0xc2 && b <= 0xdf) {
      if (isCont(i + 1)) {
        out += String.fromCodePoint(((b & 0x1f) << 6) | (at(i + 1) & 0x3f));
        i += 2;
      } else {
        out += REPLACEMENT;
        i += 1;
      }
    } else if (b >= 0xe0 && b <= 0xef) {
      // the second byte has a narrower valid range for E0 (no overlongs) and ED (no surrogates)
      const lo = b === 0xe0 ? 0xa0 : 0x80;
      const hi = b === 0xed ? 0x9f : 0xbf;
      if (isCont(i + 1, lo, hi)) {
        if (isCont(i + 2)) {
          out += String.fromCodePoint(
            ((b & 0x0f) << 12) | ((at(i + 1) & 0x3f) << 6) | (at(i + 2) & 0x3f),
          );
          i += 3;
        } else {
          out += REPLACEMENT;
          i += 2;
        }
      } else {
        out += REPLACEMENT;
        i += 1;
      }
    } else if (b >= 0xf0 && b <= 0xf4) {
      const lo = b === 0xf0 ? 0x90 : 0x80;
      const hi = b === 0xf4 ? 0x8f : 0xbf;
      if (isCont(i + 1, lo, hi)) {
        if (isCont(i + 2)) {
          if (isCont(i + 3)) {
            out += String.fromCodePoint(
              ((b & 0x07) << 18) |
                ((at(i + 1) & 0x3f) << 12) |
                ((at(i + 2) & 0x3f) << 6) |
                (at(i + 3) & 0x3f),
            );
            i += 4;
          } else {
            out += REPLACEMENT;
            i += 3;
          }
        } else {
          out += REPLACEMENT;
          i += 2;
        }
      } else {
        out += REPLACEMENT;
        i += 1;
      }
    } else {
      out += REPLACEMENT;
      i += 1;
    }
  }
  return out;
}
