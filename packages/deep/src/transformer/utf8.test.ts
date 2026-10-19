import { describe, expect, it } from 'vitest';
import { decodeUtf8, encodeUtf8 } from './utf8';

describe('utf8', () => {
  it.each(['', 'plain ascii', 'café', '€ uro', '\u{1f600} emoji', 'mix é€\u{1f600}!'])(
    'round-trips %j',
    (text) => {
      expect(decodeUtf8(encodeUtf8(text))).toBe(text);
    },
  );

  it('encodes known byte sequences', () => {
    expect(encodeUtf8('A')).toEqual([0x41]);
    expect(encodeUtf8('é')).toEqual([0xc3, 0xa9]);
    expect(encodeUtf8('€')).toEqual([0xe2, 0x82, 0xac]);
    expect(encodeUtf8('\u{1f600}')).toEqual([0xf0, 0x9f, 0x98, 0x80]);
  });

  // These match Python's bytes.decode("utf-8", errors="replace"): one U+FFFD per maximal invalid prefix.
  it.each([
    [[0xc3], '�'], // lone lead byte
    [[0xa9], '�'], // lone continuation byte
    [[0xff], '�'],
    [[0xe2, 0x82], '�'], // truncated 3-byte sequence
    [[0xf0, 0x9f, 0x98], '�'], // truncated 4-byte sequence
    [[0x41, 0xc3, 0x42], 'A�B'],
    [[0xc0, 0x80], '��'], // overlong encoding is invalid
    [[0xed, 0xa0, 0x80], '���'], // UTF-16 surrogate is invalid
  ])('replaces invalid bytes %j', (bytes, expected) => {
    expect(decodeUtf8(bytes)).toBe(expected);
  });
});
