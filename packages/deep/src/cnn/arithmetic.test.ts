import { describe, expect, it } from 'vitest';
import {
  analyseStack,
  conv,
  convParameters,
  outputSize,
  pool,
  samePadding,
  tracedField,
  transposedSize,
  type Window,
} from './arithmetic';

// the output size found by counting the positions where the window fits
const counted = (n: number, w: Window): number => {
  let c = 0;
  const span = w.dilation * (w.kernel - 1) + 1;
  for (let start = -w.padding; start + span <= n + w.padding; start += w.stride) c++;
  return c;
};

describe('output size', () => {
  it('matches counting the window positions for many settings', () => {
    for (const n of [7, 16, 28, 33])
      for (const k of [1, 2, 3, 5])
        for (const s of [1, 2, 3])
          for (const p of [0, 1, 2])
            for (const d of [1, 2]) {
              const w = conv(k, s, p, d);
              if (w.dilation * (k - 1) + 1 > n + 2 * p) continue;
              expect(outputSize(n, w)).toBe(counted(n, w));
            }
  });

  it('"same" padding keeps the size, and halving needs stride 2', () => {
    expect(outputSize(28, conv(3, 1, samePadding(3)))).toBe(28);
    expect(outputSize(28, conv(5, 1, samePadding(5)))).toBe(28);
    expect(outputSize(28, conv(3, 1, samePadding(3, 4), 4))).toBe(28);
    expect(outputSize(28, conv(3, 2, 1))).toBe(14);
    expect(outputSize(28, pool(2))).toBe(14);
  });

  it('a transposed convolution undoes the shape change of a stride-2 convolution', () => {
    const w = conv(4, 2, 1);
    expect(transposedSize(outputSize(32, w), w)).toBe(32);
  });
});

describe('receptive field', () => {
  const stacks: Record<string, Window[]> = {
    plain: [conv(3, 1, 1), conv(3, 1, 1), conv(3, 1, 1)],
    pooled: [conv(3, 1, 1), pool(2), conv(3, 1, 1), pool(2), conv(3, 1, 1)],
    strided: [conv(3, 2, 1), conv(5, 1, 2), conv(3, 2, 1), conv(3, 1, 1)],
    dilated: [1, 2, 4, 8, 16].map((d) => conv(3, 1, d, d)),
    mixed: [conv(7, 2, 3), pool(3, 2), conv(3, 1, 2, 2), conv(3, 1, 1)],
  };
  for (const [name, layers] of Object.entries(stacks))
    it(`${name}: the formula equals tracing the dependencies back`, () => {
      const stats = analyseStack(401, layers);
      expect(tracedField(401, layers)).toBe(stats.field[stats.field.length - 1]);
    });

  it('three 3-wide layers see 7 positions, and dilation doubling grows exponentially', () => {
    expect(analyseStack(100, stacks.plain as Window[]).field.at(-1)).toBe(7);
    expect(
      analyseStack(
        2000,
        [1, 2, 4, 8, 16, 32, 64, 128].map((d) => conv(3, 1, d, d)),
      ).field.at(-1),
    ).toBe(511);
  });
});

describe('cost', () => {
  it('counts weights and biases', () => {
    expect(convParameters(3, 16, 32)).toBe(3 * 16 * 32 + 32);
    expect(convParameters(3, 4, 5, 2)).toBe(9 * 4 * 5 + 5);
  });
});
