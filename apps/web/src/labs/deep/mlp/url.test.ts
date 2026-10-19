import { DEFAULT_CONFIG, type PlaygroundConfig } from '@sim/deep';
import { describe, expect, it } from 'vitest';
import { decodeConfig, encodeConfig } from './url';

const custom: PlaygroundConfig = {
  ...DEFAULT_CONFIG,
  dataset: 'spiral',
  classes: 3,
  features: ['X1', 'X2', 'sin(X1)'],
  hidden: [{ neurons: 8, activation: 'relu' }],
  lr: 0.003,
  regularization: 'l2',
  seed: 777,
};

describe('config URL codec', () => {
  it('round-trips a configuration', () => {
    const encoded = encodeConfig(custom);
    expect(encoded).toMatch(/^[A-Za-z0-9_-]+$/);
    expect(decodeConfig(encoded)).toEqual(custom);
  });

  it('omits the parameter for the default configuration and keeps links short', () => {
    expect(encodeConfig(DEFAULT_CONFIG)).toBeNull();
    expect(encodeConfig({ ...DEFAULT_CONFIG, lr: 0.1 })?.length).toBeLessThan(30);
  });

  it.each([
    ['empty', ''],
    ['null', null],
    ['not base64', '%%%'],
    ['not json', btoa('hello')],
    ['array', btoa('[1,2]')],
    ['string', btoa('"x"')],
    ['invalid values', btoa(JSON.stringify({ lr: -1 }))],
    ['unknown dataset', btoa(JSON.stringify({ dataset: 'nope' }))],
    [
      'too many layers',
      btoa(JSON.stringify({ hidden: Array(50).fill({ neurons: 2, activation: 'relu' }) })),
    ],
  ])('rejects malformed input safely: %s', (_name, input) => {
    expect(decodeConfig(input)).toBeNull();
  });

  it('ignores unknown keys instead of trusting them', () => {
    const decoded = decodeConfig(btoa(JSON.stringify({ lr: 0.1, extra: 1, constructor: 'x' })));
    expect(decoded?.lr).toBe(0.1);
    expect(decoded).not.toHaveProperty('extra');
  });
});
