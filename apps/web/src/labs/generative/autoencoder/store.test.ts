import { DEFAULT_AE_LAB, IMAGE_SIZE } from '@sim/deep';
import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import { IMAGINED, STRIP_STEPS, networkInput, resetAeSession, useAeStore } from './store';

const state = () => useAeStore.getState();
const PIXELS = IMAGE_SIZE * IMAGE_SIZE;

async function waitFor(condition: () => boolean, timeout = 8000) {
  const start = Date.now();
  while (!condition()) {
    if (Date.now() - start > timeout) throw new Error('waitFor timed out');
    await new Promise((resolve) => setTimeout(resolve, 10));
  }
}

const fast = { perClass: 10, batchSize: 20 } as const;

describe('autoencoder store', () => {
  beforeEach(() => {
    resetAeSession();
    state().init();
  });
  afterEach(() => state().setRunning(false));

  it('starts with a sample digit encoded, decoded and plotted', () => {
    expect(state().source).toBe('sample');
    expect(state().code?.mu).toHaveLength(DEFAULT_AE_LAB.latent);
    expect(state().code?.logvar).toBeNull(); // plain autoencoder
    expect(state().recon).toHaveLength(PIXELS);
    expect(state().scatter?.count).toBe(300);
    expect(state().probeImage).toHaveLength(PIXELS);
    expect(state().strip).toHaveLength(STRIP_STEPS * PIXELS);
    expect(state().imagined).toHaveLength(IMAGINED * PIXELS);
    expect(state().pair?.a.label).toBe(3);
    expect(state().pair?.b.label).toBe(8);
  });

  it('drawing replaces the sample; clearing leaves nothing to encode', () => {
    state().beginStroke(4, 4);
    state().moveStroke(11, 11);
    state().endStroke();
    expect(state().source).toBe('drawn');
    expect(state().sampleLabel).toBeNull();
    expect(state().image[8 * IMAGE_SIZE + 8]).toBeGreaterThan(0.5);
    expect(state().code).not.toBeNull();

    const before = Float64Array.from(state().image);
    state().moveStroke(1, 1); // no stroke in progress
    expect(state().image).toEqual(before);

    state().clear();
    expect(state().source).toBe('blank');
    expect(state().code).toBeNull();
    expect(state().recon).toBeNull();
  });

  it('loads samples by index and by digit', () => {
    state().loadDigit(7);
    expect(state().sampleLabel).toBe(7);
    state().loadSample(0);
    expect(state().source).toBe('sample');
    state().randomSample();
    expect(state().source).toBe('sample');
    expect(() => state().loadSample(99999)).toThrow();
  });

  it('input noise changes what the encoder sees, but is stable and reversible', () => {
    const clean = Float64Array.from(state().input);
    state().setInputNoise(0.3);
    const noisy = Float64Array.from(state().input);
    expect(noisy).not.toEqual(clean);
    state().setInputNoise(0.3);
    expect(state().input).toEqual(noisy); // the same pattern every time
    state().setInputNoise(0);
    expect(state().input).toEqual(clean);
    expect(noisy.every((v) => v >= 0 && v <= 1)).toBe(true);
  });

  it('networkInput centres and adds noise without touching the original', () => {
    const image = new Float64Array(PIXELS);
    image[200] = 1;
    const copy = Float64Array.from(image);
    expect(networkInput(image, false, 0)).toEqual(image);
    expect(networkInput(image, true, 0)).not.toEqual(image);
    expect(networkInput(image, false, 0.2)).not.toEqual(image);
    expect(image).toEqual(copy);
  });

  it('the probe decodes any point of the latent space', () => {
    const before = Float64Array.from(state().probeImage ?? []);
    state().setProbePlane(0, 1, 2, -1.5);
    expect(Array.from(state().probe)).toEqual([2, -1.5]);
    expect(state().probeImage).not.toEqual(before);
    state().setProbe(0, 0.5);
    expect(state().probe[0]).toBe(0.5);
    state().probeToImage();
    expect(Array.from(state().probe)).toEqual(Array.from(state().code?.mu ?? []));
    state().setProbe(9, 1); // out of range: ignored
    expect(state().probe).toHaveLength(2);
  });

  it('interpolation endpoints can be changed by digit', () => {
    const before = Float64Array.from(state().strip ?? []);
    state().setPairDigit('a', 1);
    expect(state().pair?.a.label).toBe(1);
    expect(state().pair?.b.label).toBe(8);
    expect(state().strip).not.toEqual(before);
    state().setPairDigit('b', null);
    expect(state().pair?.b.label).toBeGreaterThanOrEqual(0);
  });

  it('imagining draws new codes from the prior', () => {
    const before = Float64Array.from(state().imagined ?? []);
    state().imagine();
    expect(state().imagined).not.toEqual(before);
  });

  it('trains an epoch on demand', () => {
    state().patch(fast);
    state().stepEpoch();
    expect(state().epoch).toBe(1);
    expect(state().history).toHaveLength(1);
    expect(state().running).toBe(false);
  });

  it('switching to a VAE and resizing the latent rebuilds everything', () => {
    state().patch(fast);
    state().stepEpoch();
    state().patch({ mode: 'vae', latent: 4 });
    expect(state().history).toHaveLength(0);
    expect(state().code?.mu).toHaveLength(4);
    expect(state().code?.logvar).toHaveLength(4);
    expect(state().probe).toHaveLength(4);
    expect(state().scatter?.mu).toHaveLength(300 * 4);
    state().patch({ beta: 2 }); // live
    expect(state().config.beta).toBe(2);
  });

  it('rejects invalid settings with a notice', () => {
    state().patch({ latent: 1 });
    expect(state().notice).toMatch(/Latent size/);
    expect(state().config.latent).toBe(DEFAULT_AE_LAB.latent);
    state().patch({ latent: 3 });
    expect(state().notice).toBeNull();
  });

  it('runs continuously, completes epochs, and stops when paused', async () => {
    state().patch(fast);
    state().setRunning(true);
    await waitFor(() => state().history.length >= 2);
    state().setRunning(false);
    const settled = state().epoch;
    await new Promise((resolve) => setTimeout(resolve, 150));
    expect(state().epoch).toBe(settled);
    state().history.forEach((s, i) => expect(s.epoch).toBe(i + 1));
  });

  it('reset restarts from the same initial weights', () => {
    state().patch(fast);
    state().stepEpoch();
    const first = state().history[0]?.trainRecon;
    state().reset();
    expect(state().history).toHaveLength(0);
    state().stepEpoch();
    expect(state().history[0]?.trainRecon).toBe(first);
  });
});
