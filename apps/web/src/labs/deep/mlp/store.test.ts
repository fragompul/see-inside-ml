import { DEFAULT_CONFIG } from '@sim/deep';
import type * as ClientModule from '@/shared/playground/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

// Run the store against the real engine host, but in-process instead of in a Web Worker.
vi.mock('@/shared/playground/client', async (importOriginal) => {
  const original = await importOriginal<typeof ClientModule>();
  const { inProcessTransport } = await import('@/shared/playground/testing');
  return {
    ...original,
    PlaygroundClient: class extends original.PlaygroundClient {
      constructor() {
        super(inProcessTransport());
      }
    },
  };
});

const { useMlpStore } = await import('./store');

const state = () => useMlpStore.getState();

async function waitFor(condition: () => boolean, timeout = 4000) {
  const start = Date.now();
  while (!condition()) {
    if (Date.now() - start > timeout) throw new Error('waitFor timed out');
    await new Promise((resolve) => setTimeout(resolve, 5));
  }
}

describe('mlp store', () => {
  beforeEach(async () => {
    state().boot(DEFAULT_CONFIG);
    await waitFor(() => state().status === 'ready');
  });

  afterEach(() => state().dispose());

  it('boots into a ready scene with an initial snapshot', () => {
    expect(state().scene?.layerSizes).toEqual([2, 4, 3, 2]);
    expect(state().snapshot?.boundary).not.toBeNull();
    expect(state().history).toEqual([]);
  });

  it('step() advances exactly one epoch and pauses a running loop', async () => {
    state().step();
    await waitFor(() => state().history.length === 1);
    expect(state().history[0]?.epoch).toBe(1);
    expect(state().running).toBe(false);
  });

  it('applies hyper-parameters live and rebuilds on data changes', async () => {
    state().step();
    await waitFor(() => state().history.length === 1);
    const version = state().sceneVersion;

    state().patch({ lr: 0.1 });
    await waitFor(() => state().snapshot !== null);
    expect(state().config.lr).toBe(0.1);
    expect(state().history).toHaveLength(1); // progress kept
    expect(state().sceneVersion).toBe(version);

    state().patch({ dataset: 'moons' });
    await waitFor(() => state().sceneVersion > version);
    expect(state().history).toEqual([]);
    expect(state().scene?.numClasses).toBe(2);
  });

  it('coalesces rapid patches into the final value', async () => {
    for (const lr of [0.001, 0.003, 0.01, 0.1]) state().patch({ lr });
    expect(state().config.lr).toBe(0.1); // optimistic
    await waitFor(() => state().notice === null && state().snapshot !== null);
    state().step();
    await waitFor(() => state().history.length === 1);
    expect(state().history[0]?.lr).toBe(0.1);
  });

  it('restores the previous setup when the engine rejects a change', async () => {
    state().patch({ lr: -1 });
    await waitFor(() => state().notice !== null);
    expect(state().notice).toMatch(/Learning rate/);
    expect(state().config.lr).toBe(DEFAULT_CONFIG.lr);
    expect(state().status).toBe('ready'); // still usable
    state().step();
    await waitFor(() => state().history.length === 1);
  });

  it('runs continuously and stops when paused', async () => {
    state().setRunning(true);
    await waitFor(() => state().history.length >= 10);
    state().setRunning(false);
    await new Promise((resolve) => setTimeout(resolve, 80)); // let the in-flight step land
    const settled = state().history.length;
    await new Promise((resolve) => setTimeout(resolve, 120));
    expect(state().history.length).toBe(settled);
    // epochs are contiguous: nothing dropped or duplicated by the loop
    state().history.forEach((s, i) => expect(s.epoch).toBe(i + 1));
  });

  it('reset() restarts training from the same initial weights', async () => {
    state().step();
    await waitFor(() => state().history.length === 1);
    const first = state().history[0]?.trainLoss;
    state().reset();
    await waitFor(() => state().history.length === 0);
    state().step();
    await waitFor(() => state().history.length === 1);
    expect(state().history[0]?.trainLoss).toBe(first);
  });

  it('presets load a fresh setup and start training; manual edits clear the lesson', async () => {
    state().applyPreset('x', { ...DEFAULT_CONFIG, dataset: 'moons' }, 10);
    expect(state().activePreset).toBe('x');
    expect(state().running).toBe(true);
    expect(state().speed).toBe(10);
    await waitFor(() => state().history.length >= 3);
    expect(state().config.dataset).toBe('moons');
    state().patch({ noise: 0.3 });
    expect(state().activePreset).toBeNull();
  });

  describe('step inspector', () => {
    it('freezes training and loads the worst-predicted sample at stage 0', async () => {
      state().step();
      await waitFor(() => state().history.length === 1);
      state().startInspect();
      expect(state().inspecting).toBe(true);
      await waitFor(() => state().inspection !== null);
      expect(state().inspectStage).toBe(0);
      expect(typeof state().inspectSample).toBe('number');
      expect(state().inspection?.activations.length).toBe(state().scene?.layerSizes.length ?? 0);
    });

    it('stage navigation is clamped to the four stages', async () => {
      state().startInspect();
      await waitFor(() => state().inspection !== null);
      state().setStage(9);
      expect(state().inspectStage).toBe(3);
      state().setStage(-4);
      expect(state().inspectStage).toBe(0);
    });

    it('picking another sample re-inspects it', async () => {
      state().startInspect(0);
      await waitFor(() => state().inspection?.sample === 0);
      state().pickSample(5);
      await waitFor(() => state().inspection?.sample === 5);
    });

    it('refreshes when the setup changes (learning rate, reset)', async () => {
      state().startInspect(2);
      await waitFor(() => state().inspection !== null);
      state().patch({ lr: 0.1 });
      await waitFor(() => state().inspection?.lr === 0.1);
      state().reset();
      await waitFor(() => state().inspection !== null && state().history.length === 0);
      expect(state().inspecting).toBe(true);
    });

    it('training or stepping closes the inspector (the weights would change under it)', async () => {
      state().startInspect();
      await waitFor(() => state().inspection !== null);
      state().setRunning(true);
      expect(state().inspecting).toBe(false);
      expect(state().inspection).toBeNull();
      state().setRunning(false);

      state().startInspect();
      await waitFor(() => state().inspection !== null);
      state().step();
      expect(state().inspecting).toBe(false);
    });

    it('stopInspect clears it', async () => {
      state().startInspect();
      await waitFor(() => state().inspection !== null);
      state().stopInspect();
      expect(state().inspection).toBeNull();
    });
  });
});
