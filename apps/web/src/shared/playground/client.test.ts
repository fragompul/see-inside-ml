import { DEFAULT_CONFIG } from '@sim/deep';
import { describe, expect, it } from 'vitest';
import { PlaygroundClient } from './client';
import { inProcessTransport } from './testing';

describe('PlaygroundClient', () => {
  it('drives a full session through the typed API', async () => {
    const client = new PlaygroundClient(inProcessTransport());
    const scene = await client.init(DEFAULT_CONFIG, { resolution: 10 });
    expect(scene.scene.problem).toBe('classification');
    expect(scene.snapshot.boundary?.resolution).toBe(10);

    const progress = await client.step(3);
    expect(progress.stats.map((s) => s.epoch)).toEqual([1, 2, 3]);

    const live = await client.update({ lr: 0.1 });
    expect(live.type).toBe('progress');
    const rebuilt = await client.update({ dataset: 'moons' });
    expect(rebuilt.type).toBe('scene');
    await expect(client.reset()).resolves.toMatchObject({ type: 'scene' });
  });

  it('analyses the current weights: a loss-landscape slice and weight histograms', async () => {
    const client = new PlaygroundClient(inProcessTransport());
    await client.init(DEFAULT_CONFIG);
    await client.step(20);
    const res = await client.analyse({ landscape: { steps: 7, kind: 'path' }, histograms: 12 });
    expect(res.epoch).toBe(20);
    expect(res.landscape?.values).toHaveLength(49);
    expect(res.landscape?.path?.a.length).toBeGreaterThan(5);
    expect(res.histograms?.length).toBe(3);
    expect(res.histograms?.[0]?.weights.counts).toHaveLength(12);
    const only = await client.analyse({ histograms: 8 });
    expect(only.landscape).toBeNull();
    expect(only.histograms).not.toBeNull();
    // analysing never changes the weights: training continues exactly as if it had not happened
    const a = new PlaygroundClient(inProcessTransport());
    const b = new PlaygroundClient(inProcessTransport());
    await a.init(DEFAULT_CONFIG);
    await b.init(DEFAULT_CONFIG);
    await a.step(5);
    await b.step(5);
    await a.analyse({ landscape: { steps: 5 }, histograms: 6 });
    const sa = await a.step(5);
    const sb = await b.step(5);
    expect(sa.stats.map((s) => s.trainLoss)).toEqual(sb.stats.map((s) => s.trainLoss));
  });

  it('matches concurrent requests to their own responses', async () => {
    const client = new PlaygroundClient(inProcessTransport());
    await client.init(DEFAULT_CONFIG);
    const [a, b] = await Promise.all([client.step(1), client.step(2)]);
    expect(a.stats).toHaveLength(1);
    expect(b.stats).toHaveLength(2);
    expect(b.stats[1]?.epoch).toBe(3);
  });

  it('rejects when the engine reports an error', async () => {
    const client = new PlaygroundClient(inProcessTransport());
    await expect(client.step(1)).rejects.toThrow(/init/);
    await client.init(DEFAULT_CONFIG);
    await expect(client.update({ lr: -5 })).rejects.toThrow(/Learning rate/);
    await expect(client.step(1)).resolves.toBeDefined();
  });

  it('rejects pending calls when the worker crashes or the client is disposed', async () => {
    const t = inProcessTransport();
    const client = new PlaygroundClient(t);
    t.postMessage = () => {}; // simulate a worker that never answers
    const hanging = client.init(DEFAULT_CONFIG);
    t.crash('boom');
    await expect(hanging).rejects.toThrow('boom');

    const hanging2 = client.init(DEFAULT_CONFIG);
    client.dispose();
    await expect(hanging2).rejects.toThrow(/disposed/);
    await expect(client.step(1)).rejects.toThrow(/disposed/);
  });
});
