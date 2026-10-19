import { describe, expect, it } from 'vitest';
import { PlaygroundHost } from './host';
import { DEFAULT_CONFIG, PlaygroundSession, type PlaygroundConfig } from './session';

const config = (patch: Partial<PlaygroundConfig> = {}): PlaygroundConfig => ({
  ...JSON.parse(JSON.stringify(DEFAULT_CONFIG)),
  classes: 3,
  ...patch,
});

const make = (patch: Partial<PlaygroundConfig> = {}) => {
  const s = new PlaygroundSession(config(patch));
  s.step(5);
  return s;
};

describe('PlaygroundSession.inspect', () => {
  it('reports the forward pass: probabilities sum to 1 and shapes follow the architecture', () => {
    const info = make().inspect(3);
    const output = info.activations.at(-1) as number[];
    expect(output).toHaveLength(3);
    expect(output.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 10);
    expect(info.activations).toHaveLength(4); // inputs + 3 layers
    expect(info.activations[0]).toHaveLength(2);
    expect(info.preActivations).toHaveLength(3);
    expect(info.targets.reduce((a, b) => a + b, 0)).toBe(1); // one-hot
  });

  it('loss is -log of the target probability and the output delta is ŷ − y', () => {
    const info = make().inspect(7);
    const output = info.activations.at(-1) as number[];
    const target = info.targets.indexOf(1);
    expect(info.loss).toBeCloseTo(-Math.log(output[target] as number), 10);
    const delta = info.deltas.at(-1) as number[];
    output.forEach((p, k) => expect(delta[k]).toBeCloseTo(p - (info.targets[k] as number), 12));
    expect(info.point.target).toBe(target);
  });

  it('weight gradients are the outer product of a layer input and its delta', () => {
    const info = make().inspect(2);
    info.weightGrads.forEach((W, l) => {
      const input = info.layerInputs[l] as number[];
      const delta = info.deltas[l] as number[];
      W.forEach((row, i) =>
        row.forEach((g, j) =>
          expect(g).toBeCloseTo((input[i] as number) * (delta[j] as number), 12),
        ),
      );
      info.biasGrads[l]?.forEach((g, j) => expect(g).toBeCloseTo(delta[j] as number, 12));
    });
  });

  it('with skip connections deeper layers also receive the raw inputs', () => {
    const info = make({ skipConnections: true }).inspect(1);
    expect(info.layerInputs[0]).toHaveLength(2);
    expect(info.layerInputs[1]).toHaveLength(4 + 2);
    expect(info.layerInputs[2]).toHaveLength(3 + 2);
  });

  it('works for regression', () => {
    const info = new PlaygroundSession(config({ dataset: 'plane', noise: 0, hidden: [] })).inspect(
      0,
    );
    expect(info.targets).toHaveLength(1);
    expect(info.deltas[0]).toHaveLength(1);
    expect(info.loss).toBeGreaterThanOrEqual(0);
  });

  it('is read-only: it does not change the training trajectory', () => {
    const a = new PlaygroundSession(config());
    const b = new PlaygroundSession(config());
    a.step(3);
    b.step(3);
    a.inspect(5);
    a.inspect(11);
    expect(a.step(4)).toEqual(b.step(4));
  });

  it('can pick the sample the model currently gets most wrong', () => {
    const s = make();
    const info = s.inspect('worst');
    const lossOf = (i: number) => s.inspect(i).loss;
    for (let i = 0; i < s.trainSize; i += 7) {
      expect(lossOf(i)).toBeLessThanOrEqual(info.loss + 1e-12);
    }
    expect(info.sample).toBe(s.worstSample());
  });

  it('validates the sample index and the protocol exposes it', () => {
    const s = make();
    expect(() => s.inspect(-1)).toThrow(/Sample/);
    expect(() => s.inspect(s.trainSize)).toThrow(/Sample/);

    const host = new PlaygroundHost();
    host.handle({ id: 1, type: 'init', config: config() });
    const res = host.handle({ id: 2, type: 'inspect', sample: 4 });
    expect(res).toMatchObject({ id: 2, type: 'inspection' });
    if (res.type === 'inspection') expect(res.trainSize).toBeGreaterThan(50);
    expect(host.handle({ id: 3, type: 'inspect', sample: 99999 })).toMatchObject({
      type: 'error',
    });
  });
});
