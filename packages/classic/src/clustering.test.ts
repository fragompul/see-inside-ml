import { Matrix, adjustedRandIndex, createRng, makeClassification } from '@sim/core';
import { describe, expect, it } from 'vitest';
import density from '../fixtures/density.json';
import hier from '../fixtures/hierarchical.json';
import {
  Dbscan,
  MeanShift,
  estimateBandwidth,
  kDistances,
  optics,
  opticsClusters,
} from './density';
import {
  copheneticCorrelation,
  cophenetic,
  cutAtHeight,
  cutTree,
  dendrogramOrder,
  linkage,
  pairwiseDistances,
  type Linkage,
} from './hierarchical';

describe('hierarchical clustering agrees with SciPy', () => {
  const x = Matrix.from(hier.x);
  const n = x.rows;
  for (const [method, ref] of Object.entries(hier.methods)) {
    it(`${method} linkage: merge heights, sizes, dendrogram order, cophenetic distances and cuts`, () => {
      const merges = linkage(x, method as Linkage);
      merges.forEach((m, i) => {
        const r = ref.z[i] as number[];
        expect(m.a).toBe(r[0]);
        expect(m.b).toBe(r[1]);
        expect(Math.abs(m.height - (r[2] as number))).toBeLessThan(1e-9);
        expect(m.size).toBe(r[3]);
      });
      expect(dendrogramOrder(merges, n)).toEqual(ref.leaves);
      const c = cophenetic(merges, n);
      ref.cophenet.forEach((v, i) => expect(Math.abs((c[i] as number) - v)).toBeLessThan(1e-9));
      expect(adjustedRandIndex(cutTree(merges, n, 3), ref.cut3)).toBeCloseTo(1, 12);
      expect(
        adjustedRandIndex(cutAtHeight(merges, n, ref.median_height), ref.cut_height),
      ).toBeCloseTo(1, 12);
    });
  }

  it('the cophenetic correlation says average linkage keeps the geometry well', () => {
    const d = pairwiseDistances(x);
    expect(copheneticCorrelation(d, n, linkage(x, 'average'))).toBeGreaterThan(0.7);
  });

  it('single linkage chains: on two crescents it recovers them, Ward does not', () => {
    const moons = makeClassification('moons', 160, 0.06, createRng(4));
    const single = cutTree(linkage(moons.x, 'single'), 160, 2);
    const ward = cutTree(linkage(moons.x, 'ward'), 160, 2);
    expect(adjustedRandIndex(moons.y, single)).toBeGreaterThan(0.9);
    expect(adjustedRandIndex(moons.y, ward)).toBeLessThan(0.6);
  });

  it('merge heights never decrease for these linkages and cutting at k gives k clusters', () => {
    for (const method of ['single', 'complete', 'average', 'ward'] as const) {
      const m = linkage(x, method);
      for (let i = 1; i < m.length; i++)
        expect(m[i]!.height).toBeGreaterThanOrEqual(m[i - 1]!.height - 1e-12);
      for (const k of [1, 2, 5, 12]) expect(new Set(cutTree(m, n, k)).size).toBe(k);
    }
  });
});

describe('DBSCAN agrees with scikit-learn', () => {
  for (const ref of density.dbscan) {
    it(`${ref.data}, eps = ${ref.eps}, min_samples = ${ref.min_samples}`, () => {
      const x = Matrix.from(ref.data === 'moons' ? density.moons : density.blobs);
      const m = new Dbscan({ eps: ref.eps, minSamples: ref.min_samples }).fit(x);
      expect(Array.from(m.labels)).toEqual(ref.labels);
      expect(m.coreIndices).toEqual(ref.core);
    });
  }

  it('finds the crescents as two clusters and calls isolated points noise', () => {
    const x = Matrix.from(density.moons);
    const m = new Dbscan({ eps: 0.35, minSamples: 5 }).fit(x);
    expect(m.clusters).toBe(2);
    const far = Matrix.from([...density.moons, [8, 8]]);
    const withOutlier = new Dbscan({ eps: 0.35, minSamples: 5 }).fit(far);
    expect(withOutlier.labels[withOutlier.labels.length - 1]).toBe(-1);
  });

  it('the k-distance plot rises sharply where the noise starts', () => {
    const k = kDistances(Matrix.from([...density.blobs, [9, 9], [-8, 7]]), 5);
    expect(k[k.length - 1]).toBeGreaterThan(5 * (k[Math.floor(k.length / 2)] as number));
  });
});

describe('OPTICS agrees with scikit-learn', () => {
  for (const ref of density.optics) {
    it(`${ref.data}: ordering, reachability, core distances and extraction`, () => {
      const x = Matrix.from(ref.data === 'moons' ? density.moons : density.blobs);
      const r = optics(x, ref.min_samples);
      expect(r.ordering).toEqual(ref.ordering);
      ref.reachability.forEach((v, i) => {
        if (v === null) expect(r.reachability[i]).toBe(Infinity);
        else expect(Math.abs((r.reachability[i] as number) - v)).toBeLessThan(1e-9);
      });
      ref.core.forEach((v, i) =>
        expect(Math.abs((r.coreDistances[i] as number) - (v as number))).toBeLessThan(1e-9),
      );
      expect(Array.from(r.predecessor)).toEqual(ref.predecessor);
      for (const [eps, labels] of Object.entries(ref.extract))
        expect(Array.from(opticsClusters(r, Number(eps)))).toEqual(labels);
    });
  }
});

describe('mean shift agrees with scikit-learn', () => {
  for (const ref of density.meanshift) {
    it(`${ref.data}, bandwidth ${ref.bandwidth.toFixed(3)}${ref.estimated ? ' (estimated)' : ''}`, () => {
      const x = Matrix.from(ref.data === 'moons' ? density.moons : density.blobs);
      const bw = ref.estimated ? estimateBandwidth(x, 0.3) : ref.bandwidth;
      expect(Math.abs(bw - ref.bandwidth)).toBeLessThan(1e-9);
      const m = new MeanShift({ bandwidth: bw }).fit(x);
      expect(m.clusters).toBe(ref.centers.length);
      ref.centers.forEach((c, i) =>
        c.forEach((v, d) => expect(Math.abs(m.centers.get(i, d) - v)).toBeLessThan(1e-6)),
      );
      expect(Array.from(m.labels)).toEqual(ref.labels);
    });
  }

  it('every seed climbs to a peak: the path ends where one more shift goes nowhere', () => {
    const x = Matrix.from(density.blobs);
    const { path } = MeanShift.climb(x, x.data.subarray(0, 2), 1);
    expect(path.length).toBeGreaterThan(1);
    const end = path[path.length - 1] as number[];
    const again = MeanShift.climb(x, end, 1);
    expect(
      Math.hypot(
        ...(again.path[again.path.length - 1] as number[]).map((v, k) => v - (end[k] as number)),
      ),
    ).toBeLessThan(2e-3);
  });
});
